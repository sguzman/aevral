//! C009 block-based forecasts, causal linear decompositions and TiDE-style encoder.
use super::residual::sample;
use crate::adam::Adam;
#[derive(Clone)]
struct Linear{n:usize,m:usize,bias:bool,w:Vec<f64>}
impl Linear{
    fn new(n:usize,m:usize,bias:bool,s:&mut u64)->Self{
        let stride=n+usize::from(bias);let scale=(6./(n+m)as f64).sqrt();
        Self{n,m,bias,w:(0..m*stride).map(|i|if bias&&i%stride==n{0.}else{sample(s,scale)}).collect()}
    }
    fn forward(&self,x:&[f64])->Vec<f64>{
        assert_eq!(x.len(),self.n);let stride=self.n+usize::from(self.bias);
        self.w.chunks_exact(stride).map(|r|{let b=if self.bias{r[self.n]}else{0.};b+r[..self.n].iter().zip(x).map(|(a,b)|a*b).sum::<f64>()}).collect()
    }
    fn backward(&self,x:&[f64],dy:&[f64])->(Vec<f64>,Vec<f64>){
        assert_eq!(x.len(),self.n);assert_eq!(dy.len(),self.m);let stride=self.n+usize::from(self.bias);
        let mut dx=vec![0.;self.n];let mut g=vec![0.;self.w.len()];
        for(j,r)in self.w.chunks_exact(stride).enumerate(){for i in 0..self.n{dx[i]+=r[i]*dy[j];g[j*stride+i]=x[i]*dy[j];}if self.bias{g[j*stride+self.n]=dy[j];}}(dx,g)
    }
}
fn add(a:&mut[f64],b:&[f64]){assert_eq!(a.len(),b.len());for(a,b)in a.iter_mut().zip(b){*a+=b;}}
fn relu(x:Vec<f64>)->Vec<f64>{x.into_iter().map(|v|v.max(0.)).collect()}
fn relu_grad(d:&mut[f64],h:&[f64]){for(d,h)in d.iter_mut().zip(h){if *h<=0.{*d=0.;}}}
fn tanh_grad(d:&mut[f64],h:&[f64]){for(d,h)in d.iter_mut().zip(h){*d*=1.-h*h;}}
fn loss(y:&[f64],t:&[f64])->(f64,Vec<f64>){assert_eq!(y.len(),24);assert_eq!(t.len(),24);let e:Vec<_>=y.iter().zip(t).map(|(a,b)|a-b).collect();(e.iter().map(|v|v*v).sum::<f64>()/24.,e.iter().map(|v|v/12.).collect())}
fn pool(x:&[f64],p:usize)->Vec<f64>{assert_eq!(x.len(),72);x.chunks_exact(p).map(|v|v.iter().sum::<f64>()/p as f64).collect()}
fn interpolation(k:usize,t:usize)->(usize,usize,f64){
    let z=t*(k-1);let a=z/23;let b=(a+1).min(k-1);(a,b,(z%23)as f64/23.)
}
fn interp(x:&[f64])->Vec<f64>{(0..24).map(|t|{let(a,b,w)=interpolation(x.len(),t);x[a]*(1.-w)+x[b]*w}).collect()}
fn interp_grad(k:usize,dy:&[f64])->Vec<f64>{let mut d=vec![0.;k];for(t,v)in dy.iter().enumerate(){let(a,b,w)=interpolation(k,t);d[a]+=v*(1.-w);d[b]+=v*w;}d}
fn trend(x:&[f64],k:usize)->Vec<f64>{(0..72).map(|i|(0..k).map(|j|x[(i as isize+j as isize-(k/2)as isize).clamp(0,71)as usize]).sum::<f64>()/k as f64).collect()}
#[derive(Clone)]
enum Form{Blocks(Vec<(usize,usize)>),DLinear(usize),NLinear,Tide}
#[derive(Clone)]
pub(super) struct Structured{form:Form,layers:Vec<Linear>}
struct BlockCache{input:Vec<f64>,h1:Vec<f64>,h2:Vec<f64>}
impl Structured{
    pub(super) fn new(label:&str,seed:u64)->Option<Self>{
        let mut s=seed.max(1);let mut layers=Vec::new();
        let form=match label{
            "nbeats-w64-b2"|"nbeats-w128-b2"|"nbeats-w64-b4"|"nhits-w64"|"nhits-w128"=>{
                let w=if label.contains("128"){128}else{64};
                let specs=if label.starts_with("nhits"){vec![(6,4),(3,8),(1,24)]}else{vec![(1,24);if label.ends_with("b4"){4}else{2}]};
                for &(p,k) in &specs{layers.push(Linear::new(72/p+8,w,true,&mut s));layers.push(Linear::new(w,w,true,&mut s));layers.push(Linear::new(w,72,true,&mut s));layers.push(Linear::new(w,k,true,&mut s));}
                Form::Blocks(specs)
            }
            "dlinear-k7"|"dlinear-k25"=>{layers.push(Linear::new(72,24,true,&mut s));layers.push(Linear::new(72,24,true,&mut s));layers.push(Linear::new(8,24,false,&mut s));Form::DLinear(if label.ends_with("25"){25}else{7})}
            "nlinear-direct"|"nlinear-lead-bias"=>{layers.push(Linear::new(80,24,label.ends_with("bias"),&mut s));Form::NLinear}
            "tide-w64"|"tide-w128"=>{let w=if label.ends_with("128"){128}else{64};for(n,m,b)in [(80,w,true),(w,w,true),(w,w,true),(w,32,true),(32,w,true),(w,w,true),(w,w,true),(w,24,true),(72,24,false)]{layers.push(Linear::new(n,m,b,&mut s));}Form::Tide}
            _=>return None
        };Some(Self{form,layers})
    }
    pub(super) fn parameters(&self)->usize{self.layers.iter().map(|l|l.w.len()).sum()}
    pub(super) fn finite(&self)->bool{self.layers.iter().flat_map(|l|&l.w).all(|v|v.is_finite())}
    fn blocks(&self,x:&[f64],specs:&[(usize,usize)])->(Vec<f64>,Vec<BlockCache>){
        let mut r=x[..72].to_vec();let mut y=vec![0.;24];let mut cache=Vec::new();
        for(i,&(p,_))in specs.iter().enumerate(){let l=&self.layers[4*i..4*i+4];
            let mut input=pool(&r,p);input.extend_from_slice(&x[72..]);let h1=relu(l[0].forward(&input));let h2=relu(l[1].forward(&h1));
            let b=l[2].forward(&h2);for(r,b)in r.iter_mut().zip(b){*r-=b;}add(&mut y,&interp(&l[3].forward(&h2)));
            cache.push(BlockCache{input,h1,h2});
        }(y,cache)
    }
    fn tide(&self,x:&[f64])->(Vec<f64>,Vec<Vec<f64>>){
        let l=&self.layers;let h0:Vec<_>=l[0].forward(x).into_iter().map(f64::tanh).collect();
        let h1=relu(l[1].forward(&h0));let mut h2=l[2].forward(&h1);add(&mut h2,&h0);
        let h3=l[3].forward(&h2);let h4:Vec<_>=l[4].forward(&h3).into_iter().map(f64::tanh).collect();
        let h5=relu(l[5].forward(&h4));let mut h6=l[6].forward(&h5);add(&mut h6,&h4);
        let mut y=l[7].forward(&h6);add(&mut y,&l[8].forward(&x[..72]));(y,vec![h0,h1,h2,h3,h4,h5,h6])
    }
    fn linear_inputs(&self,x:&[f64])->Vec<Vec<f64>>{match self.form{
        Form::DLinear(k)=>{let a=trend(x,k);let b=x[..72].iter().zip(&a).map(|(x,a)|x-a).collect();vec![a,b,x[72..].to_vec()]}
        Form::NLinear=>{let mut a=x.to_vec();for v in &mut a[..72]{*v-=x[71];}vec![a]}
        _=>unreachable!()
    }}
    pub(super) fn predict(&self,x:&[f64])->Vec<f64>{assert_eq!(x.len(),80);match &self.form{
        Form::Blocks(s)=>self.blocks(x,s).0,Form::Tide=>self.tide(x).0,
        _=>{let a=self.linear_inputs(x);let mut y=vec![if matches!(self.form,Form::NLinear){x[71]}else{0.};24];for(l,v)in self.layers.iter().zip(a){add(&mut y,&l.forward(&v));}y}
    }}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,Vec<Vec<f64>>){self.gradients_with(x,target,loss)}
    fn gradients_with(&self,x:&[f64],target:&[f64],objective:impl Fn(&[f64],&[f64])->(f64,Vec<f64>))->(f64,Vec<Vec<f64>>){
        let mut g:Vec<_>=self.layers.iter().map(|l|vec![0.;l.w.len()]).collect();let l=&self.layers;
        match &self.form{
            Form::Blocks(specs)=>{let(y,c)=self.blocks(x,specs);let(value,dy)=objective(&y,target);let mut dr=vec![0.;72];
                for i in (0..specs.len()).rev(){let(p,k)=specs[i];let a=&c[i];let j=4*i;
                    let db:Vec<_>=dr.iter().map(|v|-v).collect();let(mut dh,g2)=l[j+2].backward(&a.h2,&db);g[j+2]=g2;
                    let(d,g3)=l[j+3].backward(&a.h2,&interp_grad(k,&dy));g[j+3]=g3;add(&mut dh,&d);relu_grad(&mut dh,&a.h2);
                    let(mut d,g1)=l[j+1].backward(&a.h1,&dh);g[j+1]=g1;relu_grad(&mut d,&a.h1);
                    let(d,g0)=l[j].backward(&a.input,&d);g[j]=g0;
                    for t in 0..72{dr[t]+=d[t/p]/p as f64;}
                }(value,g)
            }
            Form::Tide=>{let(y,h)=self.tide(x);let(value,dy)=objective(&y,target);
                g[8]=l[8].backward(&x[..72],&dy).1;let(mut d,g7)=l[7].backward(&h[6],&dy);g[7]=g7;
                let(mut z,g6)=l[6].backward(&h[5],&d);g[6]=g6;relu_grad(&mut z,&h[5]);let(z,g5)=l[5].backward(&h[4],&z);g[5]=g5;add(&mut d,&z);tanh_grad(&mut d,&h[4]);
                let(d,g4)=l[4].backward(&h[3],&d);g[4]=g4;let(mut d,g3)=l[3].backward(&h[2],&d);g[3]=g3;
                let(mut z,g2)=l[2].backward(&h[1],&d);g[2]=g2;relu_grad(&mut z,&h[1]);let(z,g1)=l[1].backward(&h[0],&z);g[1]=g1;add(&mut d,&z);tanh_grad(&mut d,&h[0]);g[0]=l[0].backward(x,&d).1;(value,g)
            }
            _=>{let(value,dy)=objective(&self.predict(x),target);for(i,a)in self.linear_inputs(x).iter().enumerate(){g[i]=l[i].backward(a,&dy).1;}(value,g)}
        }
    }
    pub(super) fn step(&mut self,x:&[f64],y:&[f64],opt:&mut Adam)->(f64,bool){
        self.step_with(x,y,opt,loss)
    }
    pub(super) fn step_with(&mut self,x:&[f64],y:&[f64],opt:&mut Adam,objective:impl Fn(&[f64],&[f64])->(f64,Vec<f64>))->(f64,bool){
        let(value,g)=self.gradients_with(x,y,objective);let norm=g.iter().flatten().map(|v|v*v).sum::<f64>().sqrt();
        if !value.is_finite()||!norm.is_finite(){return(f64::NAN,false)}
        let clipped=matches!(self.form,Form::Blocks(_)|Form::Tide)&&norm>5.;let scale=if clipped{5./norm}else{1.};
        opt.begin();for(l,g)in self.layers.iter_mut().zip(g){for(w,d)in l.w.iter_mut().zip(g){opt.update(w,d*scale);}}opt.finish();(value,clipped)
    }
    /// C012 MAE: average example gradients then clip the batch gradient as in the original block optimizer.
    pub(super) fn optimization_mae_batch(&mut self,rows:&[(&[f64],&[f64])],opt:&mut Adam,decay:f64)->(f64,bool){
        assert!(!rows.is_empty()&&decay>=0.&&decay.is_finite());
        let mut g:Vec<_>=self.layers.iter().map(|l|vec![0.;l.w.len()]).collect();let mut loss=0.;
        for (x,y) in rows{
            let(v,gr)=self.gradients_with(x,y,|p,t|{
                let e:Vec<_>=p.iter().zip(t).map(|(p,t)|p-t).collect();
                (e.iter().map(|v|v.abs()).sum::<f64>()/24.,e.iter().map(|v|if *v>0.{1./24.}else if *v<0.{-1./24.}else{0.}).collect())
            });loss+=v;for(a,b)in g.iter_mut().zip(gr){add(a,&b);}
        }
        let n=rows.len()as f64;
        for(l,g)in self.layers.iter().zip(&mut g){let stride=l.n+usize::from(l.bias);for(i,d)in g.iter_mut().enumerate(){*d/=n;if !l.bias||i%stride!=l.n{*d+=decay*l.w[i];}}}
        let norm=g.iter().flatten().map(|v|v*v).sum::<f64>().sqrt();
        if !loss.is_finite()||!norm.is_finite(){return(f64::NAN,false)}
        let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        opt.begin();for(l,g)in self.layers.iter_mut().zip(g){for(w,d)in l.w.iter_mut().zip(g){opt.update(w,d*scale);}}opt.finish();(loss/n,clipped)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn pooling_interpolation_and_decomposition(){
        let x:Vec<_>=(0..72).map(|i|i as f64).collect();assert_eq!(pool(&x,6)[0],2.5);assert_eq!(pool(&x,6)[11],68.5);
        let y=interp(&[2.,4.,6.,8.]);assert_eq!(y[0],2.);assert_eq!(y[23],8.);assert!((y[7]-(2.+6.*7./23.)).abs()<1e-12);
        let g=interp_grad(4,&[1.;24]);assert!((g.iter().sum::<f64>()-24.).abs()<1e-12);
        assert_eq!(trend(&x,7)[0],6./7.);assert!((trend(&x,7)[71]-(71.-6./7.)).abs()<1e-12);assert_eq!(trend(&[3.;72],25),vec![3.;72]);
        let a=Structured::new("nlinear-direct",42).unwrap();let x=[0.1;80];let mut z=x;for v in &mut z[..72]{*v+=2.;}for(p,q)in a.predict(&x).iter().zip(a.predict(&z)){assert!((q-p-2.).abs()<1e-12);}
        assert_eq!(a.parameters(),80*24);assert_eq!(Structured::new("nlinear-lead-bias",42).unwrap().parameters(),81*24);
    }
    #[test]fn gradient_groups_and_overfit(){
        let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let y:Vec<_>=(0..24).map(|i|0.1+(i as f64*0.25).cos()*0.05).collect();
        for label in ["nbeats-w64-b2","nbeats-w128-b2","nbeats-w64-b4","nhits-w64","nhits-w128","dlinear-k7","dlinear-k25","nlinear-direct","nlinear-lead-bias","tide-w64","tide-w128"]{
            let a=Structured::new(label,42).unwrap();let(v,g)=a.gradients(&x,&y);
            assert_eq!(a.predict(&x),Structured::new(label,42).unwrap().predict(&x));assert_ne!(a.predict(&x),Structured::new(label,123).unwrap().predict(&x));
            for(l,layer)in a.layers.iter().enumerate(){let stride=layer.n+usize::from(layer.bias);
                for j in [0,layer.m/2,layer.m-1]{for k in [0,layer.n/2,stride-1]{let i=j*stride+k;let(mut p,mut m)=(a.clone(),a.clone());p.layers[l].w[i]+=1e-6;m.layers[l].w[i]-=1e-6;
                    let d=(loss(&p.predict(&x),&y).0-loss(&m.predict(&x),&y).0)/2e-6;assert!((d-g[l][i]).abs()<1e-6,"{label} {l} {i} {d} {}",g[l][i]);}}
            }
            let mut a=a;let mut opt=Adam::new(a.parameters(),0.001);for _ in 0..100{assert!(a.step(&x,&y,&mut opt).0.is_finite());}assert!(a.finite()&&loss(&a.predict(&x),&y).0<v*0.1,"{label}");
        }
    }
    #[test]fn custom_nhits_gradient_groups(){
        let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let y:Vec<_>=(0..24).map(|i|0.4+(i as f64*0.25).cos()*0.05).collect();let a=Structured::new("nhits-w64",42).unwrap();
        let obj=|p:&[f64],t:&[f64]|{let e:Vec<_>=p.iter().zip(t).map(|(a,b)|a-b).collect();(e.iter().map(|v|v.abs()).sum::<f64>()/24.,e.iter().map(|v|v.signum()/24.).collect())};
        let(_,g)=a.gradients_with(&x,&y,obj);
        for(l,layer)in a.layers.iter().enumerate(){let stride=layer.n+usize::from(layer.bias);for j in [0,layer.m/2,layer.m-1]{for k in [0,layer.n/2,stride-1]{let i=j*stride+k;let(mut p,mut m)=(a.clone(),a.clone());p.layers[l].w[i]+=1e-6;m.layers[l].w[i]-=1e-6;let d=(obj(&p.predict(&x),&y).0-obj(&m.predict(&x),&y).0)/2e-6;assert!((d-g[l][i]).abs()<1e-6,"{l} {i}");}}}
    }
    #[test]fn optimization_mae_batch_parity(){
        let x=[0.1;80];let y=[0.4;24];let mut a=Structured::new("nhits-w64",42).unwrap();let mut b=a.clone();let mut c=a.clone();let mut oa=Adam::new(a.parameters(),0.001);let mut ob=Adam::new(b.parameters(),0.001);let mut oc=Adam::new(c.parameters(),0.001);
        let objective=|p:&[f64],t:&[f64]|{let e:Vec<_>=p.iter().zip(t).map(|(p,t)|p-t).collect();(e.iter().map(|v|v.abs()).sum::<f64>()/24.,e.iter().map(|v|if *v>0.{1./24.}else if *v<0.{-1./24.}else{0.}).collect())};
        for _ in 0..4{let v=a.step_with(&x,&y,&mut oa,objective);assert_eq!(v,b.optimization_mae_batch(&[(&x,&y)],&mut ob,0.));assert_eq!(v,c.optimization_mae_batch(&[(&x,&y),(&x,&y)],&mut oc,0.));assert_eq!(a.predict(&x),b.predict(&x));assert_eq!(a.predict(&x),c.predict(&x));}
    }
}
