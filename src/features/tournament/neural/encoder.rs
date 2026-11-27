//! C009 single-channel patch encoders; explicit reverse mode, no future inputs.
use super::residual::{Affine,sample};
use crate::adam::Adam;
type Tokens=Vec<Vec<f64>>;
fn add(a:&mut[f64],b:&[f64]){assert_eq!(a.len(),b.len());for(a,b)in a.iter_mut().zip(b){*a+=b;}}
fn zeros(s:usize,d:usize)->Tokens{vec![vec![0.;d];s]}
fn backward(l:&Affine,x:&[f64],d:&[f64],g:&mut[f64])->Vec<f64>{let(dx,dw)=l.backward(x,d);add(g,&dw);dx}
#[derive(Clone)]
struct Norm{gain:Vec<f64>,bias:Vec<f64>}
struct NormCache{z:Vec<f64>,inv:f64}
impl Norm{
    fn new(d:usize)->Self{Self{gain:vec![1.;d],bias:vec![0.;d]}}
    fn forward(&self,x:&[f64])->(Vec<f64>,NormCache){
        let n=x.len()as f64;let mean=x.iter().sum::<f64>()/n;
        let inv=1./(x.iter().map(|v|(v-mean).powi(2)).sum::<f64>()/n+1e-5).sqrt();
        let z:Vec<_>=x.iter().map(|v|(v-mean)*inv).collect();
        let y=z.iter().zip(&self.gain).zip(&self.bias).map(|((v,g),b)|v*g+b).collect();
        (y,NormCache{z,inv})
    }
    fn backward(&self,c:&NormCache,dy:&[f64],dg:&mut[f64],db:&mut[f64])->Vec<f64>{
        let n=dy.len()as f64;
        let d:Vec<_>=dy.iter().zip(&self.gain).map(|(a,b)|a*b).collect();
        let mean=d.iter().sum::<f64>()/n;
        let dot=d.iter().zip(&c.z).map(|(a,b)|a*b).sum::<f64>()/n;
        for j in 0..dy.len(){dg[j]+=dy[j]*c.z[j];db[j]+=dy[j];}
        d.iter().zip(&c.z).map(|(v,z)|c.inv*(v-mean-z*dot)).collect()
    }
}
#[derive(Clone)]
pub(super) struct Encoder{p:usize,d:usize,heads:usize,layers:Vec<Affine>,pos:Tokens,norms:Vec<Norm>}
struct Cache{e:Tokens,q:Tokens,k:Tokens,v:Tokens,a:Vec<Tokens>,context:Tokens,u:Tokens,b:Tokens,z:Tokens,n1:Vec<NormCache>,n2:Vec<NormCache>,read:Vec<f64>}
struct Grad{layers:Vec<Vec<f64>>,pos:Tokens,norm:Vec<(Vec<f64>,Vec<f64>)>}
impl Encoder{
    pub(super) fn new(p:usize,d:usize,patchtst:bool,seed:u64)->Self{
        let mut s=seed.max(1);let steps=72/p;let heads=if patchtst{2}else{1};
        let mut dims=vec![(p,d),(d,d),(d,d),(d,d)];
        if patchtst{dims.push((d,d));}
        dims.extend([(d,2*d),(2*d,d),(steps*d+8,24)]);
        let layers=dims.iter().map(|&(n,m)|Affine::new(n,m,&mut s)).collect();
        let pos=(0..steps).map(|_|(0..d).map(|_|sample(&mut s,0.1)).collect()).collect();
        let norms=if patchtst{vec![Norm::new(d),Norm::new(d)]}else{vec![]};
        Self{p,d,heads,layers,pos,norms}
    }
    pub(super) fn parameters(&self)->usize{self.layers.iter().map(|l|l.w.len()).sum::<usize>()+self.pos.len()*self.d+self.norms.len()*2*self.d}
    pub(super) fn finite(&self)->bool{self.layers.iter().flat_map(|l|&l.w).chain(self.pos.iter().flatten()).chain(self.norms.iter().flat_map(|n|n.gain.iter().chain(&n.bias))).all(|v|v.is_finite())}
    fn forward(&self,x:&[f64])->(Cache,Vec<f64>){
        assert_eq!(x.len(),80);let s=72/self.p;let d=self.d;let w=d/self.heads;
        let mut e=Vec::with_capacity(s);
        for t in 0..s{let mut v=self.layers[0].forward(&x[t*self.p..(t+1)*self.p]);add(&mut v,&self.pos[t]);e.push(v);}
        let q:Tokens=e.iter().map(|x|self.layers[1].forward(x)).collect();
        let k:Tokens=e.iter().map(|x|self.layers[2].forward(x)).collect();
        let v:Tokens=e.iter().map(|x|self.layers[3].forward(x)).collect();
        let scale=1./(w as f64).sqrt();let mut a=vec![zeros(s,s);self.heads];let mut context=zeros(s,d);
        for h in 0..self.heads{let start=h*w;let end=start+w;
            for t in 0..s{let logits:Vec<_>=(0..s).map(|j|scale*(start..end).map(|i|q[t][i]*k[j][i]).sum::<f64>()).collect();
                let max=logits.iter().copied().fold(f64::NEG_INFINITY,f64::max);
                let mut probs:Vec<_>=logits.iter().map(|l|(l-max).exp()).collect();let total=probs.iter().sum::<f64>();
                for p in &mut probs{*p/=total;}
                for j in start..end{context[t][j]=(0..s).map(|i|probs[i]*v[i][j]).sum();}
                a[h][t]=probs;
            }
        }
        let mut u=Vec::new();let mut b=Vec::new();let mut z=Vec::new();let mut n1=Vec::new();let mut n2=Vec::new();
        let patchtst=self.heads==2;let f=if patchtst{5}else{4};
        for t in 0..s{
            let mut r=if patchtst{self.layers[4].forward(&context[t])}else{context[t].clone()};add(&mut r,&e[t]);
            let p=if patchtst{let(y,c)=self.norms[0].forward(&r);n1.push(c);y}else{r.into_iter().map(f64::tanh).collect()};
            let h:Vec<_>=self.layers[f].forward(&p).into_iter().map(|v|if patchtst{v.max(0.)}else{v.tanh()}).collect();
            let mut r=self.layers[f+1].forward(&h);add(&mut r,&p);
            let out=if patchtst{let(y,c)=self.norms[1].forward(&r);n2.push(c);y}else{r.into_iter().map(f64::tanh).collect()};
            u.push(p);b.push(h);z.push(out);
        }
        let mut read:Vec<_>=z.iter().flatten().copied().collect();read.extend_from_slice(&x[72..]);let y=self.layers[f+2].forward(&read);
        (Cache{e,q,k,v,a,context,u,b,z,n1,n2,read},y)
    }
    pub(super) fn predict(&self,x:&[f64])->Vec<f64>{self.forward(x).1}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,Grad){
        assert_eq!(target.len(),24);let(c,y)=self.forward(x);let s=72/self.p;let d=self.d;let w=d/self.heads;
        let mut g:Vec<_>=self.layers.iter().map(|l|vec![0.;l.w.len()]).collect();
        let mut ng:Vec<_>=self.norms.iter().map(|_|(vec![0.;d],vec![0.;d])).collect();
        let err:Vec<_>=y.iter().zip(target).map(|(a,b)|a-b).collect();let loss=err.iter().map(|v|v*v).sum::<f64>()/24.;
        let dy:Vec<_>=err.iter().map(|v|v/12.).collect();let patchtst=self.heads==2;let f=if patchtst{5}else{4};
        let dr=backward(&self.layers[f+2],&c.read,&dy,&mut g[f+2]);
        let mut de=zeros(s,d);let mut dq=zeros(s,d);let mut dk=zeros(s,d);let mut dv=zeros(s,d);let scale=1./(w as f64).sqrt();
        for t in 0..s{
            let raw=&dr[t*d..(t+1)*d];
            let dz=if patchtst{let(gg,gb)=&mut ng[1];self.norms[1].backward(&c.n2[t],raw,gg,gb)}else{raw.iter().zip(&c.z[t]).map(|(a,z)|a*(1.-z*z)).collect()};
            let mut du=dz.clone();let mut db=backward(&self.layers[f+1],&c.b[t],&dz,&mut g[f+1]);
            for(j,v)in db.iter_mut().enumerate(){let h=c.b[t][j];*v*=if patchtst{if h>0.{1.}else{0.}}else{1.-h*h};}
            add(&mut du,&backward(&self.layers[f],&c.u[t],&db,&mut g[f]));
            let dr=if patchtst{let(gg,gb)=&mut ng[0];self.norms[0].backward(&c.n1[t],&du,gg,gb)}else{du.iter().zip(&c.u[t]).map(|(a,z)|a*(1.-z*z)).collect()};
            add(&mut de[t],&dr);
            let dc=if patchtst{backward(&self.layers[4],&c.context[t],&dr,&mut g[4])}else{dr};
            for h in 0..self.heads{let start=h*w;let end=start+w;let a=&c.a[h][t];let mut da=vec![0.;s];
                for j in 0..s{for i in start..end{da[j]+=dc[i]*c.v[j][i];dv[j][i]+=a[j]*dc[i];}}
                let mean=da.iter().zip(a).map(|(v,p)|v*p).sum::<f64>();
                for j in 0..s{let dl=scale*a[j]*(da[j]-mean);for i in start..end{dq[t][i]+=dl*c.k[j][i];dk[j][i]+=dl*c.q[t][i];}}
            }
        }
        for t in 0..s{for(l,dx)in [(1,&dq[t]),(2,&dk[t]),(3,&dv[t])]{add(&mut de[t],&backward(&self.layers[l],&c.e[t],dx,&mut g[l]));}
            let _=backward(&self.layers[0],&x[t*self.p..(t+1)*self.p],&de[t],&mut g[0]);}
        (loss,Grad{layers:g,pos:de,norm:ng})
    }
    pub(super) fn step(&mut self,x:&[f64],y:&[f64],opt:&mut Adam)->(f64,bool){
        let(loss,g)=self.gradients(x,y);let norm=g.layers.iter().flatten().chain(g.pos.iter().flatten()).chain(g.norm.iter().flat_map(|(a,b)|a.iter().chain(b))).map(|v|v*v).sum::<f64>().sqrt();
        if !loss.is_finite()||!norm.is_finite(){return(f64::NAN,false)}let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        opt.begin();for(l,g)in self.layers.iter_mut().zip(g.layers){for(w,v)in l.w.iter_mut().zip(g){opt.update(w,scale*v);}}
        for(p,g)in self.pos.iter_mut().zip(g.pos){for(w,v)in p.iter_mut().zip(g){opt.update(w,scale*v);}}
        for(n,(a,b))in self.norms.iter_mut().zip(g.norm){for(w,v)in n.gain.iter_mut().zip(a){opt.update(w,scale*v);}for(w,v)in n.bias.iter_mut().zip(b){opt.update(w,scale*v);}}opt.finish();(loss,clipped)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn norms_and_softmax(){let n=Norm::new(4);let(y,c)=n.forward(&[1.,2.,3.,4.]);assert!(y.iter().sum::<f64>().abs()<1e-12);assert!((y.iter().map(|v|v*v).sum::<f64>()/4.-1.25/(1.25+1e-5)).abs()<1e-12);
        let mut g=[0.;4];let mut b=[0.;4];let dx=n.backward(&c,&[1.,2.,3.,4.],&mut g,&mut b);assert!(dx.iter().sum::<f64>().abs()<1e-12);
        let e=Encoder::new(6,16,true,42);let(c,_)=e.forward(&[0.1;80]);for h in c.a{for p in h{assert!((p.iter().sum::<f64>()-1.).abs()<1e-12);assert!(p.iter().all(|x|*x>=0.));}}
    }
    #[test]fn gradient_groups_and_fit(){let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let y:Vec<_>=(0..24).map(|i|0.1+(i as f64*0.25).cos()*0.05).collect();
        let loss=|m:&Encoder|m.predict(&x).iter().zip(&y).map(|(a,b)|(a-b).powi(2)).sum::<f64>()/24.;
        for(p,d,pt)in [(6,16,true),(3,16,true),(6,8,false),(3,8,false),(6,16,false)]{
            let a=Encoder::new(p,d,pt,42);let(v,g)=a.gradients(&x,&y);let eps=1e-6;
            assert_eq!(a.predict(&x),Encoder::new(p,d,pt,42).predict(&x));assert_ne!(a.predict(&x),Encoder::new(p,d,pt,123).predict(&x));
            for(l,layer)in a.layers.iter().enumerate(){let n=layer.w.len();for i in [0,n/4,n/2,3*n/4,n-1]{let(mut plus,mut minus)=(a.clone(),a.clone());plus.layers[l].w[i]+=eps;minus.layers[l].w[i]-=eps;assert!(((loss(&plus)-loss(&minus))/(2.*eps)-g.layers[l][i]).abs()<2e-6,"{p} {d} {pt} layer {l} {i}");}}
            for t in [0,36/p,72/p-1]{for j in [0,d/2,d-1]{let(mut plus,mut minus)=(a.clone(),a.clone());plus.pos[t][j]+=eps;minus.pos[t][j]-=eps;assert!(((loss(&plus)-loss(&minus))/(2.*eps)-g.pos[t][j]).abs()<2e-6);}}
            for l in 0..a.norms.len(){for j in [0,d/2,d-1]{for bias in [false,true]{let(mut plus,mut minus)=(a.clone(),a.clone());let(g1,g2)=&g.norm[l];if bias{plus.norms[l].bias[j]+=eps;minus.norms[l].bias[j]-=eps;}else{plus.norms[l].gain[j]+=eps;minus.norms[l].gain[j]-=eps;}
                let expected=if bias{g2[j]}else{g1[j]};assert!(((loss(&plus)-loss(&minus))/(2.*eps)-expected).abs()<2e-6);}}}
            let mut a=a;let mut opt=Adam::new(a.parameters(),0.001);for _ in 0..100{assert!(a.step(&x,&y,&mut opt).0.is_finite());}assert!(a.finite()&&loss(&a)<v*0.1);
        }
    }
    #[test]fn matches_legacy_initialization_and_output(){let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let a=Encoder::new(6,8,false,42);let b=crate::attention::Attention::new(42);assert_eq!(a.parameters(),b.parameters());for(p,q)in a.predict(&x).iter().zip(b.predict(&x)){assert!((p-q).abs()<1e-12);}}
}
