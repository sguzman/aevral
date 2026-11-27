//! Three-layer causal residual temporal convolution, explicit backpropagation.
const C:usize=8;
const K:usize=3;
const L:usize=3;
const O:usize=24;
const A:usize=8;
type Seq=Vec<[f64;C]>;
#[derive(Clone,Copy,Debug)]
pub enum Kind{Cnn,Tcn}
impl Kind{
    pub fn parse(s:&str)->Option<Self>{match s{"cnn"=>Some(Self::Cnn),"tcn"=>Some(Self::Tcn),_=>None}}
    fn dilation(self,l:usize)->usize{match self{Self::Cnn=>1,Self::Tcn=>1<<l}}
}
#[derive(Clone)]
pub struct Temporal{kind:Kind,patch:usize,post_skip:bool,w:[Vec<f64>;L],head:Vec<f64>}
fn channels(l:usize,patch:usize)->usize{if l==0{patch}else{C}}
fn offset(n:usize,j:usize)->usize{j*(K*n+1)}
impl Temporal{
    pub fn new(kind:Kind,seed:u64)->Self{Self::build(kind,6,false,seed)}
    pub fn c009(kind:Kind,patch:usize,seed:u64)->Self{assert!([3,6].contains(&patch));Self::build(kind,patch,true,seed)}
    fn build(kind:Kind,patch:usize,post_skip:bool,seed:u64)->Self{
        let f=72/patch*C+A+1;
        let mut state=seed.max(1);
        let mut sample=|s:f64|{state^=state<<13;state^=state>>7;state^=state<<17;((state>>11)as f64/(1u64<<53)as f64*2.-1.)*s};
        let w=std::array::from_fn(|l|{
            let n=channels(l,patch);let stride=K*n+1;let scale=(6./(K*(n+C))as f64).sqrt();
            (0..C*stride).map(|i|if i%stride==stride-1{0.}else{sample(scale)}).collect()
        });
        let head=(0..O*f).map(|i|if i%f==f-1{0.}else{sample((6./(f-1+O)as f64).sqrt())}).collect();
        Self{kind,patch,post_skip,w,head}
    }
    pub fn parameters(&self)->usize{self.w.iter().map(Vec::len).sum::<usize>()+self.head.len()}
    pub fn finite(&self)->bool{self.w.iter().flatten().chain(&self.head).all(|v|v.is_finite())}
    fn forward(&self,x:&[f64])->([Seq;L+1],[f64;O]){
        assert_eq!(x.len(),72+A);
        let p=self.patch;let s=72/p;let f=s*C+A+1;
        let mut a:[Seq;L+1]=std::array::from_fn(|_|vec![[0.;C];s]);
        for t in 0..s{a[0][t][..p].copy_from_slice(&x[t*p..(t+1)*p])}
        for l in 0..L{
            let n=channels(l,self.patch);let dilation=self.kind.dilation(l);
            for t in 0..s{for j in 0..C{
                let off=offset(n,j);
                let mut v=self.w[l][off+K*n]+if l>0&&!self.post_skip{a[l][t][j]}else{0.};
                for k in 0..K{if let Some(p)=t.checked_sub(k*dilation){for i in 0..n{v+=self.w[l][off+k*n+i]*a[l][p][i]}}}
                a[l+1][t][j]=v.tanh()+if l>0&&self.post_skip{a[l][t][j]}else{0.};
            }}
        }
        let y=std::array::from_fn(|k|{
            let off=k*f;let mut v=self.head[off+f-1];
            for t in 0..s{for j in 0..C{v+=self.head[off+t*C+j]*a[L][t][j]}}
            for j in 0..A{v+=self.head[off+s*C+j]*x[72+j]}v
        });(a,y)
    }
    pub fn predict(&self,x:&[f64])->[f64;O]{self.forward(x).1}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,[Vec<f64>;L],Vec<f64>){
        assert_eq!(target.len(),O);
        let(a,y)=self.forward(x);let s=72/self.patch;let f=s*C+A+1;
        let mut gw=std::array::from_fn(|l|vec![0.;self.w[l].len()]);
        let mut go=vec![0.;self.head.len()];let mut da=vec![[0.;C];s];let mut loss=0.;
        for k in 0..O{
            let e=y[k]-target[k];loss+=e*e/O as f64;let d=2.*e/O as f64;let off=k*f;
            for t in 0..s{for j in 0..C{go[off+t*C+j]=d*a[L][t][j];da[t][j]+=d*self.head[off+t*C+j]}}
            for j in 0..A{go[off+s*C+j]=d*x[72+j]}go[off+f-1]=d;
        }
        for l in (0..L).rev(){
            let n=channels(l,self.patch);let dilation=self.kind.dilation(l);let mut prev=vec![[0.;C];s];
            for t in 0..s{for j in 0..C{
                let activation=a[l+1][t][j]-if l>0&&self.post_skip{a[l][t][j]}else{0.};
                let d=da[t][j]*(1.-activation*activation);let off=offset(n,j);
                gw[l][off+K*n]+=d;
                if l>0{prev[t][j]+=if self.post_skip{da[t][j]}else{d}}
                for k in 0..K{if let Some(p)=t.checked_sub(k*dilation){for i in 0..n{
                    let idx=off+k*n+i;gw[l][idx]+=d*a[l][p][i];prev[p][i]+=d*self.w[l][idx];
                }}}
            }}da=prev;
        }(loss,gw,go)
    }
    pub fn adam_step(&mut self,x:&[f64],y:&[f64],opt:&mut crate::adam::Adam)->(f64,bool){
        let(loss,gw,go)=self.gradients(x,y);
        let norm=gw.iter().flatten().chain(&go).map(|v|v*v).sum::<f64>().sqrt();
        let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        if !norm.is_finite()||!loss.is_finite(){return(f64::NAN,false)}
        opt.begin();
        for(l,g)in self.w.iter_mut().zip(gw){for(w,v)in l.iter_mut().zip(g){opt.update(w,scale*v)}}
        for(w,g)in self.head.iter_mut().zip(go){opt.update(w,scale*g)}
        opt.finish();(loss,clipped)
    }
    pub fn train_step(&mut self,x:&[f64],y:&[f64],lr:f64)->(f64,bool){
        let(loss,gw,go)=self.gradients(x,y);
        let norm=gw.iter().flatten().chain(&go).map(|v|v*v).sum::<f64>().sqrt();
        let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        for(l,g)in self.w.iter_mut().zip(gw){for(w,v)in l.iter_mut().zip(g){*w-=lr*scale*v}}
        for(w,g)in self.head.iter_mut().zip(go){*w-=lr*scale*g}
        (loss,clipped)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn gradient_all_parameters(){
        let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let y=[0.1;24];let eps=1e-6;
        for kind in [Kind::Cnn,Kind::Tcn]{
            let m=Temporal::new(kind,42);assert_eq!(m.parameters(),3072);
            let(_,gw,go)=m.gradients(&x,&y);
            let loss=|m:&Temporal|m.predict(&x).iter().zip(y).map(|(p,t)|(p-t).powi(2)).sum::<f64>()/24.;
            for l in 0..=L{
                let grad=if l==L{&go}else{&gw[l]};
                for(i,g)in grad.iter().enumerate(){
                    let(mut a,mut b)=(m.clone(),m.clone());
                    if l==L{a.head[i]+=eps;b.head[i]-=eps}else{a.w[l][i]+=eps;b.w[l][i]-=eps}
                    assert!(((loss(&a)-loss(&b))/(2.*eps)-g).abs()<1e-6,"{kind:?} {l} {i}");
                }
            }
            assert_eq!(m.predict(&x),Temporal::new(kind,42).predict(&x));
            let mut m=m;let before=loss(&m);for _ in 0..100{m.train_step(&x,&y,0.02);}assert!(loss(&m)<before);
        }
    }
    #[test]fn causal_prefix(){
        for kind in [Kind::Cnn,Kind::Tcn]{
            let m=Temporal::new(kind,42);let x=[0.;80];let mut y=x;y[66]=1.;
            let(a,_)=m.forward(&x);let(b,_)=m.forward(&y);
            for l in 0..=L{assert_eq!(a[l][..11],b[l][..11]);}
        }
    }

    #[test]fn c009_causal_and_gradients(){
        let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let y=[0.1;24];
        for kind in [Kind::Cnn,Kind::Tcn]{for patch in [3,6]{
            let a=Temporal::c009(kind,patch,42);let(_,gw,go)=a.gradients(&x,&y);
            let loss=|m:&Temporal|m.predict(&x).iter().zip(y).map(|(p,t)|(p-t).powi(2)).sum::<f64>()/24.;
            for l in 0..=L{let g=if l==L{&go}else{&gw[l]};for(i,v)in g.iter().enumerate(){
                let(mut p,mut m)=(a.clone(),a.clone());if l==L{p.head[i]+=1e-6;m.head[i]-=1e-6}else{p.w[l][i]+=1e-6;m.w[l][i]-=1e-6}
                assert!(((loss(&p)-loss(&m))/2e-6-v).abs()<1e-6,"{kind:?} {patch} {l} {i}");
            }}
            let mut changed=x.clone();changed[72-patch]+=1.;let(z,_)=a.forward(&x);let(v,_)=a.forward(&changed);
            for l in 0..=L{assert_eq!(z[l][..72/patch-1],v[l][..72/patch-1]);}
            let mut skip=a.clone();skip.w[1].fill(0.);skip.w[2].fill(0.);let(z,_)=skip.forward(&x);assert_eq!(z[1],z[2]);assert_eq!(z[2],z[3]);
            assert_eq!(a.predict(&x),Temporal::c009(kind,patch,42).predict(&x));assert_ne!(a.predict(&x),Temporal::c009(kind,patch,123).predict(&x));
            let mut m=a;let before=loss(&m);let mut opt=crate::adam::Adam::new(m.parameters(),0.001);for _ in 0..100{assert!(m.adam_step(&x,&y,&mut opt).0.is_finite());}assert!(m.finite()&&loss(&m)<before);
        }}
    }
}
