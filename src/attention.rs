//! Small transformer-style encoder with explicit reverse-mode derivatives.
const S:usize=12;const D:usize=8;const P:usize=6;const B:usize=16;const O:usize=24;const F:usize=S*D+8;
type Tokens=[[f64;D];S];
#[derive(Clone)]
struct Linear{n:usize,m:usize,w:Vec<f64>}
impl Linear{
    fn new(n:usize,m:usize,sample:&mut impl FnMut(f64)->f64)->Self{
        let scale=(6./(n+m)as f64).sqrt();
        Self{n,m,w:(0..m*(n+1)).map(|i|if i%(n+1)==n{0.}else{sample(scale)}).collect()}
    }
    fn forward<const M:usize>(&self,x:&[f64])->[f64;M]{
        assert_eq!(M,self.m);assert_eq!(x.len(),self.n);
        std::array::from_fn(|j|{let a=j*(self.n+1);self.w[a+self.n]+(0..self.n).map(|i|self.w[a+i]*x[i]).sum::<f64>()})
    }
    fn backward(&self,x:&[f64],dy:&[f64],gw:&mut[f64],dx:&mut[f64]){
        for j in 0..self.m{let a=j*(self.n+1);gw[a+self.n]+=dy[j];for i in 0..self.n{gw[a+i]+=dy[j]*x[i];dx[i]+=dy[j]*self.w[a+i]}}
    }
}
#[derive(Clone)]
pub struct Attention{layers:[Linear;7],pos:Tokens}
struct Cache{e:Tokens,q:Tokens,k:Tokens,v:Tokens,a:[[f64;S];S],u:Tokens,b:[[f64;B];S],z:Tokens,read:[f64;F]}
impl Attention{
    pub fn new(seed:u64)->Self{
        let mut state=seed.max(1);
        let mut sample=|s:f64|{state^=state<<13;state^=state>>7;state^=state<<17;((state>>11)as f64/(1u64<<53)as f64*2.-1.)*s};
        let dims=[(P,D),(D,D),(D,D),(D,D),(D,B),(B,D),(F,O)];
        let layers=dims.map(|(n,m)|Linear::new(n,m,&mut sample));
        let pos=std::array::from_fn(|_|std::array::from_fn(|_|sample(0.1)));
        Self{layers,pos}
    }
    pub fn parameters(&self)->usize{self.layers.iter().map(|l|l.w.len()).sum::<usize>()+S*D}
    fn forward(&self,x:&[f64])->(Cache,[f64;O]){
        assert_eq!(x.len(),80);
        let e:Tokens=std::array::from_fn(|t|{let p=self.layers[0].forward::<D>(&x[t*P..(t+1)*P]);std::array::from_fn(|j|p[j]+self.pos[t][j])});
        let q=e.map(|v|self.layers[1].forward::<D>(&v));let k=e.map(|v|self.layers[2].forward::<D>(&v));let v=e.map(|v|self.layers[3].forward::<D>(&v));
        let scale=1./(D as f64).sqrt();
        let a:[[f64;S];S]=std::array::from_fn(|t|{
            let logits:[f64;S]=std::array::from_fn(|s|scale*(0..D).map(|j|q[t][j]*k[s][j]).sum::<f64>());
            let max=logits.iter().copied().fold(f64::NEG_INFINITY,f64::max);let ex=logits.map(|l|(l-max).exp());let sum=ex.iter().sum::<f64>();ex.map(|l|l/sum)
        });
        let u:Tokens=std::array::from_fn(|t|std::array::from_fn(|j|(e[t][j]+(0..S).map(|s|a[t][s]*v[s][j]).sum::<f64>()).tanh()));
        let b=u.map(|v|self.layers[4].forward::<B>(&v).map(f64::tanh));
        let z:Tokens=std::array::from_fn(|t|{let p=self.layers[5].forward::<D>(&b[t]);std::array::from_fn(|j|(p[j]+u[t][j]).tanh())});
        let mut read=[0.;F];for t in 0..S{read[t*D..(t+1)*D].copy_from_slice(&z[t])}read[S*D..].copy_from_slice(&x[72..]);
        let y=self.layers[6].forward::<O>(&read);
        (Cache{e,q,k,v,a,u,b,z,read},y)
    }
    pub fn predict(&self,x:&[f64])->[f64;O]{self.forward(x).1}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,[Vec<f64>;7],Tokens){
        assert_eq!(target.len(),O);let(c,y)=self.forward(x);
        let mut g=self.layers.each_ref().map(|l|vec![0.;l.w.len()]);let mut dr=[0.;F];
        let err: [f64;O]=std::array::from_fn(|i|y[i]-target[i]);let loss=err.iter().map(|v|v*v).sum::<f64>()/O as f64;
        self.layers[6].backward(&c.read,&err.map(|v|2.*v/O as f64),&mut g[6],&mut dr);
        let mut de=[[0.;D];S];let mut dq=de;let mut dk=de;let mut dv=de;let scale=1./(D as f64).sqrt();
        for t in 0..S{
            let dz:[f64;D]=std::array::from_fn(|j|dr[t*D+j]*(1.-c.z[t][j]*c.z[t][j]));
            let mut du=dz;let mut db=[0.;B];
            self.layers[5].backward(&c.b[t],&dz,&mut g[5],&mut db);
            for j in 0..B{db[j]*=1.-c.b[t][j]*c.b[t][j]}
            self.layers[4].backward(&c.u[t],&db,&mut g[4],&mut du);
            for j in 0..D{du[j]*=1.-c.u[t][j]*c.u[t][j];de[t][j]+=du[j]}
            let mut da=[0.;S];
            for s in 0..S{for j in 0..D{da[s]+=du[j]*c.v[s][j];dv[s][j]+=c.a[t][s]*du[j]}}
            let mean=(0..S).map(|s|da[s]*c.a[t][s]).sum::<f64>();
            for s in 0..S{let dl=scale*c.a[t][s]*(da[s]-mean);for j in 0..D{dq[t][j]+=dl*c.k[s][j];dk[s][j]+=dl*c.q[t][j]}}
        }
        for t in 0..S{
            for (l,d) in [(1,&dq[t]),(2,&dk[t]),(3,&dv[t])]{self.layers[l].backward(&c.e[t],d,&mut g[l],&mut de[t]);}
            self.layers[0].backward(&x[t*P..(t+1)*P],&de[t],&mut g[0],&mut [0.;P]);
        }(loss,g,de)
    }
    pub fn train_step(&mut self,x:&[f64],y:&[f64],lr:f64)->(f64,bool){
        let(loss,g,p)=self.gradients(x,y);let norm=g.iter().flatten().chain(p.iter().flatten()).map(|v|v*v).sum::<f64>().sqrt();
        let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        for(l,d)in self.layers.iter_mut().zip(g){for(w,v)in l.w.iter_mut().zip(d){*w-=lr*scale*v}}
        for(t,d)in self.pos.iter_mut().zip(p){for(w,v)in t.iter_mut().zip(d){*w-=lr*scale*v}}
        (loss,clipped)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn gradients(){
        let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let y=[0.1;24];let eps=1e-6;
        let m=Attention::new(42);assert_eq!(m.parameters(),3168);let(_,g,p)=m.gradients(&x,&y);
        let loss=|m:&Attention|m.predict(&x).iter().zip(y).map(|(a,b)|(a-b).powi(2)).sum::<f64>()/24.;
        for l in 0..7{for(i,d)in g[l].iter().enumerate(){let(mut a,mut b)=(m.clone(),m.clone());a.layers[l].w[i]+=eps;b.layers[l].w[i]-=eps;assert!(((loss(&a)-loss(&b))/(2.*eps)-d).abs()<1e-6,"layer {l} weight {i}")}}
        for t in 0..S{for j in 0..D{let(mut a,mut b)=(m.clone(),m.clone());a.pos[t][j]+=eps;b.pos[t][j]-=eps;assert!(((loss(&a)-loss(&b))/(2.*eps)-p[t][j]).abs()<1e-6)}}
        let(c,_)=m.forward(&x);for row in c.a{assert!((row.iter().sum::<f64>()-1.).abs()<1e-12)}
        assert_eq!(m.predict(&x),Attention::new(42).predict(&x));
        let mut m=m;let before=loss(&m);for _ in 0..100{m.train_step(&x,&y,0.02);}assert!(loss(&m)<before);
    }
}
