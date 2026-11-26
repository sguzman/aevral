//! Three-layer causal residual temporal convolution, explicit backpropagation.
const S:usize=12;
const P:usize=6;
const C:usize=8;
const K:usize=3;
const L:usize=3;
const O:usize=24;
const A:usize=8;
const F:usize=S*C+A+1;
type Seq=[[f64;C];S];
#[derive(Clone,Copy,Debug)]
pub enum Kind{Cnn,Tcn}
impl Kind{
    pub fn parse(s:&str)->Option<Self>{match s{"cnn"=>Some(Self::Cnn),"tcn"=>Some(Self::Tcn),_=>None}}
    fn dilation(self,l:usize)->usize{match self{Self::Cnn=>1,Self::Tcn=>1<<l}}
}
#[derive(Clone)]
pub struct Temporal{kind:Kind,w:[Vec<f64>;L],head:Vec<f64>}
fn channels(l:usize)->usize{if l==0{P}else{C}}
fn offset(n:usize,j:usize)->usize{j*(K*n+1)}
impl Temporal{
    pub fn new(kind:Kind,seed:u64)->Self{
        let mut state=seed.max(1);
        let mut sample=|s:f64|{state^=state<<13;state^=state>>7;state^=state<<17;((state>>11)as f64/(1u64<<53)as f64*2.-1.)*s};
        let w=std::array::from_fn(|l|{
            let n=channels(l);let stride=K*n+1;let scale=(6./(K*(n+C))as f64).sqrt();
            (0..C*stride).map(|i|if i%stride==stride-1{0.}else{sample(scale)}).collect()
        });
        let head=(0..O*F).map(|i|if i%F==F-1{0.}else{sample((6./(F-1+O)as f64).sqrt())}).collect();
        Self{kind,w,head}
    }
    pub fn parameters(&self)->usize{self.w.iter().map(Vec::len).sum::<usize>()+self.head.len()}
    fn forward(&self,x:&[f64])->([Seq;L+1],[f64;O]){
        assert_eq!(x.len(),S*P+A);
        let mut a=[[[0.;C];S];L+1];
        for t in 0..S{a[0][t][..P].copy_from_slice(&x[t*P..(t+1)*P])}
        for l in 0..L{
            let n=channels(l);let dilation=self.kind.dilation(l);
            for t in 0..S{for j in 0..C{
                let off=offset(n,j);
                let mut v=self.w[l][off+K*n]+if l>0{a[l][t][j]}else{0.};
                for k in 0..K{if let Some(p)=t.checked_sub(k*dilation){for i in 0..n{v+=self.w[l][off+k*n+i]*a[l][p][i]}}}
                a[l+1][t][j]=v.tanh();
            }}
        }
        let y=std::array::from_fn(|k|{
            let off=k*F;let mut v=self.head[off+F-1];
            for t in 0..S{for j in 0..C{v+=self.head[off+t*C+j]*a[L][t][j]}}
            for j in 0..A{v+=self.head[off+S*C+j]*x[S*P+j]}v
        });(a,y)
    }
    pub fn predict(&self,x:&[f64])->[f64;O]{self.forward(x).1}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,[Vec<f64>;L],Vec<f64>){
        assert_eq!(target.len(),O);
        let(a,y)=self.forward(x);
        let mut gw=std::array::from_fn(|l|vec![0.;self.w[l].len()]);
        let mut go=vec![0.;self.head.len()];let mut da=[[0.;C];S];let mut loss=0.;
        for k in 0..O{
            let e=y[k]-target[k];loss+=e*e/O as f64;let d=2.*e/O as f64;let off=k*F;
            for t in 0..S{for j in 0..C{go[off+t*C+j]=d*a[L][t][j];da[t][j]+=d*self.head[off+t*C+j]}}
            for j in 0..A{go[off+S*C+j]=d*x[S*P+j]}go[off+F-1]=d;
        }
        for l in (0..L).rev(){
            let n=channels(l);let dilation=self.kind.dilation(l);let mut prev=[[0.;C];S];
            for t in 0..S{for j in 0..C{
                let d=da[t][j]*(1.-a[l+1][t][j]*a[l+1][t][j]);let off=offset(n,j);
                gw[l][off+K*n]+=d;
                if l>0{prev[t][j]+=d}
                for k in 0..K{if let Some(p)=t.checked_sub(k*dilation){for i in 0..n{
                    let idx=off+k*n+i;gw[l][idx]+=d*a[l][p][i];prev[p][i]+=d*self.w[l][idx];
                }}}
            }}da=prev;
        }(loss,gw,go)
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
}
