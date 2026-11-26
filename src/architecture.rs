//! Common architecture dispatch for controlled matched-cohort studies.
use crate::{dense::Dense,recurrent::{Kind as RKind,Recurrent},convolution::{Kind as CKind,Temporal},attention::Attention};
pub enum Model{Dense(Dense),Recurrent(Recurrent),Temporal(Temporal),Attention(Attention)}
impl Model{
    pub fn new(label:&str,seed:u64)->Result<Self,String>{
        Ok(match label{
            "mlp32"=>Self::Dense(Dense::new(&[80,32,24],seed)),
            "mlp64"=>Self::Dense(Dense::new(&[80,64,24],seed)),
            "mlp32x32"=>Self::Dense(Dense::new(&[80,32,32,24],seed)),
            "rnn"=>Self::Recurrent(Recurrent::new(RKind::Rnn,seed)),
            "cnn"=>Self::Temporal(Temporal::new(CKind::Cnn,seed)),
            "attention"=>Self::Attention(Attention::new(seed)),
            _=>return Err(format!("unknown architecture {label}"))
        })
    }
    pub fn parameters(&self)->usize{match self{Self::Dense(m)=>m.parameters(),Self::Recurrent(m)=>m.parameters(),Self::Temporal(m)=>m.parameters(),Self::Attention(m)=>m.parameters()}}
    pub fn predict(&self,x:&[f64])->[f64;24]{match self{Self::Dense(m)=>m.predict(x).try_into().expect("24 outputs"),Self::Recurrent(m)=>m.predict(x),Self::Temporal(m)=>m.predict(x),Self::Attention(m)=>m.predict(x)}}
    pub fn adam_step(&mut self,x:&[f64],y:&[f64],opt:&mut crate::adam::Adam)->(f64,bool){match self{Self::Dense(m)=>m.adam_step(x,y,opt),Self::Recurrent(m)=>m.adam_step(x,y,opt),Self::Temporal(m)=>m.adam_step(x,y,opt),Self::Attention(m)=>m.adam_step(x,y,opt)}}
    pub fn train_step(&mut self,x:&[f64],y:&[f64],lr:f64)->(f64,bool){match self{Self::Dense(m)=>(m.train_step(x,y,lr),false),Self::Recurrent(m)=>m.train_step(x,y,lr),Self::Temporal(m)=>m.train_step(x,y,lr),Self::Attention(m)=>m.train_step(x,y,lr)}}
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn dispatch_shapes(){for(label,n)in [("mlp32",3384),("mlp64",6744),("mlp32x32",4440),("rnn",968),("cnn",3072),("attention",3168)]{
        let mut m=Model::new(label,42).unwrap();assert_eq!(m.parameters(),n);let x=[0.1;80];assert_eq!(m.predict(&x),Model::new(label,42).unwrap().predict(&x));assert!(m.train_step(&x,&[0.2;24],0.005).0.is_finite());let mut opt=crate::adam::Adam::new(n,0.001);for _ in 0..10{assert!(m.adam_step(&x,&[0.2;24],&mut opt).0.is_finite());}
    }}
}
