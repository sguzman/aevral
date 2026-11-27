//! C009 shared frozen training protocol; full evaluation only after guard checks.
use super::{load,score,Fold,TrainingRow,D,Y};
use super::neighbors::{subset,Cached};
use crate::{adam::Adam,dense::Dense,benchmark::Metrics,recurrent::{Recurrent,Kind},convolution::{Temporal,Kind as CKind}};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
mod residual;
mod state_mixture;
use state_mixture::{StateSpace,Mixture};
mod encoder;
use encoder::Encoder;
mod structured;
use structured::Structured;
mod seq2seq;
use seq2seq::Seq2Seq;
use residual::Residual;
const BUDGET:usize=8192;
const EPOCHS:usize=12;
const GUARD:f64=1400.;
enum Model{Dense(Dense),Residual(Residual),Recurrent(Recurrent),Temporal(Temporal),Seq2Seq(Seq2Seq),Structured(Structured),Encoder(Encoder),StateSpace(StateSpace),Mixture(Mixture)}
impl Model{
    fn new(label:&str,seed:u64)->Result<Self,String>{
        Ok(match label{
            "mlp-w128"=>Self::Dense(Dense::new(&[D,128,Y],seed)),
            "mlp-w256"=>Self::Dense(Dense::new(&[D,256,Y],seed)),
            "mlp-128x64"=>Self::Dense(Dense::new(&[D,128,64,Y],seed)),
            "resmlp-w64-b2"=>Self::Residual(Residual::new(D,64,2,Y,seed)),
            "resmlp-w128-b2"=>Self::Residual(Residual::new(D,128,2,Y,seed)),
            "resmlp-w64-b4"=>Self::Residual(Residual::new(D,64,4,Y,seed)),
            "rnn-p6-h16"=>Self::Recurrent(Recurrent::c009(Kind::Rnn,6,seed)),
            "rnn-p3-h16"=>Self::Recurrent(Recurrent::c009(Kind::Rnn,3,seed)),
            "rnn-p1-h16"=>Self::Recurrent(Recurrent::c009(Kind::Rnn,1,seed)),
            "gru-p6-h16"=>Self::Recurrent(Recurrent::c009(Kind::Gru,6,seed)),
            "gru-p3-h16"=>Self::Recurrent(Recurrent::c009(Kind::Gru,3,seed)),
            "gru-p1-h16"=>Self::Recurrent(Recurrent::c009(Kind::Gru,1,seed)),
            "lstm-p6-h16"=>Self::Recurrent(Recurrent::c009(Kind::Lstm,6,seed)),
            "lstm-p3-h16"=>Self::Recurrent(Recurrent::c009(Kind::Lstm,3,seed)),
            "lstm-p1-h16"=>Self::Recurrent(Recurrent::c009(Kind::Lstm,1,seed)),
            "cnn-p6-c8"=>Self::Temporal(Temporal::c009(CKind::Cnn,6,seed)),
            "cnn-p3-c8"=>Self::Temporal(Temporal::c009(CKind::Cnn,3,seed)),
            "tcn-p6-c8"=>Self::Temporal(Temporal::c009(CKind::Tcn,6,seed)),
            "tcn-p3-c8"=>Self::Temporal(Temporal::c009(CKind::Tcn,3,seed)),
            "seq2seq-h16"=>Self::Seq2Seq(Seq2Seq::new(16,seed)),
            "seq2seq-h32"=>Self::Seq2Seq(Seq2Seq::new(32,seed)),
            "patchtst-p6-d16"=>Self::Encoder(Encoder::new(6,16,true,seed)),
            "patchtst-p3-d16"=>Self::Encoder(Encoder::new(3,16,true,seed)),
            "attention-p6-d8"=>Self::Encoder(Encoder::new(6,8,false,seed)),
            "attention-p3-d8"=>Self::Encoder(Encoder::new(3,8,false,seed)),
            "attention-p6-d16"=>Self::Encoder(Encoder::new(6,16,false,seed)),
            "ssm-h16"=>Self::StateSpace(StateSpace::new(16,seed)),
            "ssm-h32"=>Self::StateSpace(StateSpace::new(32,seed)),
            "moe-e2-w64"=>Self::Mixture(Mixture::new(2,64,seed)),
            "moe-e4-w64"=>Self::Mixture(Mixture::new(4,64,seed)),
            "moe-e4-w32"=>Self::Mixture(Mixture::new(4,32,seed)),
            _=>Self::Structured(Structured::new(label,seed).ok_or("unknown C009 recipe")?)
        })
    }
    fn parameters(&self)->usize{match self{Self::Dense(m)=>m.parameters(),Self::Residual(m)=>m.parameters(),Self::Recurrent(m)=>m.parameters(),Self::Temporal(m)=>m.parameters(),Self::Seq2Seq(m)=>m.parameters(),Self::Structured(m)=>m.parameters(),Self::Encoder(m)=>m.parameters(),Self::StateSpace(m)=>m.parameters(),Self::Mixture(m)=>m.parameters()}}
    fn finite(&self)->bool{match self{Self::Dense(m)=>m.finite(),Self::Residual(m)=>m.finite(),Self::Recurrent(m)=>m.finite(),Self::Temporal(m)=>m.finite(),Self::Seq2Seq(m)=>m.finite(),Self::Structured(m)=>m.finite(),Self::Encoder(m)=>m.finite(),Self::StateSpace(m)=>m.finite(),Self::Mixture(m)=>m.finite()}}
    fn predict(&self,x:&[f64;D])->Result<[f64;Y],String>{
        let v=match self{Self::Dense(m)=>m.predict(x),Self::Residual(m)=>m.predict(x),Self::Recurrent(m)=>m.predict(x).to_vec(),Self::Temporal(m)=>m.predict(x).to_vec(),Self::Seq2Seq(m)=>m.predict(x),Self::Structured(m)=>m.predict(x),Self::Encoder(m)=>m.predict(x),Self::StateSpace(m)=>m.predict(x),Self::Mixture(m)=>m.predict(x)};
        let y:[f64;Y]=v.try_into().map_err(|_|"output shape")?;
        if y.iter().any(|v|!v.is_finite()||v.abs()>1e6){return Err("numerical prediction bound".into())}Ok(y)
    }
    fn step(&mut self,r:&TrainingRow,opt:&mut Adam)->Result<(f64,bool),String>{
        let(loss,clip)=match self{Self::Dense(m)=>m.adam_step(&r.input.x,&r.normalized_target,opt),Self::Residual(m)=>m.step(&r.input.x,&r.normalized_target,opt),Self::Recurrent(m)=>m.adam_step(&r.input.x,&r.normalized_target,opt),Self::Temporal(m)=>m.adam_step(&r.input.x,&r.normalized_target,opt),Self::Seq2Seq(m)=>m.step(&r.input.x,&r.normalized_target,opt),Self::Structured(m)=>m.step(&r.input.x,&r.normalized_target,opt),Self::Encoder(m)=>m.step(&r.input.x,&r.normalized_target,opt),Self::StateSpace(m)=>m.step(&r.input.x,&r.normalized_target,opt),Self::Mixture(m)=>m.step(&r.input.x,&r.normalized_target,opt)};
        if !loss.is_finite()||!self.finite(){return Err("nonfinite training".into())}Ok((loss,clip))
    }
}
struct Rng(u64);
impl Rng{
    fn next(&mut self)->u64{self.0=self.0.wrapping_add(0x9E3779B97F4A7C15);let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9);z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);z^(z>>31)}
    fn shuffle(&mut self,x:&mut[usize]){for i in (1..x.len()).rev(){let u=(self.next()>>11)as f64/9007199254740992.;x.swap(i,(u*(i+1)as f64)as usize)}}
}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>GUARD{Err("runtime guard".into())}else{Ok(())}}
fn train(model:&mut Model,rows:&[&TrainingRow],epochs:usize,seed:u64,now:Instant,log:Option<(&str,&str)>)->Result<usize,String>{
    let mut opt=Adam::new(model.parameters(),0.001);let mut rng=Rng(seed^0xD1B54A32D192ED03);let mut order:Vec<_>=(0..rows.len()).collect();
    let mut total=0;for e in 0..epochs{rng.shuffle(&mut order);let mut loss=0.;let mut clips=0;
        for &i in &order{guard(now)?;let(v,c)=model.step(rows[i],&mut opt)?;loss+=v;clips+=usize::from(c);}
        if !loss.is_finite(){return Err("nonfinite training loss".into())}
        total+=clips;if let Some((id,label))=log{println!("neural-epoch,{id},{label},{seed},{},{:.10},{clips}",e+1,loss/rows.len()as f64);}
    }Ok(total)
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str,seed:u64)->Result<(),String>{
    if ![42,123,2026].contains(&seed){return Err("seed outside frozen recipe".into())}
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for row in &f.eval{daily.add(&row.target,&row.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    macro_rules! block{($reason:expr)=>{{println!("blocked,{id},{label},{}",$reason);out.flush().map_err(|e|e.to_string())?;return Ok(())}}}
    macro_rules! checked{($e:expr)=>{match $e{Ok(v)=>v,Err(e)=>{block!(e)}}}}
    let rows=subset(&f.train,BUDGET);let n=rows.len();let mut probe=Model::new(label,seed)?;let params=probe.parameters();
    // Upper bound for dense, recurrent, temporal, encoder-decoder and structured block recipes: weights + two Adam buffers +
    // gradients, transient activations (at most 73 frames × 112 doubles) and scratch, row refs, order and predictions.
    let workspace=8*(16*params+65536+4*n+m*(Y+2));
    println!("neural-config,{id},{label},{seed},{n},{EPOCHS},0.001,{params},{workspace}");
    if workspace>2_000_000_000{block!("preflight workspace guard")}
    let now=Instant::now();let p=n.min(128);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    let t=Instant::now();checked!(train(&mut probe,&small,1,seed,now,None));let fp=t.elapsed().as_secs_f64();
    let q=n.min(32);let t=Instant::now();for i in 0..q{checked!(guard(now));std::hint::black_box(checked!(probe.predict(&rows[i*n/q].input.x)));}let pp=t.elapsed().as_secs_f64();drop(probe);
    let estimate=fp*(n*EPOCHS)as f64/p as f64+pp*m as f64/q as f64;
    println!("neural-preflight,{id},{label},{seed},{p},{q},{fp:.6},{pp:.6},{estimate:.6}");
    if 2.*estimate+180.>GUARD{block!("preflight runtime guard")}
    let mut model=Model::new(label,seed)?;let t=Instant::now();let clips=checked!(train(&mut model,&rows,EPOCHS,seed,now,Some((id,label))));let fit=t.elapsed().as_secs_f64();
    let t=Instant::now();let mut predictions=Vec::with_capacity(m);
    for row in &f.eval{checked!(guard(now));let pred=checked!(model.predict(&row.input.x));predictions.push((row.input.origin,f.restore(pred)));}
    println!("neural-fit,{id},{label},{seed},{clips},{fit:.6},{:.6}",t.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;use super::super::OriginInput;
    #[test]fn shuffle_reproducible(){let mut a:Vec<_>=(0..128).collect();let mut b=a.clone();let mut r=Rng(0);assert_eq!(r.next(),0xe220a8397b1dcdaf);Rng(42).shuffle(&mut a);Rng(42).shuffle(&mut b);assert_eq!(a,b);assert_ne!(a,(0..128).collect::<Vec<_>>());a.sort();assert_eq!(a,(0..128).collect::<Vec<_>>());}
    #[test]fn recipes_shape_and_update(){for l in ["mlp-w128","mlp-w256","mlp-128x64","resmlp-w64-b2","resmlp-w128-b2","resmlp-w64-b4"]{
        let r=TrainingRow{input:OriginInput{origin:0,x:[0.1;D],recent:[0.;Y]},target:[0.2;Y],normalized_target:[0.2;Y]};
        let mut a=Model::new(l,42).unwrap();let p=a.predict(&r.input.x).unwrap();assert_eq!(p,Model::new(l,42).unwrap().predict(&r.input.x).unwrap());assert_ne!(p,Model::new(l,123).unwrap().predict(&r.input.x).unwrap());
        let mut opt=Adam::new(a.parameters(),0.001);let first=a.step(&r,&mut opt).unwrap().0;let mut last=first;for _ in 0..30{last=a.step(&r,&mut opt).unwrap().0;}assert!(last<first,"{l}");assert_ne!(p,a.predict(&r.input.x).unwrap());
    }}
}
