//! C010 F01-F04: fixed-input joint, independent, recursive and one-lag DirRec.
use super::{Adam,D,Y,TrainingRow,Rng,guard,Structured};
use super::super::{OriginInput,load,score,Fold};
use super::super::neighbors::{subset,Cached};
use crate::{dense::Dense,benchmark::Metrics};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
enum Mode{Joint(Dense,Adam),Hits(Structured,Adam),Heads(Vec<(Dense,Adam)>,bool),Recursive(Dense,Adam)}
struct Forecast{mode:Mode,mean:f64,sd:f64}
fn scalar(dims:&[usize],seed:u64)->(Dense,Adam){let m=Dense::new(dims,seed);let o=Adam::new(m.parameters(),0.001);(m,o)}
impl Forecast{
    fn new(label:&str,seed:u64,mean:f64,sd:f64)->Result<Self,String>{
        let mode=match label{
            "joint-mlp64"=>{let(m,o)=scalar(&[D,64,Y],seed);Mode::Joint(m,o)},
            "joint-nhits64"=>{let m=Structured::new("nhits-w64",seed).ok_or("missing N-HiTS")?;let o=Adam::new(m.parameters(),0.001);Mode::Hits(m,o)},
            "perlead-h8"|"perlead-h16"|"dirrec-h8"|"dirrec-h16"=>{let dir=label.starts_with("dirrec");let h=if label.ends_with("16"){16}else{8};Mode::Heads((1..=Y).map(|j|scalar(&[D+usize::from(dir),h,1],seed+j as u64)).collect(),dir)},
            "recursive-h16"|"recursive-h32"=>{let h=if label.ends_with("32"){32}else{16};let(m,o)=scalar(&[D+2,h,1],seed);Mode::Recursive(m,o)},
            _=>return Err("unknown C010 recipe".into())
        };Ok(Self{mode,mean,sd})
    }
    fn parameters(&self)->usize{match &self.mode{Mode::Joint(m,_)|Mode::Recursive(m,_)=>m.parameters(),Mode::Hits(m,_)=>m.parameters(),Mode::Heads(ms,_)=>ms.iter().map(|(m,_)|m.parameters()).sum()}}
    fn updates(&self)->usize{match &self.mode{Mode::Joint(..)|Mode::Hits(..)=>1,_=>Y}}
    fn z0(&self,r:&OriginInput)->f64{(r.recent[Y-1]-self.mean)/self.sd}
    fn predict(&self,r:&OriginInput)->Result<[f64;Y],String>{
        let mut out=[0.;Y];let mut prev=self.z0(r);
        match &self.mode{
            Mode::Joint(m,_)=>out.copy_from_slice(&m.predict(&r.x)),
            Mode::Hits(m,_)=>out.copy_from_slice(&m.predict(&r.x)),
            Mode::Heads(ms,dir)=>for(j,(m,_))in ms.iter().enumerate(){let mut x=r.x.to_vec();if *dir{x.push(prev)}out[j]=m.predict(&x)[0];prev=out[j];},
            Mode::Recursive(m,_)=>for j in 0..Y{let mut x=r.x.to_vec();x.extend([prev,(j+1)as f64/Y as f64]);out[j]=m.predict(&x)[0];prev=out[j];}
        }
        if out.iter().any(|v|!v.is_finite()||v.abs()>1e6){return Err("numerical prediction bound".into())}Ok(out)
    }
    fn step(&mut self,r:&TrainingRow,now:Instant)->Result<(f64,usize),String>{
        let mut prev=self.z0(&r.input);let mut loss=0.;let mut clips=0;
        match &mut self.mode{
            Mode::Joint(m,o)=>{guard(now)?;loss=m.adam_step(&r.input.x,&r.normalized_target,o).0;if !m.finite(){return Err("nonfinite dense".into())}},
            Mode::Hits(m,o)=>{guard(now)?;let(l,c)=m.step(&r.input.x,&r.normalized_target,o);loss=l;clips=usize::from(c);if !m.finite(){return Err("nonfinite hits".into())}},
            Mode::Heads(ms,dir)=>{for(j,(m,o))in ms.iter_mut().enumerate(){guard(now)?;let mut x=r.input.x.to_vec();if *dir{x.push(prev)}let y=r.normalized_target[j];loss+=m.adam_step(&x,&[y],o).0;prev=y;if !m.finite(){return Err("nonfinite head".into())}}loss/=Y as f64;},
            Mode::Recursive(m,o)=>{for j in 0..Y{guard(now)?;let mut x=r.input.x.to_vec();x.extend([prev,(j+1)as f64/Y as f64]);let y=r.normalized_target[j];loss+=m.adam_step(&x,&[y],o).0;prev=y;if !m.finite(){return Err("nonfinite recursive".into())}}loss/=Y as f64;}
        }
        if !loss.is_finite(){return Err("nonfinite loss".into())}Ok((loss,clips))
    }
}
fn train(m:&mut Forecast,rows:&[&TrainingRow],epochs:usize,seed:u64,now:Instant,log:Option<(&str,&str)>)->Result<usize,String>{
    let mut rng=Rng(seed^0xD1B54A32D192ED03);let mut order:Vec<_>=(0..rows.len()).collect();let mut total=0;
    for e in 0..epochs{rng.shuffle(&mut order);let(mut loss,mut clips)=(0.,0);for &i in &order{let(l,c)=m.step(rows[i],now)?;loss+=l;clips+=c;}if !loss.is_finite(){return Err("nonfinite epoch".into())}total+=clips;if let Some((id,label))=log{println!("strategy-epoch,{id},{label},{seed},{},{:.10},{clips}",e+1,loss/rows.len()as f64);}}
    Ok(total)
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
    let rows=subset(&f.train,8192);let n=rows.len();let mut probe=Forecast::new(label,seed,f.target_mean,f.target_sd)?;let params=probe.parameters();let updates=probe.updates();
    let workspace=8*(16*params+65536+4*n+m*(Y+2));
    println!("strategy-config,{id},{label},{seed},{n},12,0.001,{params},{workspace},{}",n*12*updates);
    if workspace>2_000_000_000{block!("preflight workspace guard")}
    let now=Instant::now();let p=n.min(128);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    let t=Instant::now();checked!(train(&mut probe,&small,1,seed,now,None));let fp=t.elapsed().as_secs_f64();
    let q=n.min(32);let t=Instant::now();for i in 0..q{checked!(guard(now));std::hint::black_box(checked!(probe.predict(&rows[i*n/q].input)));}let pp=t.elapsed().as_secs_f64();drop(probe);
    let estimate=fp*(n*12)as f64/p as f64+pp*m as f64/q as f64;
    println!("strategy-preflight,{id},{label},{seed},{p},{q},{fp:.6},{pp:.6},{estimate:.6}");
    if 2.*estimate+180.>1400.{block!("preflight runtime guard")}
    let mut model=Forecast::new(label,seed,f.target_mean,f.target_sd)?;let t=Instant::now();let clips=checked!(train(&mut model,&rows,12,seed,now,Some((id,label))));let fit=t.elapsed().as_secs_f64();
    let t=Instant::now();let mut predictions=Vec::with_capacity(m);
    for row in &f.eval{checked!(guard(now));predictions.push((row.input.origin,f.restore(checked!(model.predict(&row.input)))));}
    println!("strategy-fit,{id},{label},{seed},{clips},{fit:.6},{:.6}",t.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn row()->TrainingRow{TrainingRow{input:OriginInput{origin:0,x:[0.1;D],recent:[10.;Y]},target:[12.;Y],normalized_target:[0.4;Y]}}
    #[test]fn all_recipes_seed_shape_and_overfit(){let r=row();for label in ["joint-mlp64","joint-nhits64","perlead-h8","perlead-h16","recursive-h16","recursive-h32","dirrec-h8","dirrec-h16"]{let mut m=Forecast::new(label,42,10.,5.).unwrap();let before=m.predict(&r.input).unwrap();assert_eq!(before,Forecast::new(label,42,10.,5.).unwrap().predict(&r.input).unwrap());let first=m.step(&r,Instant::now()).unwrap().0;let mut last=first;for _ in 0..80{last=m.step(&r,Instant::now()).unwrap().0}assert!(last<first,"{label}");assert_ne!(before,m.predict(&r.input).unwrap());}}
    #[test]fn free_running_matches_manual_unroll(){let r=row();for label in ["recursive-h16","dirrec-h8"]{let m=Forecast::new(label,42,10.,5.).unwrap();let p=m.predict(&r.input).unwrap();let mut prev=0.;for j in 0..Y{let mut x=r.input.x.to_vec();x.push(prev);let v=match &m.mode{Mode::Recursive(d,_)=>{x.push((j+1)as f64/Y as f64);d.predict(&x)[0]},Mode::Heads(h,_)=>h[j].0.predict(&x)[0],_=>unreachable!()};assert_eq!(v,p[j]);prev=v;}}}
    #[test]fn joint_controls_match_original_backbones(){let r=row();let mut m=Forecast::new("joint-mlp64",42,10.,5.).unwrap();let mut d=Dense::new(&[D,64,Y],42);let mut a=Adam::new(d.parameters(),0.001);assert_eq!(m.predict(&r.input).unwrap().to_vec(),d.predict(&r.input.x));m.step(&r,Instant::now()).unwrap();d.adam_step(&r.input.x,&r.normalized_target,&mut a);assert_eq!(m.predict(&r.input).unwrap().to_vec(),d.predict(&r.input.x));}
}
