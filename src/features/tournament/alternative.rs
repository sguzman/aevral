//! C008 D08/D09 screen and annual runner.
use super::{load,score,Fold,TrainingRow,OriginInput,Y};
use super::neighbors::{subset,Cached};
use super::{markov,symbolic};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
#[derive(Clone,Copy)]enum Recipe{Markov(usize,bool),Symbolic(usize,usize)}
fn config(s:&str)->Result<Recipe,String>{Ok(match s{
"markov-b8-pooled"=>Recipe::Markov(8,false),"markov-b8-hour"=>Recipe::Markov(8,true),"markov-b16-pooled"=>Recipe::Markov(16,false),"markov-b16-hour"=>Recipe::Markov(16,true),
"symbolic-depth1-t4"=>Recipe::Symbolic(1,4),"symbolic-depth1-t8"=>Recipe::Symbolic(1,8),"symbolic-depth2-t4"=>Recipe::Symbolic(2,4),"symbolic-depth2-t8"=>Recipe::Symbolic(2,8),_=>return Err("unknown alternative recipe".into())})}
enum Model{Markov(markov::Model),Symbolic(symbolic::Model)}
impl Model{
fn fit(rows:&[&TrainingRow],r:Recipe,now:Instant)->Result<Self,String>{match r{Recipe::Markov(b,h)=>Ok(Self::Markov(markov::Model::fit(rows,b,h,now)?)),Recipe::Symbolic(d,t)=>Ok(Self::Symbolic(symbolic::Model::fit(rows,d,t,now)?))}}
fn predict(&self,x:&OriginInput)->Result<[f64;Y],String>{match self{Self::Markov(m)=>m.predict(x),Self::Symbolic(m)=>m.predict(&x.x)}}
fn diagnostics(&self,id:&str,label:&str)->Result<(),String>{match self{Self::Markov(m)=>{println!("markov-config,{id},{label},{},{},{},{:.17e}",m.pairs,m.gaps,m.empty,m.row_error);println!("markov-thresholds,{id},{label},{:?}",m.thresholds);println!("markov-means,{id},{label},{:?}",m.means);for(i,v)in m.trans.iter().enumerate(){println!("markov-transition,{id},{label},{i},{:?}",v);}Ok(())},Self::Symbolic(m)=>m.diagnostics(id,label)}}
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let r=config(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for row in &f.eval{daily.add(&row.target,&row.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    macro_rules! block{($reason:expr)=>{{println!("blocked,{id},{label},{}",$reason);out.flush().map_err(|e|e.to_string())?;return Ok(())}}}
    let rows=subset(&f.train,match r{Recipe::Markov(..)=>n,Recipe::Symbolic(..)=>4096});let n=rows.len();
    let (workspace,p)=match r{Recipe::Markov(b,_)=>(8*(n*4+6*b*b+m*(Y+1)+4096),n.min(8192)),Recipe::Symbolic(d,_)=>(8*(n*(if d==1{900}else{1924})+n*120+200000+m*(Y+1)),n.min(256))};
    if workspace>2_000_000_000{block!("preflight workspace guard")}
    let now=Instant::now();let small:Vec<_>=match r{Recipe::Markov(..)=>rows[..p].to_vec(),_ =>(0..p).map(|i|rows[i*n/p]).collect()};
    let timer=Instant::now();let(sm,_)=match Model::fit(&small,r,now){Ok(v)=>(v,()),Err(e)=>{block!(format!("preflight: {e}"))}};
    let fp=timer.elapsed().as_secs_f64();let q=n.min(32);let timer=Instant::now();for i in 0..q{match sm.predict(&rows[i*n/q].input){Ok(v)=>{std::hint::black_box(v);},Err(e)=>{block!(format!("preflight: {e}"))}}}
    let pp=timer.elapsed().as_secs_f64();drop(sm);let ratio=match r{Recipe::Markov(..)=>(n as f64/p as f64)*(n as f64).ln()/(p as f64).ln(),_=>n as f64/p as f64};let estimate=fp*ratio+pp*(m+128)as f64/q as f64;
    println!("alt-preflight,{id},{label},{n},{p},{q},{workspace},{fp:.6},{pp:.6},{estimate:.6}");
    if 2.*estimate+180.>1400.{block!("preflight runtime guard")}
    let timer=Instant::now();let model=match Model::fit(&rows,r,now){Ok(v)=>v,Err(e)=>{block!(e)}};let fit=timer.elapsed().as_secs_f64();
    let timer=Instant::now();let mut predictions=Vec::with_capacity(m);
    for row in &f.eval{if now.elapsed().as_secs_f64()>1400.{block!("runtime guard")}let p=match model.predict(&row.input){Ok(v)=>v,Err(e)=>{block!(e)}};if p.iter().any(|v|!v.is_finite()||v.abs()>1e6){block!("numerical prediction bound")}predictions.push((row.input.origin,f.restore(p)));}
    println!("alt-config,{id},{label},{n},{fit:.6},{:.6}",timer.elapsed().as_secs_f64());model.diagnostics(id,label)?;
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
