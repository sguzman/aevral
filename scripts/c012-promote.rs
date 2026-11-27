//! C012 promotion gates. Standalone, std-only; no fit or prediction.
use std::{collections::{BTreeMap,BTreeSet},fs,path::Path};
const HEADER:&str="rung,config,fold,seed,track,fit_budget,origins,mae_c,rmse_c,natural_origins,natural_mae_c,natural_rmse_c,fit_s,predict_s,fit_predict_s,parameters,wall_s,max_rss_kib,code_sha,ci_run,job_id,status,rescore,archive";
fn matrix(s:&str)->Result<BTreeSet<(String,String,String)>,String>{
 let mut l=s.lines();if l.next()!=Some("config,fold,seed"){return Err("matrix header".into())}
 let mut a=BTreeSet::new();for row in l{let v:Vec<_>=row.split(',').collect();if v.len()!=3||!a.insert((v[0].into(),v[1].into(),v[2].into())){return Err("invalid matrix row".into())}}Ok(a)
}
fn finite(s:&str)->Result<f64,String>{let v=s.parse::<f64>().map_err(|_|"invalid metric")?;if v.is_finite()&&v>=0.{Ok(v)}else{Err("nonfinite or negative metric".into())}}
fn promote(grid:&str,registry:&str,prior:&str,rung:&str)->Result<String,String>{
 let (limit,budget)=match rung{"1"=>(3,"2048"),"2"=>(1,"8192"),"3"=>(1,"32768"),_=>return Err("invalid promotion rung".into())};
 let mut g=BTreeMap::new();let mut gl=grid.lines();if gl.next()!=Some("family,model,track,seed_policy,config_id,configuration,reference"){return Err("grid header".into())}
 for l in gl{let p:Vec<_>=l.split(',').collect();if p.len()!=7||g.insert(p[4],(p[0],p[2],p[3])).is_some(){return Err("grid row".into())}}
 if g.len()!=48{return Err("grid size".into())}
 let expected=matrix(prior)?;let mut seen=BTreeSet::new();let mut jobs=BTreeSet::new();let mut ranks:BTreeMap<&str,Vec<(&str,f64,f64,f64)>>=BTreeMap::new();
 let mut lines=registry.lines();if lines.next()!=Some(HEADER){return Err("registry header".into())}
 for l in lines{
  let p:Vec<_>=l.split(',').collect();if p.len()!=24{return Err("registry width".into())}
  if !jobs.insert(p[20])||p[20].parse::<u64>().ok().filter(|x|*x>0).is_none(){return Err("duplicate/invalid CI job ID".into())}
  if p[0]!=rung{continue}
  let key=(p[1].to_owned(),p[2].to_owned(),p[3].to_owned());
  let (family,track,_)=*g.get(p[1]).ok_or("unknown config")?;
  if !expected.contains(&key)||!seen.insert(key)||p[2]!="2019"||p[3]!="42"||p[4]!=track||p[5]!=budget||p[18].len()!=40||p[19].parse::<u64>().ok().filter(|x|*x>0).is_none()||p[23].is_empty(){return Err("registry identity mismatch".into())}
  if finite(p[16])?<=0.||p[17].parse::<u64>().ok().filter(|x|*x>0).is_none(){return Err("missing resource diagnostics".into())}
  match p[21]{
   "measured"=>{if p[22]!="ci-rescored"||p[6]!="7455"{return Err("unverified score".into())}
    let a=finite(p[7])?;let b=finite(p[8])?;let t=finite(p[14])?;if (finite(p[12])?+finite(p[13])?-t).abs()>0.000002{return Err("cost sum mismatch".into())}
    ranks.entry(family).or_default().push((p[1],a,b,t));
   },
   "blocked"=>{if p[22]!="not-applicable"||p[6..16].iter().any(|v|!v.is_empty()){return Err("invalid blocked outcome".into())}},
   _=>return Err("unresolved status".into())
  }
 }
 if seen!=expected{return Err("incomplete previous rung".into())}
 let mut out=String::from("config,fold,seed\n");
 for(_,rows)in &mut ranks{
  rows.sort_by(|a,b|a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)).then(a.3.total_cmp(&b.3)).then(a.0.cmp(b.0)));
  for &(id,_,_,_) in rows.iter().take(limit){
   if rung!="3"{out.push_str(&format!("{id},2019,42\n"));}else{
    let stochastic=g[id].2=="stochastic";for fold in ["2019","2020","2021","2022"]{for seed in if stochastic{&["42","123","2026"][..]}else{&["42"][..]}{if fold!="2019"||*seed!="42"{out.push_str(&format!("{id},{fold},{seed}\n"));}}}
   }
  }
 }
 Ok(out)
}
fn main()->Result<(),String>{
 let args:Vec<_>=std::env::args().collect();if args.len()!=2{return Err("Usage: c012-promote PREVIOUS_RUNG".into())}
 let r=&args[1];let root=Path::new("campaigns/012-optimization");let read=|p:&Path|fs::read_to_string(p).map_err(|e|e.to_string());
 print!("{}",promote(&read(&root.join("candidate-grid.csv"))?,&read(Path::new("results/C012/registry.csv"))?,&read(&root.join(format!("rung-{r}.csv")))?,r)?);Ok(())
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn rejects_incomplete_and_invalid(){let grid=include_str!("../campaigns/012-optimization/candidate-grid.csv");let m="config,fold,seed\nO01-01,2019,42\n";assert!(promote(grid,HEADER,m,"1").is_err());assert!(promote(grid,HEADER,m,"5").is_err());assert!(matrix("config,fold,seed\nO01-01,2019,42\nO01-01,2019,42\n").is_err());}
 #[test]fn promotes_only_ranked_measured(){let grid=include_str!("../campaigns/012-optimization/candidate-grid.csv");let m="config,fold,seed\nO01-01,2019,42\nO01-02,2019,42\nO01-03,2019,42\nO01-04,2019,42\nO01-05,2019,42\n";let mut r=format!("{HEADER}\n");for(i,mae)in [(1,2.),(2,1.),(3,1.5),(4,1.5)]{r.push_str(&format!("1,O01-0{i},2019,42,A-fixed,2048,7455,{mae},2,7455,{mae},2,1,1,2,6744,10,100,{},123,{i},measured,ci-rescored,rung-1/test{i}.txt\n","a".repeat(40)));}r.push_str(&format!("1,O01-05,2019,42,A-fixed,2048,,,,,,,,,,,10,100,{},123,5,blocked,not-applicable,rung-1/block.txt\n","a".repeat(40)));assert_eq!(promote(grid,&r,m,"1").unwrap(),"config,fold,seed\nO01-02,2019,42\nO01-03,2019,42\nO01-04,2019,42\n");}
}
