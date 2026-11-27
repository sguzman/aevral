//! C012: frozen candidate identities, allowed evaluation requests, chronological budget selection.
use super::TrainingRow;
use std::collections::BTreeSet;

const GRID:&str=include_str!("../../../campaigns/012-optimization/candidate-grid.csv");
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Family { Mlp80, Suffix24, Nhits, Lightgbm, Local, Mars, Arx, Haar }
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct Candidate {pub id:&'static str,pub family:Family,pub configuration:&'static str,pub stochastic:bool,pub track:&'static str}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Rung { One, Two, Three, Final }
impl Rung {
 pub fn parse(s:&str)->Result<Self,String>{match s{"1"=>Ok(Self::One),"2"=>Ok(Self::Two),"3"=>Ok(Self::Three),"final"=>Ok(Self::Final),_=>Err("unknown C012 rung".into())}}
 pub fn budget(self)->usize{match self{Self::One=>2048,Self::Two=>8192,Self::Three|Self::Final=>32768}}
}
pub fn candidates()->Result<Vec<Candidate>,String>{
 let mut lines=GRID.lines();
 if lines.next()!=Some("family,model,track,seed_policy,config_id,configuration,reference"){return Err("C012 grid header mismatch".into())}
 let mut out=Vec::new();let mut seen=BTreeSet::new();let mut counts=[0usize;8];
 for line in lines {
  let p:Vec<_>=line.split(',').collect();
  if p.len()!=7||p[5].is_empty()||p[6].is_empty(){return Err("C012 malformed candidate".into())}
  let (i,family,model,track,stochastic)=match p[0] {
   "O01"=>(0,Family::Mlp80,"mlp80","A-fixed",true),
   "O02"=>(1,Family::Suffix24,"suffix24-mlp","A-derived",true),
   "O03"=>(2,Family::Nhits,"mae-nhits64","A-fixed",true),
   "O04"=>(3,Family::Lightgbm,"lightgbm","A-fixed",false),
   "O05"=>(4,Family::Local,"local-ridge","A-fixed",false),
   "O06"=>(5,Family::Mars,"mars","A-fixed",false),
   "O07"=>(6,Family::Arx,"arx","A-fixed",false),
   "O08"=>(7,Family::Haar,"haar-ridge","A-fixed",false),
   _=>return Err("unknown C012 family".into())
  };
  counts[i]+=1;
  if counts[i]>6||p[1]!=model||p[2]!=track||p[3]!=(if stochastic{"stochastic"}else{"deterministic"})||p[4]!=format!("{}-{:02}",p[0],counts[i])||!seen.insert(p[4]){
   return Err("C012 candidate metadata mismatch".into())
  }
  out.push(Candidate{id:p[4],family,configuration:p[5],stochastic,track});
 }
 if counts!=[6;8]||out.len()!=48{return Err("C012 frozen grid size mismatch".into())}Ok(out)
}
pub fn request(id:&str,rung:Rung,fold:&str,seed:u64)->Result<Candidate,String>{
 let c=candidates()?.into_iter().find(|c|c.id==id).ok_or("unknown frozen C012 candidate")?;
 if rung!=Rung::Final {
  if fold!="2019"||seed!=42{return Err("C012 screening requires 2019 seed42".into())}
 }else if !["2019","2020","2021","2022"].contains(&fold)||![42,123,2026].contains(&seed)||(!c.stochastic&&seed!=42)||(fold=="2019"&&seed==42){
  return Err("C012 invalid or reused annual finalist request".into())
 }
 Ok(c)
}
/// Rung authorization beyond R1 is checked by the frozen promotion matrix in CI.
pub fn indices(n:usize,budget:usize)->Result<Vec<usize>,String>{
 if budget==0||budget>n{return Err("C012 invalid training budget".into())}
 (0..budget).map(|i|i.checked_mul(n).map(|v|v/budget).ok_or("C012 index overflow".into())).collect()
}
pub fn subset(rows:&[TrainingRow],rung:Rung)->Result<Vec<&TrainingRow>,String>{
 if rows.windows(2).any(|w|w[0].input.origin>=w[1].input.origin){return Err("C012 unordered training rows".into())}
 Ok(indices(rows.len(),rung.budget())?.into_iter().map(|i|&rows[i]).collect())
}
/// Chronological validation tail. No selected fit target may overlap the first validation target.
pub fn early_split(origins:&[i64])->Result<(usize,usize),String>{
 if origins.len()<10||origins.windows(2).any(|w|w[0]>=w[1]){return Err("invalid early-stop origins".into())}
 let val_start=origins.len()-origins.len().div_ceil(10);
 let first_val=origins[val_start];
 let fit_end=origins[..val_start].partition_point(|&t|t.checked_add(24*60).is_some_and(|end|end<=first_val));
 if fit_end==0{return Err("empty purged early-stop training".into())}Ok((fit_end,val_start))
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn frozen_grid(){let c=candidates().unwrap();assert_eq!(c.len(),48);assert_eq!(c.iter().filter(|x|x.stochastic).count(),18);assert_eq!(c[0].configuration,"sgd-lr005-e30-b1");assert_eq!(c[47].configuration,"m64-l01");}
 #[test]fn request_gates(){assert!(request("O01-01",Rung::One,"2019",42).is_ok());assert!(request("O09-01",Rung::One,"2019",42).is_err());assert!(request("O01-01",Rung::Two,"2020",42).is_err());assert!(request("O01-01",Rung::Final,"2023",42).is_err());assert!(request("O01-01",Rung::Final,"2019",42).is_err());assert!(request("O01-01",Rung::Final,"2019",123).is_ok());assert!(request("O04-01",Rung::Final,"2020",123).is_err());assert!(request("O04-01",Rung::Final,"2020",42).is_ok());}
 #[test]fn budget_and_order(){assert_eq!(indices(10,4).unwrap(),vec![0,2,5,7]);assert!(indices(10,11).is_err());assert!(indices(10,0).is_err());for r in [Rung::One,Rung::Two,Rung::Three]{let x=indices(121849,r.budget()).unwrap();assert_eq!(x.len(),r.budget());assert!(x.windows(2).all(|w|w[0]<w[1]));assert!(x.last().unwrap()<&121849);}}
 #[test]fn purge_overlaps(){let t:Vec<_>=(0..100).map(|x|x*60).collect();assert_eq!(early_split(&t).unwrap(),(67,90));let sparse:Vec<_>=(0..101).map(|x|x*1440).collect();assert_eq!(early_split(&sparse).unwrap(),(90,90));assert!(early_split(&[1;20]).is_err());}
 #[test]fn total_budget(){assert_eq!(48+24+8+3*11+5*3,128);}
}
