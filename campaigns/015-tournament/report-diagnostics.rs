//! Pooled proper scores and matched/natural coverage from the frozen audited registry.
use std::{collections::BTreeMap,fs::{self,File},io::{BufWriter,Write},path::Path};
const COHORTS:&str=include_str!("../../results/C015/cohorts.csv");
fn num(s:&str)->f64{let x=s.parse::<f64>().unwrap();assert!(x.is_finite()&&x>=0.);x}
#[derive(Default)]
struct Sum{n:u64,v:[f64;24]}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),3,"report-diagnostics VERIFIED_REGISTRY FINAL_DIR");let reg=Path::new(&a[1]);let out=Path::new(&a[2]);
 let mut coverage=BufWriter::new(File::create(out.join("coverage.csv")).unwrap());
 writeln!(coverage,"fold,possible_origins,temperature_complete,matched_origins,natural_origins,matched_fraction,natural_fraction,suffix24_seed42_natural_mae,suffix24_seed42_natural_rmse").unwrap();
 let mut total=[0u64;4];let mut nat=Sum::default();let mut nat_folds=0;
 for line in COHORTS.lines().skip(1){
  let p:Vec<_>=line.split(',').collect();let y=p[0];let possible=p[6].parse::<u64>().unwrap();let temp=p[7].parse::<u64>().unwrap();let matched=p[4].parse::<u64>().unwrap();let natural=p[5].parse::<u64>().unwrap();
  for(i,n)in [possible,temp,matched,natural].into_iter().enumerate(){total[i]+=n;}
  let path=reg.join(format!("{y}-S-42-natural.csv"));
  let (mae,rmse)=if path.is_file(){
   let raw=fs::read_to_string(path).unwrap();let all=raw.lines().find(|l|l.starts_with("all,0,")).unwrap();let p:Vec<_>=all.split(',').collect();assert_eq!(p.len(),7);let n=p[2].parse::<u64>().unwrap();assert_eq!(n,natural*24);nat.n+=n;nat.v[0]+=num(p[3]);nat.v[1]+=num(p[4]);nat_folds+=1;(p[5].to_owned(),p[6].to_owned())
  }else{("NA".to_owned(),"NA".to_owned())};
  writeln!(coverage,"{y},{possible},{temp},{matched},{natural},{:.12},{:.12},{mae},{rmse}",matched as f64/possible as f64,natural as f64/possible as f64).unwrap();
 }
 let [p,t,m,n]=total;
 let (mae,rmse)=if nat_folds==4{(format!("{:.12}",nat.v[0]/nat.n as f64),format!("{:.12}",(nat.v[1]/nat.n as f64).sqrt()))}else{("NA".into(),"NA".into())};
 writeln!(coverage,"pooled,{p},{t},{m},{n},{:.12},{:.12},{mae},{rmse}",m as f64/p as f64,n as f64/p as f64).unwrap();coverage.flush().unwrap();
 let mut sums:BTreeMap<(String,usize),Sum>=BTreeMap::new();let mut folds=0;
 for y in 2023..=2026{
  let path=out.join(format!("{y}-P13-probabilistic.csv"));if !path.is_file(){continue}
  let raw=fs::read_to_string(path).unwrap();let mut l=raw.lines();assert_eq!(l.next().unwrap().split(',').count(),27);let mut keys=std::collections::BTreeSet::new();
  for line in l{
   let p:Vec<_>=line.split(',').collect();assert_eq!(p.len(),27);let k=(p[0].to_owned(),p[1].parse::<usize>().unwrap());assert!(keys.insert(k.clone()));
   let s=sums.entry(k).or_default();let n=p[2].parse::<u64>().unwrap();assert!(n>0);s.n+=n;for(i,v)in s.v.iter_mut().enumerate(){*v+=num(p[i+3]);}
  }
  assert!(keys.contains(&("all".into(),0)));folds+=1;
 }
 let mut w=BufWriter::new(File::create(out.join("probabilistic-pooled.csv")).unwrap());
 write!(w,"group,index,folds,targets,mae,rmse,crps").unwrap();for i in 1..=9{write!(w,",pinball{i}").unwrap()}for c in [50,80,90,95]{write!(w,",coverage{c},width{c},interval_score{c}").unwrap()}writeln!(w,",status").unwrap();
 if folds==4{assert_eq!(sums[&("all".into(),0)].n,m*24);for((g,i),s)in sums{let n=s.n as f64;write!(w,"{g},{i},{folds},{},{:.12},{:.12},{:.12}",s.n,s.v[0]/n,(s.v[1]/n).sqrt(),s.v[2]/n).unwrap();for v in &s.v[3..]{write!(w,",{:.12}",v/n).unwrap()}writeln!(w,",complete").unwrap();}}
 else{write!(w,"all,0,{folds},{}",sums.get(&("all".into(),0)).map(|s|s.n).unwrap_or(0)).unwrap();for _ in 0..24{write!(w,",NA").unwrap()}writeln!(w,",incomplete").unwrap();}
 w.flush().unwrap();
}
