//! Pooled C015 ranking from independently verified per-fold sums, never averages of fold MAEs.
use std::{collections::{BTreeMap,BTreeSet},fs::{self,File},io::{BufRead,BufReader,BufWriter,Write},path::Path};
const FIT:&str=include_str!("fit-matrix.csv");
const COMB:&str=include_str!("combination-matrix.csv");
const COHORTS:&str=include_str!("../../results/C015/cohorts.csv");
#[derive(Clone,Copy,Debug,Default)]
struct S{n:u64,ae:f64,se:f64}
impl S{
 fn add(&mut self,b:S){self.n+=b.n;self.ae+=b.ae;self.se+=b.se;}
 fn mae(self)->f64{self.ae/self.n as f64}
 fn rmse(self)->f64{(self.se/self.n as f64).sqrt()}
}
type Key=(String,usize);
fn num(v:&str)->f64{let x=v.parse::<f64>().unwrap();assert!(x.is_finite()&&x>=0.);x}
fn targets(y:u16)->u64{COHORTS.lines().skip(1).find_map(|l|{let p:Vec<_>=l.split(',').collect();if p[0].parse::<u16>().unwrap()==y{Some(p[4].parse::<u64>().unwrap()*24)}else{None}}).unwrap()}
fn read_metrics(path:&Path,expected:u64)->BTreeMap<Key,S>{
 let mut lines=BufReader::new(File::open(path).unwrap()).lines();
 assert_eq!(lines.next().unwrap().unwrap(),"group,index,targets,absolute_sum,squared_sum,mae_c,rmse_c");
 let mut out=BTreeMap::new();
 for l in lines{let l=l.unwrap();let p:Vec<_>=l.split(',').collect();assert_eq!(p.len(),7);let i=p[1].parse::<usize>().unwrap();let n=p[2].parse::<u64>().unwrap();assert!(n>0);
  match p[0]{"all"=>assert_eq!(i,0),"lead"=>assert!((1..=24).contains(&i)),"month"=>assert!((1..=12).contains(&i)),"season"=>assert!((1..=4).contains(&i)),_=>panic!("unexpected group")}
  let s=S{n,ae:num(p[3]),se:num(p[4])};assert!((s.mae()-num(p[5])).abs()<1e-9);assert!((s.rmse()-num(p[6])).abs()<1e-9);assert!(out.insert((p[0].into(),i),s).is_none());
 }
 assert_eq!(out[&("all".into(),0)].n,expected);
 for h in 1..=24{assert_eq!(out[&("lead".into(),h)].n,expected/24);}
 for g in ["lead","month","season"]{let mut s=S::default();for((k,_),v)in &out{if k==g{s.add(*v)}}assert_eq!(s.n,expected);let a=out[&("all".into(),0)];assert!((s.mae()-a.mae()).abs()<1e-9);assert!((s.rmse()-a.rmse()).abs()<1e-9);}
 out
}
#[derive(Clone,Copy,Debug)]
struct Cost{fit_predict:f64,rss_kib:f64}
fn read_cost(path:&Path)->Cost{
 let t=fs::read_to_string(path).unwrap();let p:Vec<_>=t.trim().split(',').collect();assert_eq!(p.len(),2);
 Cost{fit_predict:num(p[0]),rss_kib:num(p[1])}
}
fn median(mut x:Vec<f64>)->f64{assert!(!x.is_empty());x.sort_by(f64::total_cmp);let i=x.len()/2;if x.len()%2==0{(x[i-1]+x[i])/2.}else{x[i]}}
#[derive(Debug)]
struct Entry{id:String,seed:u64,folds:usize,s:S,compute:Option<f64>,rss:Option<f64>,groups:BTreeMap<Key,S>}
fn dominates(a:&Entry,b:&Entry)->bool{
 let (Some(x),Some(y))=(a.compute,b.compute)else{return false};
 a.s.mae()<=b.s.mae()&&x<=y&&(a.s.mae()<b.s.mae()||x<y)
}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),3,"rank VERIFIED_REGISTRY OUT");let root=Path::new(&a[1]);let out=Path::new(&a[2]);fs::create_dir_all(out).unwrap();
 // Matrix is the only allowable set; missing audited metrics remain incomplete.
 let mut ids:BTreeSet<(String,u64)>=FIT.lines().skip(1).map(|l|{let p:Vec<_>=l.split(',').collect();(p[1].into(),p[3].parse().unwrap())}).collect();
 for l in COMB.lines().skip(1){let p:Vec<_>=l.split(',').collect();ids.insert((p[1].into(),42));}
 ids.insert(("daily".into(),42));
 let years=[2023u16,2024,2025,2026];
 let mut rows=Vec::new();
 let mut annual=BufWriter::new(File::create(out.join("annual.csv")).unwrap());
 writeln!(annual,"fold,id,seed,targets,mae,rmse,fit_predict_s,rss_kib").unwrap();
 for(id,seed)in ids{
  let mut groups:BTreeMap<Key,S>=BTreeMap::new();let(mut folds,mut costs,mut rss)=(0,Vec::new(),Vec::new());
  for y in years{let key=if id=="daily"{format!("{y}-daily")}else{format!("{y}-{id}-{seed}")};let path=root.join(format!("{key}.csv"));if !path.is_file(){continue}
   let metrics=read_metrics(&path,targets(y));let s=metrics[&("all".into(),0)];
   let cost=root.join(format!("{key}-cost.csv"));let c=if cost.is_file(){Some(read_cost(&cost))}else{None};
   if let Some(c)=c{costs.push(c.fit_predict);rss.push(c.rss_kib);}
   let cf=c.map(|c|format!("{:.9}",c.fit_predict)).unwrap_or("NA".into());let cr=c.map(|c|format!("{:.0}",c.rss_kib)).unwrap_or("NA".into());
   writeln!(annual,"{y},{id},{seed},{},{:.12},{:.12},{cf},{cr}",s.n,s.mae(),s.rmse()).unwrap();
   folds+=1;for(k,s)in metrics{groups.entry(k).or_default().add(s);}
  }
  let s=groups.get(&("all".into(),0)).copied().unwrap_or_default();
  let compute=if costs.len()==4{Some(median(costs))}else{None};let rss=if rss.len()==4{Some(median(rss))}else{None};
  rows.push(Entry{id,seed,folds,s,compute,rss,groups});
 }
 annual.flush().unwrap();
 let mut pooled=BufWriter::new(File::create(out.join("pooled.csv")).unwrap());
 writeln!(pooled,"id,seed,folds,targets,mae,rmse,median_fit_predict_s,median_rss_kib,status").unwrap();
 let mut diag=BufWriter::new(File::create(out.join("diagnostics.csv")).unwrap());
 writeln!(diag,"id,seed,group,index,targets,mae,rmse").unwrap();
 for r in &rows{
  let (mae,rmse)=if r.folds==4{(format!("{:.12}",r.s.mae()),format!("{:.12}",r.s.rmse()))}else{("NA".into(),"NA".into())};
  let c=r.compute.map(|v|format!("{v:.9}")).unwrap_or("NA".into());let m=r.rss.map(|v|format!("{v:.0}")).unwrap_or("NA".into());let status=if r.folds==4{"complete"}else{"incomplete"};
  writeln!(pooled,"{},{},{},{},{mae},{rmse},{c},{m},{status}",r.id,r.seed,r.folds,r.s.n).unwrap();
  if r.folds==4{for((g,i),s)in &r.groups{writeln!(diag,"{},{},{g},{i},{},{:.12},{:.12}",r.id,r.seed,s.n,s.mae(),s.rmse()).unwrap();}}
 }
 pooled.flush().unwrap();diag.flush().unwrap();
 let mut ranking=BufWriter::new(File::create(out.join("ranking.csv")).unwrap());
 writeln!(ranking,"track,rank,id,mae,rmse,median_fit_predict_s,pareto_scope,pareto").unwrap();
 let mut complete:Vec<_>=rows.iter().filter(|r|r.seed==42&&r.folds==4&&r.id!="daily"&&r.id!="W").collect();
 // Exact f64 pooled metrics, deterministic tie-break by compute then ID.
 complete.sort_by(|a,b|a.s.mae().total_cmp(&b.s.mae()).then(a.s.rmse().total_cmp(&b.s.rmse())).then(a.compute.unwrap_or(f64::INFINITY).total_cmp(&b.compute.unwrap_or(f64::INFINITY))).then(a.id.cmp(&b.id)));
 for(i,r)in complete.iter().enumerate(){
  let scope=if r.id.starts_with('E')||r.id=="P13"{"combination-incremental"}else{"single-model"};
  let front=if r.compute.is_none(){"NA"}else if complete.iter().any(|x|{let other=if x.id.starts_with('E')||x.id=="P13"{"combination-incremental"}else{"single-model"};scope==other&&dominates(x,r)}){"no"}else{"yes"};
  let c=r.compute.map(|v|format!("{v:.9}")).unwrap_or("NA".into());
  writeln!(ranking,"A,{},{},{:.12},{:.12},{c},{scope},{front}",i+1,r.id,r.s.mae(),r.s.rmse()).unwrap();
 }
 if let Some(r)=rows.iter().find(|r|r.id=="W"&&r.seed==42&&r.folds==4){let c=r.compute.map(|v|format!("{v:.9}")).unwrap_or("NA".into());writeln!(ranking,"B,1,W,{:.12},{:.12},{c},expanded-input,NA",r.s.mae(),r.s.rmse()).unwrap();}
 ranking.flush().unwrap();
 let mut sens=BufWriter::new(File::create(out.join("seed-sensitivity.csv")).unwrap());
 writeln!(sens,"id,seed42_mae,seed123_mae,seed2026_mae,range,population_sd,status").unwrap();
 for id in ["M","S","N","W"]{
  let vals:Vec<_>=[42,123,2026].into_iter().map(|seed|rows.iter().find(|r|r.id==id&&r.seed==seed&&r.folds==4).map(|r|r.s.mae())).collect();
  let cells:Vec<_>=vals.iter().map(|x|x.map(|v|format!("{v:.12}")).unwrap_or("NA".into())).collect();
  if vals.iter().all(Option::is_some){let x:Vec<_>=vals.iter().map(|x|x.unwrap()).collect();let mean=x.iter().sum::<f64>()/3.;let sd=(x.iter().map(|v|(v-mean).powi(2)).sum::<f64>()/3.).sqrt();let range=x.iter().copied().fold(f64::NEG_INFINITY,f64::max)-x.iter().copied().fold(f64::INFINITY,f64::min);writeln!(sens,"{id},{},{range:.12},{sd:.12},complete",cells.join(",")).unwrap();}
  else{writeln!(sens,"{id},{},NA,NA,incomplete",cells.join(",")).unwrap();}
 }
 sens.flush().unwrap();
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn weighted(){let mut a=S{n:1,ae:10.,se:100.};a.add(S{n:9,ae:0.,se:0.});assert_eq!(a.mae(),1.);assert_eq!(a.rmse(),10f64.sqrt());}
 #[test]fn med(){assert_eq!(median(vec![4.,1.,3.,2.]),2.5);}
 #[test]fn cohort_counts(){assert_eq!(targets(2023),8026*24);assert_eq!(targets(2026),6521*24);}
}
