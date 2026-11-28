//! Pool archived lead/month errors, averaging seeds within year first.
use std::{collections::{BTreeMap,BTreeSet},fs,path::Path};
#[derive(Default)]
struct Acc { n:f64, ae:f64, se:f64 }
impl Acc { fn add(&mut self,n:f64,a:f64,r:f64){self.n+=n;self.ae+=n*a;self.se+=n*r*r;} }
fn num(s:&str)->f64{let x:f64=s.parse().unwrap();assert!(x.is_finite()&&x>=0.);x}
fn main(){
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),3,"usage: aggregate-diagnostics RESULTS_DIR OUTPUT_DIR");
 let root=Path::new(&args[1]);let raw=fs::read_to_string(root.join("registry.csv")).unwrap();
 let rows:Vec<Vec<&str>>=raw.lines().skip(1).map(|l|l.split(',').collect()).filter(|r|matches!(r[0],"3"|"final")).collect();
 assert_eq!(rows.len(),56);
 let mut counts=BTreeMap::new();for r in &rows {if r[21]=="measured"{*counts.entry((r[1],r[2])).or_insert(0usize)+=1;}}
 assert_eq!(counts.values().sum::<usize>(),53);
 let mut accum:BTreeMap<(&str,&str,&str,usize),Acc>=BTreeMap::new();
 let mut foldout=String::from("config,fold,seed,cohort,kind,index,origins,mae_c,rmse_c\n");
 for r in rows.iter().filter(|r|r[21]=="measured"){
  let file=fs::read_to_string(root.join(r[23])).unwrap();
  let seeds=counts[&(r[1],r[2])] as f64;
  for (cohort,lead,month,ncol,acol,rcol) in [("matched","lead","month",6,7,8),("natural","natural-lead","natural-month",9,10,11)]{
   if cohort=="natural" && r[4]!="A-derived"{continue;}
   let n=num(r[ncol]);let mut seen=BTreeSet::new();let mut total=0.;let mut check=Acc::default();let mut leadcheck=Acc::default();
   for line in file.lines(){
    let p:Vec<_>=line.split(',').collect();
    let (kind,limit,w,a,e)=if p[0]==lead {assert_eq!(p.len(),6);("lead",24,n,num(p[4]),num(p[5]))}else if p[0]==month{assert_eq!(p.len(),7);let w=num(p[4]);total+=w;("month",12,w,num(p[5]),num(p[6]))}else{continue};
    assert_eq!(p[1],r[2]);assert_eq!(p[2],r[1]);
    let index:usize=p[3].parse().unwrap();assert!((1..=limit).contains(&index));assert!(seen.insert((kind,index)));assert!(w>0.);
    if kind=="month"{check.add(w,a,e);}else{leadcheck.add(w,a,e);}
    accum.entry((r[1],cohort,kind,index)).or_default().add(w/seeds,a,e);
    foldout+=&format!("{},{},{},{cohort},{kind},{index},{w:.0},{a:.8},{e:.8}\n",r[1],r[2],r[3]);
   }
   assert_eq!(seen.len(),36);assert_eq!(total,n);
   for c in [check,leadcheck]{assert!((c.ae/c.n-num(r[acol])).abs()<1e-7,"MAE mismatch {}",r[23]);assert!(((c.se/c.n).sqrt()-num(r[rcol])).abs()<1e-7,"RMSE mismatch {}",r[23]);}
  }
 }
 let mut out=String::from("config,cohort,kind,index,origins,mae_c,rmse_c,coverage\n");
 assert_eq!(accum.len(),9*36);
 for ((id,cohort,kind,index),v) in accum{
  let coverage=if id=="O03-04"{"available-case"}else{"complete"};
  out+=&format!("{id},{cohort},{kind},{index},{:.0},{:.8},{:.8},{coverage}\n",v.n,v.ae/v.n,(v.se/v.n).sqrt());
 }
 let dest=Path::new(&args[2]);fs::create_dir_all(dest).unwrap();fs::write(dest.join("annual-diagnostics.csv"),out).unwrap();fs::write(dest.join("annual-diagnostic-fold-seeds.csv"),foldout).unwrap();
}
