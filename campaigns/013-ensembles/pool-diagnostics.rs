//! Reproduce the archived C013 pooled diagnostics from rounded fold metrics.
use std::{collections::{BTreeMap,BTreeSet},fs};
#[derive(Default)]struct Acc{n:usize,ae:f64,se:f64,years:BTreeSet<u16>}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),3);
 let raw=fs::read_to_string(&a[1]).unwrap();
 let mut ids=Vec::new();let mut sums:BTreeMap<(&str,&str,usize),Acc>=BTreeMap::new();
 for line in raw.lines(){
  let p:Vec<_>=line.split(',').collect();let kind=p[0];let year:u16=p[1].parse().unwrap();assert!(matches!(year,2021|2022));
  let id=p[2];if !ids.contains(&id){ids.push(id);}
  let (idx,k)=match kind{"score"=>{assert_eq!(p.len(),8);(0,3)},"lead"|"month"=>{assert_eq!(p.len(),7);(p[3].parse::<usize>().unwrap(),4)},_=>panic!("kind")};
  assert!(match kind{"score"=>idx==0,"lead"=>(1..=24).contains(&idx),_=>(1..=12).contains(&idx)});
  let n:usize=p[k].parse().unwrap();let mae:f64=p[k+1].parse().unwrap();let rmse:f64=p[k+2].parse().unwrap();
  assert!(n>0&&mae.is_finite()&&rmse.is_finite()&&mae>=0.&&rmse>=0.);
  if kind!="month"{assert_eq!(n,if year==2021{7501}else{7639});}
  let s=sums.entry((id,kind,idx)).or_default();assert!(s.years.insert(year));s.n+=n;s.ae+=n as f64*mae;s.se+=n as f64*rmse.powi(2);
 }
 assert_eq!(ids.len(),33);assert_eq!(sums.len(),1221);
 let mut out=String::from("model,metric_kind,index,origins,mae_c,rmse_c\n");
 for id in ids{
  for (kind,first,last) in [("score",0,0),("lead",1,24),("month",1,12)]{
   let(mut n,mut ae,mut se)=(0,0.,0.);
   for i in first..=last{
    let s=&sums[&(id,kind,i)];assert_eq!(s.years.len(),2);
    out+=&format!("{id},{kind},{i},{},{:.8},{:.8}\n",s.n,s.ae/s.n as f64,(s.se/s.n as f64).sqrt());
    n+=s.n;ae+=s.ae;se+=s.se;
   }
   let total=&sums[&(id,"score",0)];
   assert_eq!(n,total.n*if kind=="lead"{24}else{1});
   assert!((ae/n as f64-total.ae/total.n as f64).abs()<1e-7);
   assert!(((se/n as f64).sqrt()-(total.se/total.n as f64).sqrt()).abs()<1e-7);
  }
 }
 fs::write(&a[2],out).unwrap();
}
