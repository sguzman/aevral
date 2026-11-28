//! Independently reproduce the durable C014 pooled diagnostics.
use std::{collections::BTreeMap,fs};
const HEADER:&str="recipe,group,index,scalar_count,mae_c,rmse_c,crps_c,pin025,pin05,pin10,pin25,pin50,pin75,pin90,pin95,pin975,cov50,width50,is50,cov80,width80,is80,cov90,width90,is90,cov95,width95,is95";
fn read(path:&str,year:u16)->Vec<Vec<String>>{
 let s=fs::read_to_string(path).unwrap();let mut rows=Vec::new();
 for line in s.lines(){let p:Vec<_>=line.split(',').map(str::to_owned).collect();assert_eq!(p.len(),29);assert_eq!(p[0].parse::<u16>().unwrap(),year);rows.push(p);}
 assert_eq!(rows.len(),888);rows
}
fn num(s:&str)->f64{let v:f64=s.parse().unwrap();assert!(v.is_finite()&&v>=0.);v}
fn main(){
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),4);
 let a=read(&args[1],2021);let b=read(&args[2],2022);
 let mut map=BTreeMap::new();for r in &b{assert!(map.insert(r[1..4].join(","),r).is_none());}
 let mut out=String::from(HEADER);out.push('\n');let mut totals=BTreeMap::new();
 for r in &a{let key=r[1..4].join(",");let s=map.remove(&key).unwrap();let n1=r[4].parse::<usize>().unwrap();let n2=s[4].parse::<usize>().unwrap();assert!(n1>0&&n2>0);
 let n=n1+n2;let id=r[1].strip_prefix('P').unwrap().parse::<u8>().unwrap();assert!((1..=24).contains(&id));let group=r[2].as_str();let idx=r[3].parse::<usize>().unwrap();assert!(match group{"all"=>idx==0,"lead"=>(1..=24).contains(&idx),"month"=>(1..=12).contains(&idx),_=>false});
 let t=totals.entry((id,group)).or_insert((0usize,0usize));t.0+=n;t.1+=1;
 out.push_str(&format!("{key},{n}"));
 for k in 5..29{
  if id>=21&&(7..17).contains(&k){assert_eq!(r[k],"NA");assert_eq!(s[k],"NA");out.push_str(",NA");continue;}
  let x=num(&r[k]);let y=num(&s[k]);if [17,20,23,26].contains(&k){assert!(x<=1.&&y<=1.);}
  let value=if k==6{((n1 as f64*x*x+n2 as f64*y*y)/n as f64).sqrt()}else{(n1 as f64*x+n2 as f64*y)/n as f64};
  out.push_str(&format!(",{value:.8}"));
 }out.push('\n');}
 assert!(map.is_empty());
 for id in 1..=24{assert_eq!(totals[&(id,"all")],(363360,1));assert_eq!(totals[&(id,"lead")],(363360,24));assert_eq!(totals[&(id,"month")],(363360,12));}
 fs::write(&args[3],out).unwrap();println!("c014-pool,888,pass");
}
