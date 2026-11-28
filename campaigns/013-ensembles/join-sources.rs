//! C013 strict source audit: join verified long-form predictions into causal fold tables.
use std::{fs,io::{BufRead,BufReader,BufWriter,Write},path::Path,collections::{BTreeMap,BTreeSet}};
const HEADER:&str="fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c";
const ALIASES:&str="ILRMSAHX";
#[derive(Debug,Clone)]
struct Row {origin:i64,lead:u8,target:f64,pred:f64}
fn num(s:&str)->f64{let n:f64=s.parse().unwrap();assert!(n.is_finite());n}
fn read(path:&Path,year:&str,allowed:&[&str])->BTreeMap<String,Vec<Row>>{
 let mut lines=BufReader::new(fs::File::open(path).unwrap()).lines();
 assert_eq!(lines.next().unwrap().unwrap(),HEADER);
 let mut groups:BTreeMap<String,Vec<Row>>=BTreeMap::new();
 for line in lines {
  let line=line.unwrap();let p:Vec<_>=line.split(',').collect();assert_eq!(p.len(),6);
  assert_eq!(p[0],year);assert!(allowed.contains(&p[1]),"unexpected model {}",p[1]);
  let row=Row{origin:p[2].parse().unwrap(),lead:p[3].parse().unwrap(),target:num(p[4]),pred:num(p[5])};
  let g=groups.entry(p[1].to_owned()).or_default();
  assert_eq!(row.lead,(g.len()%24+1) as u8);
  if let Some(prev)=g.last(){if row.lead==1{assert!(row.origin>prev.origin);}else{assert_eq!(row.origin,prev.origin);}}
  g.push(row);
 }
 assert_eq!(groups.len(),allowed.len());groups
}
fn align(a:&[Row],b:&[Row]){
 assert_eq!(a.len(),b.len());
 for (x,y) in a.iter().zip(b){assert_eq!((x.origin,x.lead),(y.origin,y.lead));assert!((x.target-y.target).abs()<=1e-8,"target mismatch");}
}
fn metrics(rows:&[Row])->(f64,f64){
 let (mut a,mut s)=(0.,0.);for r in rows{let e=r.pred-r.target;a+=e.abs();s+=e*e;}
 (a/rows.len() as f64,(s/rows.len() as f64).sqrt())
}
fn audit(rows:&[Row],n:usize,mae:f64,rmse:f64){
 assert_eq!(rows.len(),24*n);let (a,r)=metrics(rows);
 assert!((a-mae).abs()<1e-7&&(r-rmse).abs()<1e-7,"score mismatch {a} {r} expected {mae} {rmse}");
}
fn main(){
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),4,"usage: join-sources SOURCE_DIR INVENTORY OUTPUT_DIR");
 let root=Path::new(&args[1]);let out=Path::new(&args[3]);fs::create_dir_all(out).unwrap();
 let inv=fs::read_to_string(&args[2]).unwrap();let rows:Vec<Vec<&str>>=inv.lines().skip(1).map(|l|l.split(',').collect()).collect();
 assert_eq!(rows.len(),24);assert!(rows.iter().all(|r|r.len()==13));
 for (year,n,dm,dr,start,end) in [("2020",7543,2.07087759,2.70871411,10519200i64,11046240i64),("2021",7501,2.14898785,2.81802465,11046240,11571840),("2022",7639,2.10544725,2.81708680,11571840,12097440)] {
  let mut base:Vec<Vec<Row>>=Vec::new();let mut daily=Vec::new();
  for alias in ALIASES.chars(){
   let v:Vec<_>=rows.iter().filter(|r|r[0]==year&&r[1]==alias.to_string()).collect();assert_eq!(v.len(),1);let r=v[0];assert_eq!(r[8].parse::<usize>().unwrap(),n);
   let path=root.join(year).join(alias.to_string()).join("predictions.csv");
   let mut groups=if alias=='I'{read(&path,year,&["mlp64","daily","persistence","climatology"])}else{read(&path,year,&[r[2]])};
   if alias=='I'{daily=groups.remove("daily").unwrap();audit(&daily,n,dm,dr);}
   let p=groups.remove(r[2]).unwrap();audit(&p,n,num(r[9]),num(r[10]));
   assert!(p.iter().all(|x|x.origin+60>=start&&x.origin+1440<end));
   if alias=='I'{align(&p,&daily);for other in groups.values(){align(&p,other);}}else{align(&base[0],&p);}
   let (a,e)=metrics(&p);println!("base-audit,{year},{alias},{n},{a:.8},{e:.8},pass");
   base.push(p);
  }
  let mut csv=BufWriter::new(fs::File::create(out.join(format!("{year}.csv"))).unwrap());
  let mut origins=BufWriter::new(fs::File::create(out.join(format!("{year}-origins.txt"))).unwrap());
  writeln!(csv,"fold,origin,lead,target,daily,I,L,R,M,S,A,H,X").unwrap();
  for i in 0..24*n {let r=&base[0][i];if r.lead==1{writeln!(origins,"{}",r.origin).unwrap();}
   write!(csv,"{year},{},{},{},{}",r.origin,r.lead,r.target,daily[i].pred).unwrap();
   for b in &base{write!(csv,",{}",b[i].pred).unwrap();}writeln!(csv).unwrap();
  }
  println!("joined,{year},{n},{},8,pass",24*n);
 }
 assert_eq!(rows.iter().map(|r|(r[0],r[1])).collect::<BTreeSet<_>>().len(),24);
}
#[cfg(test)]
mod tests{
 use super::*;
 fn row(o:i64,l:u8,t:f64,p:f64)->Row{Row{origin:o,lead:l,target:t,pred:p}}
 #[test]fn score(){let r=[row(0,1,1.,2.),row(0,2,1.,3.)];assert_eq!(metrics(&r),(1.5,2.5f64.sqrt()));}
 #[test]#[should_panic]fn target_mismatch(){align(&[row(0,1,0.,0.)],&[row(0,1,1.,0.)]);}
 #[test]#[should_panic]fn key_mismatch(){align(&[row(0,1,0.,0.)],&[row(60,1,0.,0.)]);}
}
