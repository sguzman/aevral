//! Independent C014 export/metric audit, no imports from fitting or scoring engine.
mod reconstruct;
use reconstruct::{Row,Forecast,Reconstruct};
use std::{collections::BTreeMap,fs,io::{Read,BufReader},path::Path,process::{Command,Stdio}};
const Q:[f64;9]=[0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975];
const C:[f64;4]=[0.50,0.80,0.90,0.95];
fn eq(a:f64,b:f64){assert!(a.is_finite()&&b.is_finite()&&(a-b).abs()<=1e-7,"mismatch {a} vs {b}");}
fn bytes<const N:usize>(r:&mut impl Read)->[u8;N]{let mut v=[0;N];r.read_exact(&mut v).unwrap();v}
fn f(r:&mut impl Read)->f64{let x=f64::from_le_bytes(bytes(r));assert!(x.is_finite());x}
fn quantile(v:&[f64],p:f64)->f64{let z=p*v.len()as f64;let n=if(z-z.round()).abs()<1e-10{z.round()}else{z.ceil()};v[(n as usize).clamp(1,v.len())-1]}
fn pin(y:f64,q:f64,p:f64)->f64{p*(y-q).max(0.)+(1.-p)*(q-y).max(0.)}
fn terms(v:&[f64],y:f64)->(f64,f64){
 let n=v.len()as f64;let abs=v.iter().map(|x|(x-y).abs()).sum::<f64>()/n;
 // Half pairwise distance via adjacent gaps, independent of weighted-order formula.
 let pair=v.windows(2).enumerate().map(|(i,w)|{let k=(i+1)as f64;(w[1]-w[0])*k*(n-k)}).sum::<f64>()/(n*n);(abs,pair)
}
fn month(year:u16,origin:i64)->usize{
 let start=if year==2021{11046240}else{11571840};
 let d=(origin+60-start).div_euclid(1440);assert!((0..365).contains(&d));
 [31,59,90,120,151,181,212,243,273,304,334,365].iter().position(|&x|d<x).unwrap()
}
#[derive(Clone,Default)]
struct Sum{n:usize,ae:f64,se:f64,crps:f64,pin:[f64;9],cov:[f64;4],width:[f64;4],interval:[f64;4]}
impl Sum{
 fn add(&mut self,y:f64,median:f64,crps:f64,pin:[f64;9],bands:[(f64,f64);4]){
  self.n+=1;self.ae+=(median-y).abs();self.se+=(median-y).powi(2);self.crps+=crps;
  for j in 0..9{self.pin[j]+=pin[j];}
  for j in 0..4{let(l,u)=bands[j];assert!(l<=u);self.cov[j]+=if y>=l&&y<=u{1.}else{0.};self.width[j]+=u-l;self.interval[j]+=u-l+2./(1.-C[j])*((l-y).max(0.)+(y-u).max(0.));}
 }
 fn compare(&self,p:&[String],dist:bool){
  assert_eq!(p.len(),29);assert_eq!(p[4].parse::<usize>().unwrap(),self.n);let n=self.n as f64;let val=|i:usize|reconstruct::number(&p[i]);
  eq(val(5),self.ae/n);eq(val(6),(self.se/n).sqrt());
  if dist{eq(val(7),self.crps/n);for j in 0..9{eq(val(8+j),self.pin[j]/n);}}else{assert!(p[7..17].iter().all(|x|x=="NA"));}
  for j in 0..4{eq(val(17+3*j),self.cov[j]/n);eq(val(18+3*j),self.width[j]/n);eq(val(19+3*j),self.interval[j]/n);}
 }
}
fn audit_record(r:&mut impl Read,row:&Row,m:&Reconstruct,id:u8)->(f64,f64,[f64;9],[(f64,f64);4],bool){
 assert_eq!(i64::from_le_bytes(bytes(r)),row.origin);let flag=bytes::<2>(r);assert_eq!(flag[0]as usize,row.lead);assert!(flag[1]<=1);
 assert_eq!(f(r),row.target);let median=f(r);
 let(mut crps,mut pinballs)=(0.,[0.;9]);
 let(expected_median,expected_bands,cross)=match m.forecast(row){
  Forecast::Distribution(v,cross)=>{
   assert!(id<=20);assert_eq!(v.len(),match id{17=>4,18=>8,19|20=>396,_=>99});assert!(v.windows(2).all(|w|w[0]<=w[1]));
   let qs=Q.map(|p|quantile(&v,p));for q in qs{eq(f(r),q);}
   let(a,b)=terms(&v,row.target);crps=(a-b).max(0.);eq(f(r),crps);eq(f(r),a);eq(f(r),b);
   for j in 0..9{pinballs[j]=pin(row.target,qs[j],Q[j]);eq(f(r),pinballs[j]);}
   (qs[4],C.map(|c|(quantile(&v,(1.-c)/2.),quantile(&v,(1.+c)/2.))),cross)
  }
  Forecast::Intervals(m,bands)=>{assert!(id>=21);(m,bands,false)}
 };
 eq(median,expected_median);assert_eq!(flag[1],u8::from(cross));
 let bands=std::array::from_fn(|j|{let l=f(r);let u=f(r);eq(l,expected_bands[j].0);eq(u,expected_bands[j].1);(l,u)});
 assert!(bands.windows(2).all(|w|w[1].0<=w[0].0+1e-10&&w[1].1>=w[0].1-1e-10));
 (median,crps,pinballs,bands,cross)
}
fn main(){
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),4,"audit JOINED_DIR OUTPUT_DIR YEAR");let root=Path::new(&args[1]);let out=Path::new(&args[2]);let year:u16=args[3].parse().unwrap();assert!(matches!(year,2021|2022));
 let cal:Vec<_>=(2020..year).flat_map(|y|reconstruct::read_rows(&root.join(format!("{y}.csv")),y)).collect();
 let data=reconstruct::read_rows(&root.join(format!("{year}.csv")),year);
 let mut table=BTreeMap::new();
 for l in fs::read_to_string(out.join("metrics.csv")).unwrap().lines(){
  let p:Vec<String>=l.split(',').map(str::to_owned).collect();assert_eq!(p.len(),29);assert_eq!(p[0].parse::<u16>().unwrap(),year);
  let id=p[1].strip_prefix('P').unwrap().parse::<u8>().unwrap();assert!((1..=24).contains(&id));let i=p[3].parse::<usize>().unwrap();
  assert!(match p[2].as_str(){"all"=>i==0,"lead"=>(1..=24).contains(&i),"month"=>(1..=12).contains(&i),_=>false});
  assert!(table.insert((id,p[2].clone(),i),p).is_none());
 }
 assert_eq!(table.len(),888);
 let costs=fs::read_to_string(out.join("costs.csv")).unwrap();let mut l=costs.lines();assert_eq!(l.next().unwrap(),"fold,recipe,fit_s,issue_score_export_s,crossing_rows");
 for id in 1..=24u8{
  let m=Reconstruct::load(&out.join(format!("P{id:02}-parameters.csv")),year,id,&cal);
  let mut child=Command::new("gzip").arg("-dc").arg(out.join(format!("P{id:02}-forecasts.bin.gz"))).stdout(Stdio::piped()).spawn().unwrap();
  let mut r=BufReader::new(child.stdout.take().unwrap());assert_eq!(&bytes::<8>(&mut r),b"C014BIN1");assert_eq!(u16::from_le_bytes(bytes(&mut r)),year);assert_eq!(bytes::<1>(&mut r),[id]);
  let mut total=Sum::default();let mut leads:[Sum;24]=std::array::from_fn(|_|Sum::default());let mut months:[Sum;12]=std::array::from_fn(|_|Sum::default());let mut crosses=0;
  for row in &data{let(median,crps,pin,bands,cross)=audit_record(&mut r,row,&m,id);crosses+=usize::from(cross);
   for s in [&mut total,&mut leads[row.lead-1],&mut months[month(year,row.origin)]]{s.add(row.target,median,crps,pin,bands);}
  }
  let mut extra=[0];assert_eq!(r.read(&mut extra).unwrap(),0);drop(r);assert!(child.wait().unwrap().success());
  assert_eq!(total.n,data.len());total.compare(&table.remove(&(id,"all".into(),0)).unwrap(),id<=20);
  for(i,s)in leads.iter().enumerate(){s.compare(&table.remove(&(id,"lead".into(),i+1)).unwrap(),id<=20);}
  for(i,s)in months.iter().enumerate(){s.compare(&table.remove(&(id,"month".into(),i+1)).unwrap(),id<=20);}
  let c:Vec<_>=l.next().unwrap().split(',').collect();assert_eq!(c.len(),5);assert_eq!(c[0].parse::<u16>().unwrap(),year);assert_eq!(c[1],format!("P{id:02}"));assert!(reconstruct::number(c[2])>=0.&&reconstruct::number(c[3])>=0.);assert_eq!(c[4].parse::<usize>().unwrap(),crosses);
  println!("audit,{year},P{id:02},{},37,{crosses},pass",total.n);
 }
 assert!(table.is_empty()&&l.next().is_none());println!("c014-audit,{year},24,888,pass");
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn independent_crps(){let(a,b)=terms(&[0.,2.],1.);eq(a-b,0.5);let(a,b)=terms(&[3.],1.);eq(a-b,2.);}
 #[test]fn independent_quantiles(){assert_eq!(quantile(&[1.,2.,3.,4.],0.5),2.);eq(pin(0.,1.,0.1),0.9);}
 #[test]fn month_boundaries(){assert_eq!(month(2021,11046240),0);assert_eq!(month(2021,11046240+31*1440-60),1);assert_eq!(month(2022,11571840+364*1440),11);}
}
