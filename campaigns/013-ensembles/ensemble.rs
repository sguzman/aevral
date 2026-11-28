//! C013 frozen combination engine. Prediction receives features, never targets.
use std::{fs,io::{BufRead,BufReader},path::Path};
pub const N:usize=8;
pub const T:&[usize]=&[0,1,2];
pub const U:&[usize]=&[0,1,2,3,4,5,6,7];
pub const Q:&[usize]=&[0,1,2,3];
pub const HEADER:&str="fold,origin,lead,target,daily,I,L,R,M,S,A,H,X";
#[derive(Clone,Copy,Debug)]
pub struct Features {pub origin:i64,pub lead:usize,pub month:usize,pub base:[f64;N]}
#[derive(Clone,Copy,Debug)]
pub struct Row {pub x:Features,pub target:f64,pub daily:f64}
#[derive(Debug)]
pub struct Fold {pub year:u16,pub rows:Vec<Row>}
fn finite(s:&str)->f64{let x:f64=s.parse().expect("invalid number");assert!(x.is_finite());x}
pub fn month(year:u16,origin:i64)->usize{
 let start=match year{2020=>10519200,2021=>11046240,2022=>11571840,_=>panic!("unsupported fold")};
 let days=(origin+60-start).div_euclid(1440);assert!(days>=0);
 let lengths=[31,if year==2020{29}else{28},31,30,31,30,31,31,30,31,30,31];
 let mut d=days;for (i,n) in lengths.into_iter().enumerate(){if d<n{return i+1;}d-=n;}panic!("origin outside year");
}
pub fn load(path:&Path,year:u16)->Fold{
 let n=match year{2020=>7543,2021=>7501,2022=>7639,_=>panic!("unsupported fold")};
 let mut lines=BufReader::new(fs::File::open(path).unwrap()).lines();
 assert_eq!(lines.next().unwrap().unwrap(),HEADER);
 let mut rows:Vec<Row>=Vec::with_capacity(n*24);
 for line in lines{
  let l=line.unwrap();let p:Vec<_>=l.split(',').collect();assert_eq!(p.len(),13);
  assert_eq!(p[0].parse::<u16>().unwrap(),year);
  let origin:i64=p[1].parse().unwrap();let lead:usize=p[2].parse().unwrap();
  assert_eq!(lead,rows.len()%24+1);
  let end=match year{2020=>11046240,2021=>11571840,_=>12097440};
  assert!(origin+1440<end);
  if let Some(last)=rows.last(){if lead==1{assert!(origin>last.x.origin);}else{assert_eq!(origin,last.x.origin);}}
  let base=std::array::from_fn(|i|finite(p[5+i]));
  rows.push(Row{x:Features{origin,lead,month:month(year,origin),base},target:finite(p[3]),daily:finite(p[4])});
 }
 assert_eq!(rows.len(),24*n);Fold{year,rows}
}
/// Return only the frozen prior-year calibration folds; reject any future or assessment row.
pub fn calibration<'a>(folds:&'a [Fold],assessment:u16)->Vec<&'a Row>{
 assert!(matches!(assessment,2021|2022));
 let expected:Vec<u16>=(2020..assessment).collect();
 assert_eq!(folds.iter().map(|f|f.year).collect::<Vec<_>>(),expected);
 assert!(folds.iter().all(|f|!f.rows.is_empty()));
 folds.iter().flat_map(|f|f.rows.iter()).collect()
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Static {Mean,Median}
pub fn static_recipe(id:u8)->Option<(Static,&'static [usize])>{
 use Static::*;
 Some(match id{1=>(Mean,T),2=>(Median,T),3=>(Mean,U),4=>(Median,U),5=>(Mean,&[0,1]),6=>(Mean,&[0,2]),7=>(Mean,&[1,2]),8=>(Mean,Q),9=>(Mean,&[5,6,7]),10=>(Mean,&[0,4]),_=>return None})
}
pub fn mean(x:&Features,cols:&[usize])->f64{assert!(!cols.is_empty());cols.iter().map(|&i|x.base[i]).sum::<f64>()/cols.len() as f64}
pub fn median(mut v:Vec<f64>)->f64{assert!(!v.is_empty());v.sort_by(f64::total_cmp);let k=v.len()/2;if v.len()%2==1{v[k]}else{(v[k-1]+v[k])/2.}}
pub fn static_predict(id:u8,x:&Features)->f64{
 let (method,cols)=static_recipe(id).expect("not static recipe");
 match method{Static::Mean=>mean(x,cols),Static::Median=>median(cols.iter().map(|&i|x.base[i]).collect())}
}
#[derive(Clone,Copy,Debug,Default)]
pub struct Metric{pub count:usize,pub abs:f64,pub squared:f64}
impl Metric{
 pub fn add(&mut self,y:f64,p:f64){assert!(y.is_finite()&&p.is_finite());let e=p-y;self.count+=1;self.abs+=e.abs();self.squared+=e*e;}
 pub fn mae(&self)->f64{assert!(self.count>0);self.abs/self.count as f64}
 pub fn rmse(&self)->f64{assert!(self.count>0);(self.squared/self.count as f64).sqrt()}
}
#[derive(Default)]
pub struct Scores{pub all:Metric,pub leads:[Metric;24],pub months:[Metric;12]}
impl Scores{
 pub fn add(&mut self,row:&Row,p:f64){self.all.add(row.target,p);self.leads[row.x.lead-1].add(row.target,p);self.months[row.x.month-1].add(row.target,p);}
 pub fn validate(&self,n:usize){assert_eq!(self.all.count,n*24);assert!(self.leads.iter().all(|m|m.count==n));assert_eq!(self.months.iter().map(|m|m.count).sum::<usize>(),n*24);assert!(self.months.iter().all(|m|m.count%24==0));}
}
#[cfg(test)]
mod tests{
 use super::*;
 fn features()->Features{Features{origin:10519140,lead:1,month:1,base:[1.,2.,9.,4.,5.,6.,7.,8.]}}
 fn row()->Row{Row{x:features(),target:3.,daily:2.}}
 #[test]fn all_static_recipes(){
  let x=features();let expected=[4.,2.,5.25,5.5,1.5,5.,5.5,4.,7.,3.];
  for (i,v) in expected.iter().enumerate(){assert!((static_predict(i as u8+1,&x)-v).abs()<1e-12);}
  assert!(static_recipe(11).is_none());
 }
 #[test]fn month_boundaries(){assert_eq!(month(2020,10519140),1);assert_eq!(month(2020,10519200+60*1440-60),3);assert_eq!(month(2021,11046240+59*1440-60),3);}
 #[test]fn metric_parity(){let mut m=Metric::default();m.add(1.,2.);m.add(1.,3.);assert_eq!(m.mae(),1.5);assert_eq!(m.rmse(),2.5f64.sqrt());}
 #[test]fn calibration_years(){let f=[Fold{year:2020,rows:vec![row()]},Fold{year:2021,rows:vec![row()]}];assert_eq!(calibration(&f,2022).len(),2);assert_eq!(calibration(&f[..1],2021).len(),1);}
 #[test]#[should_panic]fn future_calibration_rejected(){let f=[Fold{year:2020,rows:vec![row()]},Fold{year:2021,rows:vec![row()]}];calibration(&f,2021);}
 #[test]#[should_panic]fn wrong_calibration_order(){let f=[Fold{year:2021,rows:vec![row()]},Fold{year:2020,rows:vec![row()]}];calibration(&f,2022);}
}
