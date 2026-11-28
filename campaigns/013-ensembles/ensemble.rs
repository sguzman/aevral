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

/// Affine model in original Celsius coordinates, optionally anchored on a base column.
#[derive(Clone,Debug)]
pub struct Linear { pub weights:[f64;N],pub intercept:f64,pub iterations:usize,pub converged:bool,pub objective:f64 }
impl Linear {
 pub fn predict(&self,x:&Features)->f64{self.intercept+self.weights.iter().zip(x.base).map(|(w,p)|w*p).sum::<f64>()}
}
pub fn simplex(v:&[f64])->Vec<f64>{
 assert!(!v.is_empty()&&v.iter().all(|x|x.is_finite()));
 let mut u:Vec<(usize,f64)>=v.iter().copied().enumerate().collect();
 u.sort_by(|a,b|b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
 let(mut sum,mut theta)=(0.,0.);
 for (k,&(_,x)) in u.iter().enumerate(){sum+=x;let t=(sum-1.)/(k+1)as f64;if x>t{theta=t;}}
 let w:Vec<_>=v.iter().map(|x|(x-theta).max(0.)).collect();
 assert!((w.iter().sum::<f64>()-1.).abs()<1e-9);w
}
pub fn convex(rows:&[&Row],cols:&[usize])->Linear{
 assert!(!rows.is_empty()&&!cols.is_empty());
 let k=cols.len();let anchor=cols[0];let(mut gram,mut cross,mut yy)=([[0.;N];N],[0.;N],0.);
 for r in rows{
  let y=r.target-r.x.base[anchor];let d:[f64;N]=std::array::from_fn(|i|if i<k{r.x.base[cols[i]]-r.x.base[anchor]}else{0.});
  yy+=y*y;
  for i in 0..k{cross[i]+=d[i]*y;for j in 0..k{gram[i][j]+=d[i]*d[j];}}
 }
 let n=rows.len()as f64;yy/=n;for i in 0..k{cross[i]/=n;for j in 0..k{gram[i][j]/=n;}}
 let step=1./(2.*(0..k).map(|i|gram[i][i]).sum::<f64>()).max(1e-12);
 let mut w=vec![1./k as f64;k];let(mut iterations,mut converged)=(0,false);
 for it in 1..=2000{
  let v:Vec<_>=(0..k).map(|i|w[i]-step*2.*((0..k).map(|j|gram[i][j]*w[j]).sum::<f64>()-cross[i])).collect();
  let next=simplex(&v);let delta=next.iter().zip(&w).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);w=next;iterations=it;
  if delta<=1e-10{converged=true;break;}
 }
 let mut weights=[0.;N];for(i,&c)in cols.iter().enumerate(){weights[c]=w[i];}
 let objective=yy-2.*(0..k).map(|i|cross[i]*w[i]).sum::<f64>()+(0..k).map(|i|(0..k).map(|j|w[i]*gram[i][j]*w[j]).sum::<f64>()).sum::<f64>();
 assert!(objective.is_finite()&&objective>=-1e-9);
 Linear{weights,intercept:0.,iterations,converged,objective:objective.max(0.)}
}
fn cholesky(a:[[f64;N];N],b:[f64;N],k:usize)->[f64;N]{
 let mut l=[[0.;N];N];
 for i in 0..k{for j in 0..=i{let s=a[i][j]-(0..j).map(|p|l[i][p]*l[j][p]).sum::<f64>();if i==j{assert!(s>0.&&s.is_finite());l[i][j]=s.sqrt();}else{l[i][j]=s/l[j][j];}}}
 let(mut y,mut x)=([0.;N],[0.;N]);
 for i in 0..k{y[i]=(b[i]-(0..i).map(|j|l[i][j]*y[j]).sum::<f64>())/l[i][i];}
 for i in (0..k).rev(){x[i]=(y[i]-(i+1..k).map(|j|l[j][i]*x[j]).sum::<f64>())/l[i][i];}
 x
}
pub fn ridge(rows:&[&Row],cols:&[usize])->Linear{
 assert!(!rows.is_empty()&&cols.len()>=2&&cols[0]==0);
 let k=cols.len()-1;let n=rows.len()as f64;let(mut mx,mut my)=([0.;N],0.);
 for r in rows{my+=r.target-r.x.base[0];for i in 0..k{mx[i]+=r.x.base[cols[i+1]]-r.x.base[0];}}
 my/=n;for m in &mut mx[..k]{*m/=n;}
 let(mut a,mut b)=([[0.;N];N],[0.;N]);
 for r in rows{
  let y=r.target-r.x.base[0]-my;
  let d:[f64;N]=std::array::from_fn(|i|if i<k{r.x.base[cols[i+1]]-r.x.base[0]-mx[i]}else{0.});
  for i in 0..k{b[i]+=d[i]*y;for j in 0..k{a[i][j]+=d[i]*d[j];}}
 }
 for i in 0..k{b[i]/=n;for j in 0..k{a[i][j]/=n;}a[i][i]+=0.1;}
 let beta=cholesky(a,b,k);let intercept=my-(0..k).map(|i|beta[i]*mx[i]).sum::<f64>();
 let mut weights=[0.;N];weights[0]=1.-beta[..k].iter().sum::<f64>();for i in 0..k{weights[cols[i+1]]=beta[i];}
 let mut model=Linear{weights,intercept,iterations:1,converged:true,objective:0.};
 model.objective=rows.iter().map(|r|(r.target-model.predict(&r.x)).powi(2)).sum::<f64>()/n+0.1*beta[..k].iter().map(|b|b*b).sum::<f64>();
 assert!(model.objective.is_finite()&&model.weights.iter().all(|w|w.is_finite())&&model.intercept.is_finite());model
}
#[derive(Clone,Debug)]
pub enum Learned { Global(Linear),Lead(Vec<Linear>) }
impl Learned{pub fn predict(&self,x:&Features)->f64{match self{Self::Global(m)=>m.predict(x),Self::Lead(m)=>m[x.lead-1].predict(x)}}}
pub fn fit_linear(id:u8,rows:&[&Row])->Learned{
 let (solver,cols,by_lead):(fn(&[&Row],&[usize])->Linear,&[usize],bool)=match id{
  11=>(convex,T,false),12=>(convex,T,true),13=>(convex,U,false),14=>(convex,U,true),
  15=>(ridge,T,false),16=>(ridge,T,true),17=>(ridge,U,false),18=>(ridge,U,true),_=>panic!("not linear recipe")};
 if by_lead{Learned::Lead((1..=24).map(|lead|{let group:Vec<_>=rows.iter().copied().filter(|r|r.x.lead==lead).collect();solver(&group,cols)}).collect())}else{Learned::Global(solver(rows,cols))}
}
#[cfg(test)]
mod linear_tests{
 use super::*;
 fn row(a:f64,b:f64,y:f64,lead:usize)->Row{Row{x:Features{origin:10519140,lead,month:1,base:[a,b,a,a,a,a,a,a]},target:y,daily:0.}}
 #[test]fn projection(){assert_eq!(simplex(&[2.,0.,-1.]),vec![1.,0.,0.]);let w=simplex(&[0.2,0.2]);assert!((w[0]-0.5).abs()<1e-12&&w[0]==w[1]);}
 #[test]fn convex_known_mix(){let r=[row(0.,2.,1.,1),row(0.,4.,2.,1)];let refs:Vec<_>=r.iter().collect();let m=convex(&refs,&[0,1]);assert!((m.weights[0]-0.5).abs()<1e-8);assert!(m.objective<1e-12&&m.converged);}
 #[test]fn convex_identical(){let r=[row(2.,2.,3.,1)];let m=convex(&[&r[0]],T);assert!(m.converged);assert!((m.predict(&r[0].x)-2.).abs()<1e-12);}
 #[test]fn ridge_analytic(){let r=[row(0.,-1.,1.,1),row(0.,1.,3.,1)];let refs:Vec<_>=r.iter().collect();let m=ridge(&refs,&[0,1]);assert!((m.intercept-2.).abs()<1e-12);assert!((m.weights[1]-1./1.1).abs()<1e-12);assert!((m.weights.iter().sum::<f64>()-1.).abs()<1e-12);}
 #[test]fn ridge_collinear(){let r=[row(4.,4.,6.,1),row(2.,2.,4.,1)];let refs:Vec<_>=r.iter().collect();let m=ridge(&refs,U);assert!((m.intercept-2.).abs()<1e-12);assert!(m.objective<1e-12);}
 #[test]fn all_linear_recipes(){let rows:Vec<_>=(1..=24).flat_map(|h|[row(0.,2.,1.,h),row(0.,4.,2.,h)]).collect();let refs:Vec<_>=rows.iter().collect();for id in 11..=18{let m=fit_linear(id,&refs);for r in &rows{assert!(m.predict(&r.x).is_finite());}assert!(matches!(m,Learned::Lead(_))==(id%2==0));}}
}
