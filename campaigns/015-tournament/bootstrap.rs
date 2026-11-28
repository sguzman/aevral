//! Paired calendar-block bootstrap on audited, complete seed42 matched exports.
//! No producer imports and no tuning. All candidates use identical block draws.
use std::{fs::File,io::{BufRead,BufReader},path::Path};
const COHORTS:&str=include_str!("../../results/C015/cohorts.csv");
const HEADER:&str="fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c";
const IDS:[(&str,&str);15]=[("L","O04-04"),("R","O05-02"),("M","O01-06"),("S","O02-06"),("A","O06-06"),("H","O08-06"),("X","O07-06"),("G","xgboost-128"),("K","fitc-s1-m256"),("N","mae-nhits64"),("W","pressure-tendency-short"),("E08","E08"),("E13","E13"),("E14","E14"),("P13","P13")];
const B:usize=10000;
fn cohort(year:u16)->(i64,i64,usize){let l=COHORTS.lines().skip(1).find(|l|l.split(',').next()==Some(&year.to_string())).unwrap();let p:Vec<_>=l.split(',').collect();(p[1].parse().unwrap(),p[2].parse().unwrap(),p[4].parse().unwrap())}
fn num(s:&str)->f64{let x=s.parse::<f64>().unwrap();assert!(x.is_finite());x}
#[derive(Clone)]
struct Origin{t:i64,y:[f64;24],loss:f64}
fn load(path:&Path,year:u16,recipe:&str)->Vec<Origin>{
 let(start,end,n)=cohort(year);let mut l=BufReader::new(File::open(path).unwrap()).lines();assert_eq!(l.next().unwrap().unwrap(),HEADER);
 let mut out:Vec<Origin>=Vec::with_capacity(n);
 for(i,l)in l.enumerate(){let l=l.unwrap();let p:Vec<_>=l.split(',').collect();assert_eq!(p.len(),6);assert_eq!(p[0].parse::<u16>().unwrap(),year);assert_eq!(p[1],recipe);let t=p[2].parse::<i64>().unwrap();let h=i%24;assert_eq!(p[3].parse::<usize>().unwrap(),h+1);let y=num(p[4]);let v=num(p[5]);assert!(t%60==0&&t+60>=start&&t+1440<end);
  if h==0{assert!(out.last().is_none_or(|r|r.t<t));out.push(Origin{t,y:[0.;24],loss:0.});}
  let r=out.last_mut().unwrap();assert_eq!(r.t,t);r.y[h]=y;r.loss+=(v-y).abs()/24.;
 }
 assert_eq!(out.len(),n);assert!(out.iter().all(|r|r.loss.is_finite()));out
}
#[derive(Clone,Copy,Default,Debug)]
struct Block{n:usize,d:f64}
fn blocks(a:&[Origin],b:&[Origin],start:i64,days:i64)->Vec<Block>{
 assert_eq!(a.len(),b.len());let mut out:Vec<Block>=Vec::new();let mut prev=None;
 for (a,b) in a.iter().zip(b){assert_eq!(a.t,b.t);assert_eq!(a.y,b.y);
  let k=(a.t+60-start).div_euclid(days*1440);assert!(k>=0);
  if prev!=Some(k){assert!(prev.is_none_or(|v|v<k));out.push(Block::default());prev=Some(k);}
  let r=out.last_mut().unwrap();r.n+=1;r.d+=b.loss-a.loss;
 }out
}
struct Rng(u64);
impl Rng{
 fn next(&mut self)->u64{self.0=self.0.wrapping_add(0x9e3779b97f4a7c15);let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);z^(z>>31)}
 fn bounded(&mut self,n:usize)->usize{assert!(n>0);let n=n as u64;let threshold=n.wrapping_neg()%n;loop{let x=self.next();if x>=threshold{return(x%n)as usize}}}
}
fn quantile(v:&[f64],p:f64)->f64{let x=p*(v.len()-1)as f64;let i=x.floor()as usize;v[i]+(v[x.ceil()as usize]-v[i])*(x-i as f64)}
fn bootstrap(folds:&[Vec<Block>],seed:u64,reps:usize)->(f64,f64,f64){
 let n:usize=folds.iter().flatten().map(|b|b.n).sum();assert!(n>0);
 let point=folds.iter().flatten().map(|b|b.d).sum::<f64>()/n as f64;
 let mut rng=Rng(seed);let mut samples=Vec::with_capacity(reps);
 for _ in 0..reps{let(mut n,mut d)=(0,0.);for f in folds{assert!(!f.is_empty());for _ in 0..f.len(){let b=f[rng.bounded(f.len())];n+=b.n;d+=b.d;}}samples.push(d/n as f64);}
 samples.sort_by(f64::total_cmp);(point,quantile(&samples,0.025),quantile(&samples,0.975))
}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),2,"bootstrap VERIFIED_PLAIN_EXPORT_DIR");let root=Path::new(&a[1]);
 let years=[2023,2024,2025,2026];
 println!("id,block_days,seed,replicates,blocks_2023,blocks_2024,blocks_2025,blocks_2026,origins,delta_mae,lower95,upper95,status");
 if years.iter().any(|y|!root.join(format!("{y}-I.csv")).is_file()){for(id,_)in IDS{println!("{id},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,incumbent-incomplete");}return}
 let incumbent:Vec<_>=years.iter().map(|&y|load(&root.join(format!("{y}-I.csv")),y,"mlp64")).collect();
 for(id,recipe)in IDS{
  if years.iter().any(|y|!root.join(format!("{y}-{id}.csv")).is_file()){println!("{id},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,candidate-incomplete");continue}
  let candidate:Vec<_>=years.iter().map(|&y|load(&root.join(format!("{y}-{id}.csv")),y,recipe)).collect();
  for(days,seed)in [(7,150042),(1,150001),(14,150014)]{
   let folds:Vec<_>=years.iter().enumerate().map(|(i,&y)|blocks(&incumbent[i],&candidate[i],cohort(y).0,days)).collect();
   let(p,l,u)=bootstrap(&folds,seed,B);
   let n:usize=folds.iter().flatten().map(|b|b.n).sum();
   println!("{id},{days},{seed},{B},{},{},{},{},{n},{p:.12},{l:.12},{u:.12},complete",folds[0].len(),folds[1].len(),folds[2].len(),folds[3].len());
  }
 }
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn rng_reference(){assert_eq!(Rng(0).next(),0xe220a8397b1dcdaf);let mut r=Rng(42);for _ in 0..10000{assert!(r.bounded(7)<7);}}
 #[test]fn constant_difference(){let f=vec![vec![Block{n:1,d:2.},Block{n:5,d:10.}],vec![Block{n:2,d:4.}]];assert_eq!(bootstrap(&f,42,1000),(2.,2.,2.));}
 #[test]fn weighting(){let f=vec![vec![Block{n:1,d:2.},Block{n:3,d:0.}]];assert_eq!(bootstrap(&f,42,1000).0,0.5);}
 #[test]fn calendar_blocks(){let a=vec![Origin{t:0,y:[0.;24],loss:1.},Origin{t:14340,y:[0.;24],loss:2.}];let mut b=a.clone();b[0].loss=3.;let v=blocks(&a,&b,0,7);assert_eq!(v.len(),2);assert_eq!(v[0].d,2.);}
 #[test]fn interpolation(){assert_eq!(quantile(&[0.,10.],0.25),2.5);}
}
