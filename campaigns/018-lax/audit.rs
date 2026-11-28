//! Independent reconstruction from exported predictions. No producer imports.
use std::{fs::File,io::{BufRead,BufReader},path::Path};
const N:usize=7406*24;
fn lines(p:&Path)->impl Iterator<Item=String>{BufReader::new(File::open(p).unwrap()).lines().map(Result::unwrap)}
fn num(x:&str)->f64{let v:f64=x.parse().unwrap();assert!(v.is_finite());v}
fn close(a:f64,b:f64){assert!((a-b).abs()<1e-8,"metric mismatch {a} {b}")}
fn rank(v:&[f64],p:f64)->f64{let q=p*v.len()as f64;let n=if(q-q.round()).abs()<1e-10{q.round()}else{q.ceil()};v[(n as usize).clamp(1,v.len())-1]}
#[derive(Default)]
struct Sum{n:usize,abs:f64,sq:f64,crps:f64,coverage:[usize;4],width:[f64;4]}
impl Sum{
 fn add(&mut self,y:f64,p:f64,s:Option<&[f64]>){
  self.n+=1;let e=y-p;self.abs+=e.abs();self.sq+=e*e;
  if let Some(v)=s{
   assert_eq!(v.len(),99);assert!(v.windows(2).all(|w|w[0]<=w[1]));assert_eq!(p,v[49]);
   let a=v.iter().map(|x|(x-y).abs()).sum::<f64>()/99.;
   let b=v.windows(2).enumerate().map(|(i,w)|((i+1)*(98-i))as f64*(w[1]-w[0])).sum::<f64>()/9801.;
   self.crps+=a-b;
   for(i,c)in [0.5,0.8,0.9,0.95].into_iter().enumerate(){let l=rank(v,(1.-c)/2.);let u=rank(v,(1.+c)/2.);self.coverage[i]+=usize::from(l<=y&&y<=u);self.width[i]+=u-l;}
  }
 }
 fn verify(&self,p:&[&str],prob:bool){
  assert_eq!(p.len(),14);assert_eq!(p[2].parse::<usize>().unwrap(),self.n);
  let n=self.n as f64;close(num(p[3]),self.abs/n);close(num(p[4]),(self.sq/n).sqrt());
  if prob{close(num(p[5]),self.crps/n);for i in 0..4{close(num(p[6+i*2]),self.coverage[i]as f64/n);close(num(p[7+i*2]),self.width[i]/n);}}
  else{assert!(p[5..].iter().all(|s|s.is_empty()));}
 }
}
fn main(){
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),2,"audit ROOT");let root=Path::new(&args[1]);
 let mut reference=Vec::new();
 for id in ["transfer","mlp","p13"]{
  let dir=root.join(id);let mut rows=lines(&dir.join("predictions.csv"));assert_eq!(rows.next().unwrap(),"origin,lead,target,daily,point,I,L,R,M");
  let mut dist=(id=="p13").then(||lines(&dir.join("distribution.csv")));
  if let Some(d)=&mut dist{let h=d.next().unwrap();assert_eq!(h,format!("origin,lead,{}",(0..99).map(|i|format!("s{i}")).collect::<Vec<_>>().join(",")));}
  let mut sum:[Sum;25]=std::array::from_fn(|_|Sum::default());let mut daily:[Sum;25]=std::array::from_fn(|_|Sum::default());let mut prev=0;let mut count=0;
  for (i,line) in rows.enumerate(){
   let p:Vec<_>=line.split(',').collect();assert_eq!(p.len(),9);let t:i64=p[0].parse().unwrap();let h:usize=p[1].parse().unwrap();
   assert_eq!(h,i%24+1);assert!(t%60==0);assert!(if h==1{t>prev}else{t==prev});prev=t;
   assert!(t+60>=13150080&&t+1440<13675680,"outside 2025 assessment");
   let y=num(p[2]);let d=num(p[3]);let m=num(p[4]);
   if id=="transfer"{reference.push((t,h,y,d));}else{assert_eq!(reference[i],(t,h,y,d));}
   let support=dist.as_mut().map(|it|{let l=it.next().unwrap();let q:Vec<_>=l.split(',').collect();assert_eq!(q.len(),101);assert_eq!(q[..2],p[..2]);for b in &p[5..]{num(b);}q[2..].iter().map(|x|num(x)).collect::<Vec<_>>()});
   if id!="p13"{assert!(p[5..].iter().all(|x|x.is_empty()));}
   sum[0].add(y,m,support.as_deref());sum[h].add(y,m,support.as_deref());daily[0].add(y,d,None);daily[h].add(y,d,None);count+=1;
  }
  assert_eq!(count,N);if let Some(d)=&mut dist{assert!(d.next().is_none());}
  let mut metrics=lines(&dir.join("metrics.csv"));assert_eq!(metrics.next().unwrap(),"model,lead,n,mae,rmse,crps,coverage50,width50,coverage80,width80,coverage90,width90,coverage95,width95");
  for h in 0..25{for (name,s,prob) in [(id,&sum[h],id=="p13"),("daily",&daily[h],false)]{let l=metrics.next().unwrap();let p:Vec<_>=l.split(',').collect();assert_eq!(p[0],name);assert_eq!(p[1].parse::<usize>().unwrap(),h);s.verify(&p,prob);}}
  assert!(metrics.next().is_none());println!("independent-audit,{id},{count},pass");
 }
}
