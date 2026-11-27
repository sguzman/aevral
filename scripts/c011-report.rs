use std::{collections::BTreeMap,fs};
fn field<'a>(s:&'a str,p:&str)->&'a str {
 let mut matches=s.lines().filter(|l|l.starts_with(p));
 let l=matches.next().unwrap_or_else(||panic!("missing {p}"));
 assert!(matches.next().is_none(),"duplicate {p}");l
}
fn number(s:&str)->f64 {let x:f64=s.parse().unwrap();assert!(x.is_finite()&&x>=0.);x}
fn main(){
 let reg=fs::read_to_string("results/C011/registry.csv").unwrap();
 let mut groups:BTreeMap<String,Vec<(u64,f64,f64,f64,u64)>>=BTreeMap::new();
 for line in reg.lines().skip(1){
  let x:Vec<_>=line.split(',').collect();assert_eq!(x.len(),19);
  let path=if x[2]=="2019"&&x[4]=="42"{format!("results/C011/screen/{}-{}.txt",x[1],x[3])}else{format!("results/C011/annual/{}-{}-{}-{}.txt",x[1],x[3],x[2],x[4])};
  let s=fs::read_to_string(&path).unwrap_or_else(|e|panic!("{path}: {e}"));
  let config:Vec<_>=field(&s,"representation-config,").split(',').collect();
  assert_eq!(&config[1..8],&[x[2],x[3],x[4],"8192","12","0.001",x[6]]);
  assert_eq!(config[10],"98304");
  let params:u64=config[8].parse().unwrap();let dim:u64=x[6].parse().unwrap();assert_eq!(params,(dim+1)*64+65*24);
  let fit:Vec<_>=field(&s,"representation-fit,").split(',').collect();assert_eq!(&fit[1..4],&[x[2],x[3],x[4]]);
  let wall=field(&s,"\tElapsed (wall clock) time (h:mm:ss or m:ss): ").split_once("): ").unwrap().1.split(':').fold(0.,|v,a|v*60.+number(a));
  let rss:u64=field(&s,"\tMaximum resident set size (kbytes): ").split_once(": ").unwrap().1.parse().unwrap();assert!(rss>0);
  assert_eq!(field(&s,"\tExit status: "), "\tExit status: 0");
  groups.entry(x[3].to_owned()).or_default().push((params,number(fit[4]),number(fit[5]),wall,rss));
 }
 assert_eq!(groups.len(),29);
 println!("model,outcomes,parameters,total_fit_s,total_predict_s,total_fit_predict_s,total_wall_s,max_rss_kib");
 for (name,v) in groups{assert_eq!(v.len(),12);let p=v[0].0;assert!(v.iter().all(|x|x.0==p));let fit:f64=v.iter().map(|x|x.1).sum();let pred:f64=v.iter().map(|x|x.2).sum();let wall:f64=v.iter().map(|x|x.3).sum();let rss=v.iter().map(|x|x.4).max().unwrap();println!("{name},12,{p},{fit:.6},{pred:.6},{:.6},{wall:.2},{rss}",fit+pred);}
}
