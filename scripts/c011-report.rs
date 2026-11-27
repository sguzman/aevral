use std::{collections::BTreeMap,fs};
fn field<'a>(s:&'a str,p:&str)->&'a str {
 let mut matches=s.lines().map(str::trim_start).filter(|l|l.starts_with(p));
 let l=matches.next().unwrap_or_else(||panic!("missing {p}"));
 assert!(matches.next().is_none(),"duplicate {p}");l
}
fn number(s:&str)->f64 {let x:f64=s.parse().unwrap();assert!(x.is_finite()&&x>=0.);x}
fn main(){
 let reg=fs::read_to_string("results/C011/registry.csv").unwrap();
 let supplement=fs::read_to_string("/tmp/c011-timing-recovered.csv").unwrap();
 let mut recovered:BTreeMap<&str,(f64,u64)>=BTreeMap::new();
 for l in supplement.lines().skip(1){let v:Vec<_>=l.split(',').collect();assert_eq!(v.len(),3);assert!(recovered.insert(v[0],(number(v[1]),v[2].parse().unwrap())).is_none());}
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
  let wall_prefix="Elapsed (wall clock) time (h:mm:ss or m:ss): ";
  let rss_prefix="Maximum resident set size (kbytes): ";
  let wall_line=s.lines().map(str::trim_start).find(|l|l.starts_with(wall_prefix));
  let rss_line=s.lines().map(str::trim_start).find(|l|l.starts_with(rss_prefix));
  let (wall,rss)=match(wall_line,rss_line){
   (Some(a),Some(b))=>{assert!(!recovered.contains_key(x[16]));(a.strip_prefix(wall_prefix).unwrap().split(':').fold(0.,|v,a|v*60.+number(a)),b.strip_prefix(rss_prefix).unwrap().parse::<u64>().unwrap())},
   _=>{let pair=recovered.remove(x[16]).unwrap_or_else(||panic!("missing timing for {path}"));if let Some(a)=wall_line{assert!((a.strip_prefix(wall_prefix).unwrap().split(':').fold(0.,|v,a|v*60.+number(a))-pair.0).abs()<1e-8);}if let Some(b)=rss_line{assert_eq!(b.strip_prefix(rss_prefix).unwrap().parse::<u64>().unwrap(),pair.1);}pair}
  };assert!(rss>0);
  assert_eq!(field(&s,"Exit status: "), "Exit status: 0");
  groups.entry(x[3].to_owned()).or_default().push((params,number(fit[4]),number(fit[5]),wall,rss));
 }
 assert!(recovered.is_empty(),"unused recovered timing records");
 assert_eq!(groups.len(),29);
 println!("model,outcomes,parameters,total_fit_s,total_predict_s,total_fit_predict_s,total_wall_s,max_rss_kib");
 for (name,v) in groups{assert_eq!(v.len(),12);let p=v[0].0;assert!(v.iter().all(|x|x.0==p));let fit:f64=v.iter().map(|x|x.1).sum();let pred:f64=v.iter().map(|x|x.2).sum();let wall:f64=v.iter().map(|x|x.3).sum();let rss=v.iter().map(|x|x.4).max().unwrap();println!("{name},12,{p},{fit:.6},{pred:.6},{:.6},{wall:.2},{rss}",fit+pred);}
}
