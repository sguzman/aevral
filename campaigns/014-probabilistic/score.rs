//! Frozen C014 scoring, lossless binary prediction exports and exact parameter evidence.
#[path="probabilistic.rs"] mod probabilistic;
use probabilistic::{ensemble::{self,Row},core::{self,Metrics,Issued},Model,regression::Node};
use std::{fs,io::{BufWriter,Write},path::Path,process::{Command,Stdio},time::Instant};
fn values(out:&mut impl Write,v:&[f64]){for x in v{write!(out,",{x:.17e}").unwrap();}writeln!(out).unwrap();}
fn node(out:&mut impl Write,h:usize,path:&str,n:&Node){
 match n{
 Node::Leaf{count,impurity,support}=>{write!(out,"leaf,{h},{path},{count},{impurity:.17e}").unwrap();values(out,support);}
 Node::Split{count,impurity,objective,feature,threshold,left,right}=>{
 writeln!(out,"split,{h},{path},{count},{impurity:.17e},{objective:.17e},{feature},{threshold:.17e}").unwrap();
 node(out,h,&format!("{path}L"),left);node(out,h,&format!("{path}R"),right);
 }}
}
fn parameters(out:&mut impl Write,m:&Model,year:u16,id:u8,rows:usize){
 writeln!(out,"recipe,{year},P{id:02},calibration_rows,{rows}").unwrap();
 match m{
 Model::Residual(r)=>{
 writeln!(out,"residual,{},{},{}",r.b,r.global,r.conditional).unwrap();
 for (h,t) in r.thresholds.iter().enumerate(){writeln!(out,"threshold,{},{t:.17e}",h+1).unwrap();}
 for (i,g) in r.groups.iter().enumerate(){write!(out,"group,{i},{},{},{:.17e},{:.17e}",g.count,g.fallback,g.mean,g.sd).unwrap();values(out,&g.support);}
 }
 Model::Quantile(q)=>{writeln!(out,"quantile,{}",q.b).unwrap();for(h,c)in q.curves.iter().enumerate(){for(j,v)in c.iter().enumerate(){writeln!(out,"curve,{},{},{:.17e},{:.17e},{:.17e}",h+1,j+1,v.intercept,v.slope,v.loss).unwrap();}}}
 Model::Tree(t)=>{writeln!(out,"tree,{}",t.b).unwrap();for(h,n)in t.roots.iter().enumerate(){node(out,h+1,"R",n);}}
 Model::Analog(a)=>{writeln!(out,"analog,{},{},128,{}",a.b,a.by_spread,if year==2021{"2020"}else{"2020+2021"}).unwrap();for(h,v)in a.leads.iter().enumerate(){writeln!(out,"analog-count,{},{}",h+1,v.len()).unwrap();}}
 Model::Mixture(m)=>{writeln!(out,"mixture,{}",m.id).unwrap();for(h,v)in m.residuals.iter().enumerate(){for(j,s)in v.iter().enumerate(){write!(out,"member-support,{},{j}",h+1).unwrap();values(out,s);}}}
 Model::Conformal(m)=>{writeln!(out,"conformal,{},{}",m.b,m.normalized).unwrap();for(h,v)in m.bands.iter().enumerate(){write!(out,"bands,{},{}",h+1,m.counts[h]).unwrap();for(l,u)in v{write!(out,",{l:.17e},{u:.17e}").unwrap();}writeln!(out).unwrap();}}
 }
}
fn f(out:&mut impl Write,x:f64){out.write_all(&x.to_le_bytes()).unwrap();}
/// Binary v1 record: i64 origin, u8 lead, u8 crossing, f64 target, median,
/// then for distributions nine quantiles, CRPS, mean absolute distance, half-pairwise term,
/// nine pinball losses; finally four lower/upper pairs. All little-endian.
fn record(out:&mut impl Write,r:&Row,p:&Issued,support:Option<&[f64]>,cross:bool){
 out.write_all(&r.x.origin.to_le_bytes()).unwrap();out.write_all(&[r.x.lead as u8,u8::from(cross)]).unwrap();f(out,r.target);f(out,p.median);
 if let Some(v)=support{
  for q in p.quantiles.unwrap(){f(out,q);}
  let n=v.len()as f64;let a=v.iter().map(|x|(x-r.target).abs()).sum::<f64>()/n;
  let b=v.iter().enumerate().map(|(i,x)|(2.*(i+1)as f64-n-1.)*(x-r.target)).sum::<f64>()/(n*n);
  f(out,p.crps.unwrap());f(out,a);f(out,b);
  for(i,q)in p.quantiles.unwrap().iter().enumerate(){f(out,core::pinball(r.target,*q,core::QUANTILES[i]));}
 }
 for(l,u)in p.bands{f(out,l);f(out,u);}
}
fn metric(out:&mut impl Write,year:u16,id:u8,group:&str,index:usize,s:&Metrics){
 assert!(s.n>0);let n=s.n as f64;let dist=id<=20;assert_eq!(s.distribution_n,if dist{s.n}else{0});
 write!(out,"{year},P{id:02},{group},{index},{},{:.8},{:.8}",s.n,s.ae/n,(s.se/n).sqrt()).unwrap();
 if dist{write!(out,",{:.8}",s.crps/n).unwrap();for p in s.pinball{write!(out,",{:.8}",p/n).unwrap();}}else{for _ in 0..10{write!(out,",NA").unwrap();}}
 for i in 0..4{write!(out,",{:.8},{:.8},{:.8}",s.coverage[i]/n,s.width[i]/n,s.interval_score[i]/n).unwrap();}
 writeln!(out).unwrap();
}
fn main(){
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),4,"score JOINED_DIR OUTPUT_DIR YEAR");
 let root=Path::new(&args[1]);let out=Path::new(&args[2]);let year:u16=args[3].parse().unwrap();assert!(matches!(year,2021|2022));fs::create_dir_all(out).unwrap();
 let history:Vec<_>=(2020..year).map(|y|ensemble::load(&root.join(format!("{y}.csv")),y)).collect();
 let calibration=ensemble::calibration(&history,year);assert_eq!(calibration.len(),if year==2021{181032}else{361056});
 // Freeze every model before loading current assessment. No assessment labels enter any fit.
 let mut fitted=Vec::new();
 for id in 1..=24{
  let now=Instant::now();let m=probabilistic::fit(id,&calibration);let fit=now.elapsed().as_secs_f64();
  let mut p=BufWriter::new(fs::File::create(out.join(format!("P{id:02}-parameters.csv"))).unwrap());parameters(&mut p,&m,year,id,calibration.len());p.flush().unwrap();
  fitted.push((id,m,fit));
 }
 drop(calibration);drop(history);
 let assessment=ensemble::load(&root.join(format!("{year}.csv")),year);
 let mut scores=BufWriter::new(fs::File::create(out.join("metrics.csv")).unwrap());
 let mut costs=BufWriter::new(fs::File::create(out.join("costs.csv")).unwrap());writeln!(costs,"fold,recipe,fit_s,issue_score_export_s,crossing_rows").unwrap();
 for(id,m,fit)in fitted{
  let file=fs::File::create(out.join(format!("P{id:02}-forecasts.bin.gz"))).unwrap();
  let mut child=Command::new("gzip").args(["-n","-1","-c"]).stdin(Stdio::piped()).stdout(Stdio::from(file)).spawn().unwrap();
  let mut pred=BufWriter::new(child.stdin.take().unwrap());pred.write_all(b"C014BIN1").unwrap();pred.write_all(&year.to_le_bytes()).unwrap();pred.write_all(&[id]).unwrap();
  let mut total=Metrics::default();let mut leads:[Metrics;24]=std::array::from_fn(|_|Metrics::default());let mut months:[Metrics;12]=std::array::from_fn(|_|Metrics::default());let mut crossings=0usize;let now=Instant::now();
  for r in &assessment.rows{
   let (support,cross)=m.support(&r.x);crossings+=usize::from(cross);
   let p=match &support{Some(v)=>core::distribution(v,r.target),None=>m.issue(&r.x,r.target).0};
   total.add(r.target,&p);leads[r.x.lead-1].add(r.target,&p);months[r.x.month-1].add(r.target,&p);record(&mut pred,r,&p,support.as_deref(),cross);
  }
  pred.flush().unwrap();drop(pred);assert!(child.wait().unwrap().success());
  let elapsed=now.elapsed().as_secs_f64();assert_eq!(total.n,assessment.rows.len());
  metric(&mut scores,year,id,"all",0,&total);for(i,s)in leads.iter().enumerate(){metric(&mut scores,year,id,"lead",i+1,s);}for(i,s)in months.iter().enumerate(){metric(&mut scores,year,id,"month",i+1,s);}
  writeln!(costs,"{year},P{id:02},{fit:.6},{elapsed:.6},{crossings}").unwrap();
  println!("completed,{year},P{id:02},{},{fit:.6},{elapsed:.6},{crossings}",total.n);
 }
 scores.flush().unwrap();costs.flush().unwrap();
}
