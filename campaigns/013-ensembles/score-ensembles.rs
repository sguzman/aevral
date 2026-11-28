//! Execute the frozen 24 recipes for 2021 and 2022, using prior-fold labels only.
mod ensemble;
use ensemble::{Model,Learned,Linear,Row,Scores,N};
use std::{fs,io::{BufWriter,Write},path::Path,time::Instant};
const ALIASES:&[u8]=b"ILRMSAHX";
fn coeff(out:&mut impl Write,year:u16,id:u8,group:&str,m:&Linear,n:usize,fallback:bool){
 write!(out,"{year},E{id:02},{group},{n},{fallback},{},{},{:.17},{:.17}",m.iterations,m.converged,m.objective,m.intercept).unwrap();
 for w in m.weights{write!(out,",{w:.17}").unwrap();}writeln!(out).unwrap();
}
fn record(out:&mut impl Write,extras:&mut impl Write,year:u16,id:u8,m:&Model,cal:&[&Row]){
 match m{
  Model::Static(k)=>{let (method,cols)=ensemble::static_recipe(*k).unwrap();writeln!(extras,"static,{year},E{id:02},{method:?},{}",cols.iter().map(|&i|char::from(ALIASES[i])).collect::<String>()).unwrap();}
  Model::Linear(Learned::Global(l))=>coeff(out,year,id,"global",l,cal.len(),false),
  Model::Linear(Learned::Lead(v))=>for (h,l) in v.iter().enumerate(){coeff(out,year,id,&format!("lead-{}",h+1),l,cal.iter().filter(|r|r.x.lead==h+1).count(),false);},
  Model::Corrected{base,bias}=>{
   record(out,extras,year,id,base,cal);
   for(h,b)in bias.iter().enumerate(){writeln!(extras,"bias,{year},E{id:02},{},{b:.17}",h+1).unwrap();}
  }
  Model::Spread{threshold,models,counts}=>{
   writeln!(extras,"spread-threshold,{year},E{id:02},{threshold:.17}").unwrap();
   for (g,l) in models.iter().enumerate(){coeff(out,year,id,&format!("spread-{g}"),l,counts[g],counts[g]<1000);}
  }
  Model::Seasonal{models,counts}=>for(g,l)in models.iter().enumerate(){coeff(out,year,id,&format!("season-{g}"),l,counts[g],counts[g]<1000);},
  Model::Selected{cols,correlations}=>{
   writeln!(extras,"selected,{year},E{id:02},{}",cols.iter().map(|&i|char::from(ALIASES[i])).collect::<String>()).unwrap();
   for i in 0..N{for j in i+1..N{let v=correlations[i][j].map(|x|format!("{x:.17}")).unwrap_or_else(||"undefined".into());writeln!(extras,"correlation,{year},E{id:02},{},{},{v}",char::from(ALIASES[i]),char::from(ALIASES[j])).unwrap();}}
  }
 }
}
fn report(out:&mut impl Write,year:u16,id:&str,s:&Scores,n:usize,fit:f64,predict:f64){
 s.validate(n);
 writeln!(out,"score,{year},{id},{n},{:.8},{:.8},{fit:.6},{predict:.6}",s.all.mae(),s.all.rmse()).unwrap();
 for(h,m)in s.leads.iter().enumerate(){writeln!(out,"lead,{year},{id},{},{},{:.8},{:.8}",h+1,m.count,m.mae(),m.rmse()).unwrap();}
 for(h,m)in s.months.iter().enumerate(){assert!(m.count>0);writeln!(out,"month,{year},{id},{},{},{:.8},{:.8}",h+1,m.count/24,m.mae(),m.rmse()).unwrap();}
}
fn main(){
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),3,"usage: score-ensembles JOINED_DIR OUTPUT_DIR");
 let root=Path::new(&args[1]);let out=Path::new(&args[2]);fs::create_dir_all(out).unwrap();
 let mut metrics=BufWriter::new(fs::File::create(out.join("metrics.csv")).unwrap());
 let mut coefficients=BufWriter::new(fs::File::create(out.join("coefficients.csv")).unwrap());
 writeln!(coefficients,"fold,recipe,group,calibration_rows,fallback,iterations,converged,objective,intercept,I,L,R,M,S,A,H,X").unwrap();
 let mut extras=BufWriter::new(fs::File::create(out.join("parameters.csv")).unwrap());
 let mut history=vec![ensemble::load(&root.join("2020.csv"),2020)];
 for year in [2021u16,2022]{
  let calibration=ensemble::calibration(&history,year);
  writeln!(extras,"calibration,{year},{},{}",history.iter().map(|f|f.year.to_string()).collect::<Vec<_>>().join("+"),calibration.len()).unwrap();
  // All fits complete before the current assessment file is loaded.
  let mut fitted=Vec::new();
  for id in 1..=24{let now=Instant::now();let m=ensemble::fit(id,&calibration);let fit=now.elapsed().as_secs_f64();record(&mut coefficients,&mut extras,year,id,&m,&calibration);fitted.push((id,m,fit));}
  drop(calibration);
  let assessment=ensemble::load(&root.join(format!("{year}.csv")),year);let n=assessment.rows.len()/24;
  let mut predictions=BufWriter::new(fs::File::create(out.join(format!("predictions-{year}.csv"))).unwrap());
  write!(predictions,"fold,origin,lead,target,daily,I,L,R,M,S,A,H,X").unwrap();for id in 1..=24{write!(predictions,",E{id:02}").unwrap();}writeln!(predictions).unwrap();
  let mut baselines:Vec<Scores>=(0..9).map(|_|Scores::default()).collect();
  let mut scores:Vec<Scores>=(0..24).map(|_|Scores::default()).collect();let mut timings=[0.;24];
  for r in &assessment.rows{
   write!(predictions,"{year},{},{},{},{}",r.x.origin,r.x.lead,r.target,r.daily).unwrap();
   baselines[0].add(r,r.daily);
   for(i,p)in r.x.base.iter().enumerate(){baselines[i+1].add(r,*p);write!(predictions,",{p}").unwrap();}
   for (i,(_,m,_)) in fitted.iter().enumerate(){
    let now=Instant::now();let p=m.predict(&r.x);timings[i]+=now.elapsed().as_secs_f64();
    scores[i].add(r,p);write!(predictions,",{p}").unwrap();
   }
   writeln!(predictions).unwrap();
  }
  predictions.flush().unwrap();
  report(&mut metrics,year,"daily",&baselines[0],n,0.,0.);
  for(i,&alias)in ALIASES.iter().enumerate(){report(&mut metrics,year,&char::from(alias).to_string(),&baselines[i+1],n,0.,0.);}
  for(i,(id,_,fit))in fitted.iter().enumerate(){report(&mut metrics,year,&format!("E{id:02}"),&scores[i],n,*fit,timings[i]);}
  history.push(assessment);
 }
 metrics.flush().unwrap();coefficients.flush().unwrap();extras.flush().unwrap();
}
