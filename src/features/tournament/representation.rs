//! C011 causal same-archive features. Origin history is a prefix, targets are separate.
use super::{Fold,PreparedFold,D,Y,H,collect,load_bins};
use crate::{adam::Adam,dense::Dense,benchmark::Metrics,covariates::Weather,weather::{minute,Split},windows::Calendar};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
type Bin=Option<Weather>;
#[derive(Clone,Copy,Debug)]
enum Recipe{Base,Suffix(usize),Lag(&'static [usize]),Rolling(&'static [usize]),Gradient(&'static [usize]),Harmonics(usize),Depression(bool),Pressure(&'static [usize]),Wind(bool),WeatherHistory(&'static [usize])}
impl Recipe{
 fn parse(label:&str)->Result<Self,String>{Ok(match label{
 "base80"=>Self::Base,"suffix24"=>Self::Suffix(24),"suffix48"=>Self::Suffix(48),
 "sparse-lags-14d"=>Self::Lag(&[72,95,119,143,167,335]),"sparse-lags-28d"=>Self::Lag(&[72,95,119,143,167,335,503,671]),
 "rolling-6-24-72"=>Self::Rolling(&[6,24,72]),"rolling-3-12-48"=>Self::Rolling(&[3,12,48]),
 "gradient-short"=>Self::Gradient(&[1,3,6]),"gradient-long"=>Self::Gradient(&[6,12,24]),
 "harmonics-2-3"=>Self::Harmonics(3),"harmonics-2-4"=>Self::Harmonics(4),
 "dew-depression-add"=>Self::Depression(false),"dew-depression-replace"=>Self::Depression(true),
 "pressure-tendency-short"=>Self::Pressure(&[3,6,12]),"pressure-tendency-long"=>Self::Pressure(&[6,24,48]),
 "wind-speed"=>Self::Wind(false),"wind-unit"=>Self::Wind(true),
 "weather-history-short"=>Self::WeatherHistory(&[1,3,6]),"weather-history-long"=>Self::WeatherHistory(&[6,12,24]),_=>return Err("unknown representation".into())})}
 fn optional(self)->bool{matches!(self,Self::Lag(_)|Self::Pressure(_)|Self::WeatherHistory(_))}
 fn binary(self,j:usize)->bool{matches!(self,Self::Wind(true))&&j==3}
 fn width(self)->usize{match self{Self::Base=>D,Self::Suffix(n)=>n+8,Self::Lag(l)=>D+2*l.len(),Self::Rolling(l)=>D+3*l.len(),Self::Gradient(l)=>D+2*l.len(),Self::Harmonics(k)=>D+4*(k-1),Self::Depression(replace)=>D+usize::from(!replace),Self::Pressure(l)=>D+2*l.len(),Self::Wind(unit)=>D+if unit{4}else{1},Self::WeatherHistory(l)=>D+8*l.len()}}
 fn extras(self,prefix:&[Bin],origin:i64)->Result<Vec<Option<f64>>,String>{
  let report=|lag:usize|prefix.len().checked_sub(lag+1).and_then(|i|prefix[i]);
  let at=|lag:usize|report(lag).and_then(|b|b.temperature);
  let required=|h|at(h).ok_or_else(||"missing required lag".to_string());
  let mut out=Vec::new();match self{
   Self::Lag(l)=>out.extend(l.iter().map(|&h|at(h))),
   Self::Rolling(l)=>for &n in l{let x=(0..n).rev().map(required).collect::<Result<Vec<_>,_>>()?;let mean=x.iter().sum::<f64>()/n as f64;let range=x.iter().copied().fold(f64::NEG_INFINITY,f64::max)-x.iter().copied().fold(f64::INFINITY,f64::min);let var=x.iter().map(|v|(v-mean).powi(2)).sum::<f64>()/n as f64;out.extend([Some(mean),Some(range),Some(var)]);},
   Self::Gradient(l)=>for &h in l{let a=required(0)?;let b=required(h)?;let c=required(2*h)?;out.extend([Some((a-b)/h as f64),Some((a-2.*b+c)/(h*h)as f64)]);},
   Self::Harmonics(max)=>{let phase=Calendar::Both.features(origin);let daily=phase[0].atan2(phase[1]);let annual=phase[2].atan2(phase[3]);for k in 2..=max{let a=k as f64*daily;let b=k as f64*annual;out.extend([Some(a.sin()),Some(a.cos()),Some(b.sin()),Some(b.cos())]);}},
   Self::Depression(_)=>{let dew=report(0).and_then(|b|b.dew_c).ok_or("missing origin dew point")?;out.push(Some(required(0)?-dew));},
   Self::Pressure(l)=>{let p=report(0).and_then(|b|b.pressure_hpa).ok_or("missing origin pressure")?;for &h in l{out.push(report(h).and_then(|b|b.pressure_hpa).map(|v|(p-v)/h as f64));}},
   Self::Wind(unit)=>{let [u,v]=report(0).and_then(|b|b.wind_ms).ok_or("missing origin wind")?;let speed=u.hypot(v);out.push(Some(speed));if unit{let calm=speed==0.;out.extend([Some(if calm{0.}else{u/speed}),Some(if calm{0.}else{v/speed}),Some(if calm{1.}else{0.})]);}},
   Self::WeatherHistory(l)=>for &h in l{let r=report(h);let wind=r.and_then(|b|b.wind_ms);out.extend([r.and_then(|b|b.dew_c),r.and_then(|b|b.pressure_hpa),wind.map(|w|w[0]),wind.map(|w|w[1])]);},
   _=>{}
  }Ok(out)
 }
}
fn base(prefix:&[Bin],origin:i64,n:usize,s:&[f64;10])->Option<Vec<f64>>{
 let start=prefix.len().checked_sub(n)?;let r=prefix.last().copied().flatten()?;
 let w=r.wind_ms?;let weather=[r.dew_c?,r.pressure_hpa?,w[0],w[1]];
 let mut x:Vec<_>=prefix[start..].iter().map(|b|b.and_then(|v|v.temperature).map(|v|(v-s[0])/s[1])).collect::<Option<_>>()?;
 x.extend(Calendar::Both.features(origin));x.extend(weather.iter().enumerate().map(|(i,v)|(v-s[2*i+2])/s[2*i+3]));Some(x)
}
fn target(future:&[Bin])->Option<[f64;Y]>{if future.len()!=Y{return None}future.iter().map(|b|b.and_then(|v|v.temperature)).collect::<Option<Vec<_>>>()?.try_into().ok()}
fn index(first:i64,t:i64,len:usize)->Result<usize,String>{let d=t-first;if d<0||d%60!=0||d/60>=len as i64{return Err("origin outside bins".into())}Ok((d/60)as usize)}
#[derive(Clone,Copy,Debug,PartialEq)]
struct ExtraScale{mean:f64,sd:f64,observed:usize,constant:bool}
fn fit_scales(rows:&[Vec<Option<f64>>])->Result<Vec<ExtraScale>,String>{
 let d=rows.first().ok_or("empty training extras")?.len();if rows.iter().any(|r|r.len()!=d||r.iter().flatten().any(|v|!v.is_finite())){return Err("bad extras".into())}
 Ok((0..d).map(|j|{let v:Vec<_>=rows.iter().filter_map(|r|r[j]).collect();let n=v.len();let mean=if n==0{0.}else{v.iter().sum::<f64>()/n as f64};let sd=if n==0{0.}else{(v.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/n as f64).sqrt()};ExtraScale{mean,sd:if sd==0.{1.}else{sd},observed:n,constant:sd==0.}}).collect())
}
fn encode(recipe:Recipe,mut x:Vec<f64>,raw:&[Option<f64>],scales:&[ExtraScale])->Result<Vec<f64>,String>{
 if let Recipe::Suffix(n)=recipe{x=x[H-n..].to_vec()}
 if raw.len()!=scales.len(){return Err("scale dimension".into())}
 for (j,(&v,s)) in raw.iter().zip(scales).enumerate(){if v.is_none()&&!recipe.optional(){return Err("required feature missing".into())}let value=if recipe.binary(j){let b=v.ok_or("missing binary feature")?;if b!=0.&&b!=1.{return Err("invalid binary feature".into())}b}else{v.map_or(0.,|a|(a-s.mean)/s.sd)};
  if matches!(recipe,Recipe::Depression(true)){x[76]=value}else{x.push(value)}if recipe.optional(){x.push(if v.is_some(){1.}else{0.});}}
 if x.len()!=recipe.width()||x.iter().any(|v|!v.is_finite()){return Err("encoded dimension or finiteness".into())}Ok(x)
}
fn verify(f:&PreparedFold,bins:&[Bin],first:i64)->Result<(),String>{
 let(mut ti,mut ei)=(0,0);let mut possible=0;
 for i in H-1..bins.len().saturating_sub(Y){let origin=first+i as i64*60;let Some(split)=f.fold.partition(origin)else{continue};if split==Split::Validation{possible+=1;}
  let Some(x)=base(&bins[..=i],origin,H,&f.scales)else{continue};let Some(y)=target(&bins[i+1..i+1+Y])else{continue};
  let(input,expected)=if split==Split::Train{let r=f.train.get(ti).ok_or("extra train origin")?;ti+=1;(&r.input,&r.target)}else{let r=f.eval.get(ei).ok_or("extra eval origin")?;ei+=1;(&r.input,&r.target)};
  if input.origin!=origin||x.as_slice()!=input.x.as_slice()||&y!=expected{return Err("old/new origin, input or target mismatch".into())}
 }
 if(ti,ei,possible)!=(f.train.len(),f.eval.len(),f.possible){return Err("old/new cohort mismatch".into())}Ok(())
}
struct Rng(u64);
impl Rng{fn next(&mut self)->u64{self.0=self.0.wrapping_add(0x9E3779B97F4A7C15);let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9);z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);z^(z>>31)}fn shuffle(&mut self,x:&mut[usize]){for i in (1..x.len()).rev(){let u=(self.next()>>11)as f64/9007199254740992.;x.swap(i,(u*(i+1)as f64)as usize)}}}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>1400.{Err("runtime guard".into())}else{Ok(())}}
fn fit(x:&[Vec<f64>],y:&[[f64;Y]],seed:u64,epochs:usize,now:Instant,log:Option<(&str,&str)>)->Result<Dense,String>{
 let mut m=Dense::new(&[x[0].len(),64,Y],seed);let mut opt=Adam::new(m.parameters(),0.001);let mut rng=Rng(seed^0xD1B54A32D192ED03);let mut order:Vec<_>=(0..x.len()).collect();
 for epoch in 1..=epochs{rng.shuffle(&mut order);let mut loss=0.;for &i in &order{guard(now)?;loss+=m.adam_step(&x[i],&y[i],&mut opt).0;if !m.finite(){return Err("nonfinite fit".into())}}if !loss.is_finite(){return Err("nonfinite loss".into())}if let Some((id,label))=log{println!("representation-epoch,{id},{label},{seed},{epoch},{:.10}",loss/x.len()as f64);}}Ok(m)
}
fn forecast(m:&Dense,x:&[f64],f:&PreparedFold)->Result<[f64;Y],String>{let p:[f64;Y]=m.predict(x).try_into().map_err(|_|"prediction width")?;let p=f.restore(p);if p.iter().any(|v|!v.is_finite()){return Err("nonfinite prediction".into())}Ok(p)}
fn write_rows(path:&Path,id:&str,label:&str,rows:&[(i64,[f64;Y],[f64;Y])],tag:&str)->Result<Metrics,String>{
 let mut out=BufWriter::new(std::fs::File::create(path).map_err(|e|e.to_string())?);writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
 let year=id.parse::<i64>().map_err(|e|e.to_string())?;let edges=(1..=12).map(|m|minute(&format!("{year}-{m:02}-01 00:00"))).collect::<Result<Vec<_>,_>>()?;
 let mut metric=Metrics::default();let mut months:[Metrics;12]=std::array::from_fn(|_|Metrics::default());
 for (t,y,p) in rows{metric.add(y,p);let i=edges.partition_point(|v|*v<=*t+60).checked_sub(1).ok_or("month outside fold")?;months[i].add(y,p);for j in 0..Y{writeln!(out,"{id},{label},{t},{},{},{}",j+1,y[j],p[j]).map_err(|e|e.to_string())?;}}
 out.flush().map_err(|e|e.to_string())?;
 println!("{tag}score,{id},{label},{},{:.8},{:.8}",metric.count,metric.mae(),metric.rmse());
 for j in 0..Y{let(a,b)=metric.lead(j);println!("{tag}lead,{id},{label},{},{a:.8},{b:.8}",j+1)}
 for(i,m)in months.iter().enumerate(){if m.count>0{println!("{tag}month,{id},{label},{},{},{:.8},{:.8}",i+1,m.count,m.mae(),m.rmse())}}
 Ok(metric)
}
fn origins(path:&Path,ts:impl Iterator<Item=i64>)->Result<(),String>{let mut w=BufWriter::new(std::fs::File::create(path).map_err(|e|e.to_string())?);for t in ts{writeln!(w,"{t}").map_err(|e|e.to_string())?;}w.flush().map_err(|e|e.to_string())}
fn coverage(recipe:Recipe,bins:&[Bin],first:i64,ts:impl Iterator<Item=i64>,d:usize)->Result<(usize,usize,Vec<usize>),String>{
 let(mut n,mut all)=(0,0);let mut seen=vec![0;d];for t in ts{let i=index(first,t,bins.len())?;let raw=recipe.extras(&bins[..=i],t)?;if raw.len()!=d{return Err("coverage dimension".into())}n+=1;all+=usize::from(raw.iter().all(Option::is_some));for (k,v) in raw.iter().enumerate(){seen[k]+=usize::from(v.is_some());}}Ok((n,all,seen))
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str,seed:u64)->Result<(),String>{
 if ![42,123,2026].contains(&seed){return Err("seed outside frozen recipe".into())}let recipe=Recipe::parse(label)?;let now=Instant::now();let fold=Fold::development(id)?;let(first,bins)=load_bins(data,fold)?;let f=collect(&bins,first,fold)?;verify(&f,&bins,first)?;
 let(n,m)=match id{"2019"=>(121849,7455),"2020"=>(129327,7543),"2021"=>(136893,7501),"2022"=>(144417,7639),_=>return Err("annual folds only".into())};if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort".into())}
 let dir=output.parent().ok_or("output directory")?;origins(&dir.join("train-origins.txt"),f.train.iter().map(|r|r.input.origin))?;origins(&dir.join("matched-origins.txt"),f.eval.iter().map(|r|r.input.origin))?;
 let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
 let subset:Vec<_>=(0..8192).map(|i|&f.train[i*n/8192]).collect();origins(&dir.join("fit-origins.txt"),subset.iter().map(|r|r.input.origin))?;
 let raw=subset.iter().map(|r|{let i=index(first,r.input.origin,bins.len())?;recipe.extras(&bins[..=i],r.input.origin)}).collect::<Result<Vec<_>,_>>()?;let mut es=fit_scales(&raw)?;for(j,s)in es.iter_mut().enumerate(){if recipe.binary(j){s.mean=0.;s.sd=1.;}}
 for (j,s) in es.iter().enumerate(){println!("representation-scale,{id},{label},{j},{},{},{},{}",s.mean,s.sd,s.observed,s.constant)}
 let tr=subset.iter().zip(&raw).map(|(r,a)|encode(recipe,r.input.x.to_vec(),a,&es)).collect::<Result<Vec<_>,_>>()?;let ty:Vec<_>=subset.iter().map(|r|r.normalized_target).collect();
 let ev=f.eval.iter().map(|r|{let i=index(first,r.input.origin,bins.len())?;encode(recipe,r.input.x.to_vec(),&recipe.extras(&bins[..=i],r.input.origin)?,&es)}).collect::<Result<Vec<_>,_>>()?;
 if recipe.optional(){for(split,ts)in [("train",f.train.iter().map(|r|r.input.origin).collect::<Vec<_>>()),("eval",f.eval.iter().map(|r|r.input.origin).collect::<Vec<_>>())]{let(n,all,seen)=coverage(recipe,&bins,first,ts.into_iter(),es.len())?;println!("representation-available,{id},{label},{split},{n},{all}");for(j,count)in seen.into_iter().enumerate(){println!("representation-availability,{id},{label},{split},{j},{count},{}",n-count);}}}
 let dim=recipe.width();let params=Dense::new(&[dim,64,Y],seed).parameters();let workspace=8*(16*params+(8192+m)*dim+8192*Y+2*m*Y+65536);
 println!("representation-config,{id},{label},{seed},8192,12,0.001,{dim},{params},{workspace},98304");
 let prep=now.elapsed().as_secs_f64();let t=Instant::now();let px:Vec<_>=(0..128).map(|i|tr[i*8192/128].clone()).collect();let py:Vec<_>=(0..128).map(|i|ty[i*8192/128]).collect();let probe=fit(&px,&py,seed,1,now,None)?;let fp=t.elapsed().as_secs_f64();let t=Instant::now();for i in 0..32{std::hint::black_box(forecast(&probe,&tr[i*8192/32],&f)?);}let pp=t.elapsed().as_secs_f64();
 let estimate=fp*768.+pp*(f.possible.max(m)as f64)/32.+prep;println!("representation-preflight,{id},{label},{seed},{prep:.6},{fp:.6},{pp:.6},{estimate:.6}");
 if workspace>2_000_000_000||2.*estimate+180.>1400.{println!("blocked,{id},{label},preflight resource guard");return Ok(())}
 let t=Instant::now();let model=fit(&tr,&ty,seed,12,now,Some((id,label)))?;let fit_seconds=t.elapsed().as_secs_f64();let t=Instant::now();
 let mut matched=Vec::with_capacity(m);for(r,x)in f.eval.iter().zip(&ev){guard(now)?;matched.push((r.input.origin,r.target,forecast(&model,x,&f)?));}
 let metric=write_rows(output,id,label,&matched,"")?;
 if label=="base80"{let reference=include_str!("../../../results/C010/registry.csv");let row=reference.lines().find_map(|l|{let x:Vec<_>=l.split(',').collect();if x.len()==12&&x[1]=="F01"&&x[2]==id&&x[3]=="joint-mlp64"&&x[4]==seed.to_string(){Some(x)}else{None}}).ok_or("missing reference")?;let a=row[6].parse::<f64>().map_err(|e|e.to_string())?;let b=row[7].parse::<f64>().map_err(|e|e.to_string())?;if(metric.mae()-a).abs()>1e-7||(metric.rmse()-b).abs()>1e-7{return Err("C010 control parity".into())}println!("representation-control,{id},{seed},pass");}
 if let Recipe::Suffix(w)=recipe{let mut natural=Vec::new();let(mut possible,mut overlap)=(0,0);for i in w-1..bins.len().saturating_sub(Y){let origin=first+i as i64*60;if fold.partition(origin)!=Some(Split::Validation){continue}possible+=1;let Some(x)=base(&bins[..=i],origin,w,&f.scales)else{continue};let Some(y)=target(&bins[i+1..i+1+Y])else{continue};guard(now)?;let p=forecast(&model,&x,&f)?;if let Ok(j)=matched.binary_search_by_key(&origin,|v|v.0){if matched[j].1!=y||matched[j].2!=p{return Err("natural overlap parity".into())}overlap+=1;}natural.push((origin,y,p));}if overlap!=m{return Err("natural missing matched origins".into())}origins(&dir.join("natural-origins.txt"),natural.iter().map(|v|v.0))?;write_rows(&dir.join("natural-predictions.csv"),id,label,&natural,"natural-")?;println!("representation-natural,{id},{label},{possible},{},{overlap}",natural.len());}else{println!("representation-natural,{id},{label},{},{m},{m}",f.possible);}
 println!("representation-fit,{id},{label},{seed},{fit_seconds:.6},{:.6}",t.elapsed().as_secs_f64());Ok(())
}
#[cfg(test)]mod tests{
 use super::*;
 fn fixture(n:usize)->Vec<Bin>{(0..n).map(|i|Some(Weather{minute:i as i64*60,temperature:Some(i as f64),dew_c:Some(i as f64/2.),pressure_hpa:Some(1000.),wind_ms:Some([1.,2.])})).collect()}
 #[test]fn rolling_exact(){let b=fixture(72);let x=Recipe::Rolling(&[3]).extras(&b,(b.len()-1)as i64*60).unwrap();assert_eq!(x[0],Some(70.));assert_eq!(x[1],Some(2.));assert!((x[2].unwrap()-2./3.).abs()<1e-12)}
 #[test]fn gradients_exact(){let b=fixture(72);assert_eq!(Recipe::Gradient(&[1,3,6]).extras(&b,(b.len()-1)as i64*60).unwrap(),vec![Some(1.),Some(0.),Some(1.),Some(0.),Some(1.),Some(0.)]);}
 #[test]fn optional_exact_masks(){let mut b=fixture(100);b[4]=None;let r=Recipe::Lag(&[72,95,119]);let raw=r.extras(&b,(b.len()-1)as i64*60).unwrap();assert_eq!(raw,vec![Some(27.),None,None]);let s=fit_scales(&[raw.clone(),vec![Some(29.),Some(5.),None]]).unwrap();assert_eq!(encode(r,vec![0.;80],&raw,&s).unwrap()[80..],[-1.,1.,0.,0.,0.,0.]);}
 #[test]fn suffix_and_future_invariance(){let mut b=fixture(100);let s=[0.,1.,0.,1.,0.,1.,0.,1.,0.,1.];let a=base(&b[..72],71*60,72,&s).unwrap();let f=Recipe::Gradient(&[1,3]).extras(&b[..72],71*60).unwrap();b[72]=None;assert_eq!(base(&b[..72],71*60,72,&s).unwrap(),a);assert_eq!(Recipe::Gradient(&[1,3]).extras(&b[..72],71*60).unwrap(),f);b[0]=None;assert!(base(&b[..72],71*60,72,&s).is_none());assert_eq!(base(&b[..72],71*60,24,&s).unwrap(),a[48..]);}
 #[test]fn all_first_recipes_dimensions(){let b=fixture(700);for label in ["base80","suffix24","suffix48","sparse-lags-14d","sparse-lags-28d","rolling-6-24-72","rolling-3-12-48","gradient-short","gradient-long"]{let r=Recipe::parse(label).unwrap();let raw=r.extras(&b,(b.len()-1)as i64*60).unwrap();let s=fit_scales(&[raw.clone()]).unwrap();assert_eq!(encode(r,vec![0.;D],&raw,&s).unwrap().len(),r.width());}}
 #[test]fn deterministic_fit(){let x=vec![vec![1.,2.],vec![0.,-1.]];let y=vec![[1.;Y],[0.;Y]];let a=fit(&x,&y,42,2,Instant::now(),None).unwrap();let b=fit(&x,&y,42,2,Instant::now(),None).unwrap();assert_eq!(a.predict(&x[0]),b.predict(&x[0]));}

 #[test]fn harmonics_calendar_and_leap(){let b=fixture(1);let r=Recipe::Harmonics(4);let start=minute("2020-01-01 00:00").unwrap();let a=r.extras(&b,start).unwrap();for pair in a.chunks_exact(2){assert!(pair[0].unwrap().abs()<1e-12);assert!((pair[1].unwrap()-1.).abs()<1e-12)}let t=minute("2020-07-02 06:00").unwrap();let phase=std::f64::consts::TAU*(t-start)as f64/(366.*1440.);let a=r.extras(&b,t).unwrap();assert!((a[0].unwrap()).abs()<1e-12);assert!((a[1].unwrap()+1.).abs()<1e-12);assert!((a[2].unwrap()-(2.*phase).sin()).abs()<1e-12);assert!((a[3].unwrap()-(2.*phase).cos()).abs()<1e-12);}
 #[test]fn dew_depression_replacement(){let b=fixture(73);let raw=Recipe::Depression(true).extras(&b,72*60).unwrap();assert_eq!(raw,vec![Some(36.)]);let scales=fit_scales(&[raw.clone(),vec![Some(34.)]]).unwrap();let x:Vec<_>=(0..80).map(|i|i as f64).collect();let y=encode(Recipe::Depression(true),x.clone(),&raw,&scales).unwrap();assert_eq!(y[76],1.);for i in 0..80{if i!=76{assert_eq!(x[i],y[i]);}}assert_eq!(encode(Recipe::Depression(false),x,&raw,&scales).unwrap()[80],1.);}
 #[test]fn pressure_exact_bin_and_missing(){let mut b=fixture(73);for(i,r)in b.iter_mut().enumerate(){r.as_mut().unwrap().pressure_hpa=Some(900.+2.*i as f64)}b[66].as_mut().unwrap().pressure_hpa=None;let r=Recipe::Pressure(&[3,6,12]);let a=r.extras(&b,72*60).unwrap();assert_eq!(a,vec![Some(2.),None,Some(2.)]);let s=fit_scales(&[a.clone()]).unwrap();assert_eq!(encode(r,vec![0.;80],&a,&s).unwrap()[80..],[0.,1.,0.,0.,0.,1.]);}
 #[test]fn wind_calm_and_binary(){let mut b=fixture(1);b[0].as_mut().unwrap().wind_ms=Some([3.,4.]);let r=Recipe::Wind(true);let a=r.extras(&b,0).unwrap();assert_eq!(a,vec![Some(5.),Some(0.6),Some(0.8),Some(0.)]);b[0].as_mut().unwrap().wind_ms=Some([0.,0.]);let c=r.extras(&b,0).unwrap();assert_eq!(c,vec![Some(0.),Some(0.),Some(0.),Some(1.)]);let s=fit_scales(&[a.clone(),c.clone()]).unwrap();assert_eq!(encode(r,vec![0.;80],&a,&s).unwrap()[83],0.);assert_eq!(encode(r,vec![0.;80],&c,&s).unwrap()[83],1.);}
 #[test]fn weather_history_individual_masks_and_future_invariance(){let mut b=fixture(80);b[71].as_mut().unwrap().dew_c=None;b[69].as_mut().unwrap().wind_ms=None;b[66]=None;let r=Recipe::WeatherHistory(&[1,3,6]);let a=r.extras(&b[..73],72*60).unwrap();assert_eq!(a,vec![None,Some(1000.),Some(1.),Some(2.),Some(34.5),Some(1000.),None,None,None,None,None,None]);b[73]=None;assert_eq!(r.extras(&b[..73],72*60).unwrap(),a);let s=fit_scales(&[a.clone()]).unwrap();let x=encode(r,vec![0.;80],&a,&s).unwrap();assert_eq!(x[81],0.);assert_eq!(x[83],1.);assert_eq!(x[93],0.);assert_eq!(x.len(),104);}
 #[test]fn second_batch_dimensions(){let b=fixture(100);for label in ["harmonics-2-3","harmonics-2-4","dew-depression-add","dew-depression-replace","pressure-tendency-short","pressure-tendency-long","wind-speed","wind-unit","weather-history-short","weather-history-long"]{let r=Recipe::parse(label).unwrap();let raw=r.extras(&b,99*60).unwrap();let s=fit_scales(&[raw.clone()]).unwrap();assert_eq!(encode(r,vec![0.;D],&raw,&s).unwrap().len(),r.width());}}
}
