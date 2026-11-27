//! C003 fixed-information benchmark. Fit gets only training rows; predict gets only origin inputs.
use super::{Case,Scale,input,H,Y};
use crate::{architecture::Model,benchmark::{Metrics,Climatology},covariates::Weather,weather::{minute,Split,Example,Archive}};
use std::{io::{BufReader,BufWriter,Write},path::Path,time::Instant};
pub const D:usize=80;
#[derive(Clone,Copy,Debug)]
pub struct Fold{pub id:&'static str,pub start:i64,pub end:i64}
impl Fold{
    pub fn development(id:&str)->Result<Self,String>{
        let (id,a,b)=match id{"legacy"=>("legacy",2019,2023),"2019"=>("2019",2019,2020),"2020"=>("2020",2020,2021),"2021"=>("2021",2021,2022),"2022"=>("2022",2022,2023),_=>return Err("unknown development fold".into())};
        Ok(Self{id,start:minute(&format!("{a}-01-01 00:00"))?,end:minute(&format!("{b}-01-01 00:00"))?})
    }
    pub fn partition(&self,t:i64)->Option<Split>{
        let a=t+60;let b=t+60*Y as i64;
        if b<self.start{Some(Split::Train)}else if a>=self.start&&b<self.end{Some(Split::Validation)}else{None}
    }
}
#[derive(Clone)]
pub struct OriginInput{pub origin:i64,pub x:[f64;D],pub recent:[f64;Y]}
pub struct TrainingRow{pub input:OriginInput,pub target:[f64;Y],pub normalized_target:[f64;Y]}
pub struct EvaluationRow{pub input:OriginInput,pub target:[f64;Y]}
pub struct PreparedFold{pub fold:Fold,pub train:Vec<TrainingRow>,pub eval:Vec<EvaluationRow>,pub possible:usize,pub temperature_complete:usize,pub target_mean:f64,pub target_sd:f64,pub scales:[f64;10]}
impl PreparedFold{
    pub fn restore(&self,y:[f64;Y])->[f64;Y]{y.map(|v|v*self.target_sd+self.target_mean)}
}
fn collect(bins:&[Option<Weather>],first:i64,fold:Fold)->Result<PreparedFold,String>{
    let(mut tr,mut ev)=(Vec::new(),Vec::new());let(mut possible,mut temperature_complete)=(0,0);
    for i in H-1..bins.len().saturating_sub(Y){
        let origin=first+i as i64*60;
        let Some(split)=fold.partition(origin)else{continue};
        if split==Split::Validation{possible+=1;}
        let v:Option<Vec<_>>=bins[i+1-H..=i+Y].iter().map(|r|r.and_then(|v|v.temperature)).collect();
        let Some(v)=v else{continue};
        if split==Split::Validation{temperature_complete+=1;}
        let Some(r)=bins[i]else{continue};
        let(Some(dew),Some(pressure),Some(wind))=(r.dew_c,r.pressure_hpa,r.wind_ms)else{continue};
        let row=Case{origin,split,x:v[..H].try_into().unwrap(),y:v[H..].try_into().unwrap(),weather:[dew,pressure,wind[0],wind[1]]};
        if split==Split::Train{tr.push(row)}else{ev.push(row)}
    }
    if tr.is_empty()||ev.is_empty(){return Err("empty fold".into())}
    let temp=Scale::fit(tr.iter().flat_map(|r|r.x))?;
    let weather:[Scale;4]=(0..4).map(|i|Scale::fit(tr.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    let make=|r:&Case|OriginInput{origin:r.origin,x:input(r,7,temp,weather).try_into().unwrap(),recent:r.x[H-Y..].try_into().unwrap()};
    let scales=[temp.mean,temp.sd,weather[0].mean,weather[0].sd,weather[1].mean,weather[1].sd,weather[2].mean,weather[2].sd,weather[3].mean,weather[3].sd];
    Ok(PreparedFold{fold,train:tr.iter().map(|r|TrainingRow{input:make(r),target:r.y,normalized_target:r.y.map(|v|temp.norm(v))}).collect(),eval:ev.iter().map(|r|EvaluationRow{input:make(r),target:r.y}).collect(),possible,temperature_complete,target_mean:temp.mean,target_sd:temp.sd,scales})
}
/// Only reads through the selected development fold. Historical holdout bytes not parsed.
pub fn load(data:&Path,fold:Fold)->Result<PreparedFold,String>{
    let mut a=Archive::default();let mut cov=crate::covariates::Archive::default();
    for y in 2000..=2022{
        if minute(&format!("{y}-01-01 00:00"))?>=fold.end{break}
        let p=data.join(format!("SEA-{y}.csv"));
        let open=||std::fs::File::open(&p).map(BufReader::new).map_err(|e|format!("{}: {e}",p.display()));
        a.ingest(open()?)?;cov.ingest(open()?)?;
    }
    let first=minute("2000-01-01 00:00")?+60;
    let bins=a.hourly(first,fold.end-60);let c=cov.hourly(first,fold.end-60);
    // Match the C002 alignment check (no quarantining in development).
    crate::covariates::audit(&c,first,&bins)?;
    collect(&c,first,fold)
}
pub trait Predictor{fn predict(&self,x:&OriginInput)->[f64;Y];}
pub struct Incumbent{model:Model,mean:f64,sd:f64}
impl Incumbent{
    pub fn fit(train:&[TrainingRow],mean:f64,sd:f64,seed:u64)->Result<Self,String>{
        if train.is_empty()||![42,123,2026].contains(&seed){return Err("invalid fit".into())}
        let mut model=Model::new("mlp64",seed)?;let mut order:Vec<_>=(0..train.len()).collect();let mut rng=seed;
        for epoch in 1..=30{
            for i in (1..order.len()).rev(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;order.swap(i,(rng%(i as u64+1))as usize);}
            let mut loss=0.;for &i in &order{let r=&train[i];loss+=model.train_step(&r.input.x,&r.normalized_target,0.005).0;}
            if !loss.is_finite(){return Err("nonfinite incumbent training".into())}
            println!("fit,mlp64,{epoch},{:.8}",loss/train.len()as f64);
        }
        Ok(Self{model,mean,sd})
    }
}
impl Predictor for Incumbent{fn predict(&self,x:&OriginInput)->[f64;Y]{self.model.predict(&x.x).map(|v|v*self.sd+self.mean)}}
struct Baseline{daily:bool}
impl Predictor for Baseline{fn predict(&self,x:&OriginInput)->[f64;Y]{if self.daily{x.recent}else{[x.recent[Y-1];Y]}}}
struct Climate(Climatology);
impl Predictor for Climate{fn predict(&self,x:&OriginInput)->[f64;Y]{self.0.predict(&Example{origin:x.origin,split:Split::Validation,x:x.recent,y:[0.;Y]})}}
/// Writes long-form matched predictions; score in Celsius. No target is passed to predictor.
pub fn score(p:&dyn Predictor,label:&str,fold:&PreparedFold,out:&mut impl Write)->Result<Metrics,String>{
    let mut m=Metrics::default();
    let mut months:[Metrics;12]=std::array::from_fn(|_|Metrics::default());
    let start_year=if fold.fold.id=="legacy"{2019}else{fold.fold.id.parse::<i64>().map_err(|e|e.to_string())?};
    let edges:Vec<_>=(start_year..=2022).flat_map(|y|(1..=12).map(move|m|(y,m))).map(|(y,m)|minute(&format!("{y}-{m:02}-01 00:00")).map(|t|(t,m as usize))).collect::<Result<_,_>>()?;
    let now=Instant::now();for r in &fold.eval{
        let pred=p.predict(&r.input);
        if pred.iter().any(|v|!v.is_finite()){return Err("nonfinite prediction".into())}
        m.add(&r.target,&pred);
        let idx=edges.partition_point(|(t,_)|*t<=r.input.origin+60)-1;
        months[edges[idx].1-1].add(&r.target,&pred);
        for h in 0..Y{writeln!(out,"{},{label},{},{},{},{}",fold.fold.id,r.input.origin,h+1,r.target[h],pred[h]).map_err(|e|e.to_string())?;}
    }
    println!("score,{},{label},{},{:.8},{:.8},{:.3}",fold.fold.id,m.count,m.mae(),m.rmse(),now.elapsed().as_secs_f64());
    for h in 0..Y{let(a,b)=m.lead(h);println!("lead,{},{label},{},{a:.8},{b:.8}",fold.fold.id,h+1);}
    for(i,v)in months.iter().enumerate(){if v.count>0{println!("month,{},{label},{},{},{:.8},{:.8}",fold.fold.id,i+1,v.count,v.mae(),v.rmse());}}
    Ok(m)
}
pub fn run(data:&Path,id:&str,output:&Path)->Result<(),String>{
    let f=load(data,Fold::development(id)?)?;
    println!("cohort,{id},{},{},{},{}",f.train.len(),f.eval.len(),f.possible,f.temperature_complete);
    println!("scales,{id},{:?}",f.scales);
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let daily=score(&Baseline{daily:true},"daily",&f,&mut out)?;
    score(&Baseline{daily:false},"persistence",&f,&mut out)?;
    let rows:Vec<_>=f.train.iter().map(|r|Example{origin:r.input.origin,split:Split::Train,x:r.input.recent,y:r.target}).collect();
    let clim=Climate(Climatology::fit(&rows.iter().collect::<Vec<_>>())?);drop(rows);
    score(&clim,"climatology",&f,&mut out)?;
    let now=Instant::now();let m=Incumbent::fit(&f.train,f.target_mean,f.target_sd,42)?;
    println!("fit-seconds,{id},mlp64,{:.3}",now.elapsed().as_secs_f64());
    let inc=score(&m,"mlp64",&f,&mut out)?;
    out.flush().map_err(|e|e.to_string())?;
    if id=="legacy"&&(f.train.len()!=121849||f.eval.len()!=30207||(daily.mae()-2.05315123).abs()>1e-6||(inc.mae()-1.35014439).abs()>1e-6){return Err("legacy reproduction mismatch".into())}
    Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn strict_boundaries(){
        for id in ["legacy","2019","2020","2021","2022"]{let f=Fold::development(id).unwrap();
            assert_eq!(f.partition(f.start-25*60),Some(Split::Train));
            assert_eq!(f.partition(f.start-24*60),None);
            assert_eq!(f.partition(f.start-60),Some(Split::Validation));
            assert_eq!(f.partition(f.end-25*60),Some(Split::Validation));
            assert_eq!(f.partition(f.end-24*60),None);
        }
    }
    #[test]fn metric_parity(){
        let mut m=Metrics::default();m.add(&[2.;Y],&[3.;Y]);m.add(&[4.;Y],&[2.;Y]);
        assert_eq!(m.mae(),1.5);assert!((m.rmse()-2.5f64.sqrt()).abs()<1e-12);assert_eq!(m.lead(0).0,1.5);
    }
    #[test]fn input_excludes_targets(){
        let x=OriginInput{origin:0,x:[0.;D],recent:[2.;Y]};
        assert_eq!(Baseline{daily:true}.predict(&x),[2.;Y]);
    }
}

#[cfg(test)]
mod causal_tests{
    use super::*;
    fn fixture()->(Fold,i64,Vec<Option<Weather>>){
        let f=Fold::development("2019").unwrap();let first=f.start-200*60;
        let bins=(0..400).map(|i|Some(Weather{minute:first+i*60,temperature:Some((i%31)as f64),dew_c:Some((i%17)as f64),pressure_hpa:Some(1000.+(i%13)as f64),wind_ms:Some([(i%11)as f64,(i%7)as f64])})).collect();(f,first,bins)
    }
    #[test]fn validation_labels_cannot_change_training_scales(){
        let(f,first,mut bins)=fixture();let a=collect(&bins,first,f).unwrap();
        for b in &mut bins[200..]{b.as_mut().unwrap().temperature=Some(99.);}
        let b=collect(&bins,first,f).unwrap();
        assert_eq!(a.scales,b.scales);assert_eq!(a.train.len(),b.train.len());
        assert!(a.train.iter().zip(&b.train).all(|(x,y)|x.input.x==y.input.x&&x.target==y.target));
        assert_ne!(a.eval[0].target,b.eval[0].target);
    }
    #[test]fn missing_temperature_excludes_windows_and_missing_covariate_excludes_origin(){
        let(f,first,mut bins)=fixture();let a=collect(&bins,first,f).unwrap();
        bins[260].as_mut().unwrap().dew_c=None;
        let b=collect(&bins,first,f).unwrap();assert_eq!(a.eval.len(),b.eval.len()+1);
        assert_eq!(a.temperature_complete,b.temperature_complete);
        bins[270].as_mut().unwrap().temperature=None;
        let c=collect(&bins,first,f).unwrap();
        assert!(c.temperature_complete<b.temperature_complete);
        let missing=first+270*60;
        assert!(c.eval.iter().all(|r|missing<r.input.origin-71*60||missing>r.input.origin+24*60));
    }
}

pub mod linear;

pub mod sparse;

pub mod robust;

pub mod latent;

pub mod basis;

pub mod mars;

pub mod autoreg;

pub mod arima;

pub mod smoothing;

pub mod kalman;

pub mod switching;

pub mod cart;

pub mod forest;

pub mod boosting;

pub mod upstream;

pub mod neighbors;
