//! Reusable weather forecast bundle; fitting, calibration and inference are separate.
use super::{artifact::{self,Artifact,Recipe,SourceFingerprint},local::LocalRidge,neural::AdamMlp,p13::{P13,CalibrationOrigin,SUPPORT},upstream::LightGbm,station::StationBins,Incumbent,OriginScales,OriginInput,TrainingRow,Predictor,Y};
use crate::weather::Station;
use std::path::Path;

pub struct Members{incumbent:Incumbent,light:LightGbm,ridge:LocalRidge,adam:AdamMlp}
impl Members{
    pub fn fit(train:&[TrainingRow],scales:OriginScales)->Result<Self,String>{
        let v=scales.values();let(mean,sd)=(v[0],v[1]);
        Ok(Self{incumbent:Incumbent::fit(train,mean,sd,42)?,light:LightGbm::fit(train,mean,sd)?,ridge:LocalRidge::fit(train,mean,sd)?,adam:AdamMlp::fit(train,mean,sd)?})
    }
    pub fn predict_batch(&self,rows:&[OriginInput])->Result<Vec<[[f64;Y];4]>,String>{
        let l=self.light.predict_batch(rows)?;
        rows.iter().zip(l).map(|(x,l)|{
            let p=[self.incumbent.predict(x),l,self.ridge.predict(x)?,self.adam.predict(x)?];
            if p.iter().flatten().any(|v|!v.is_finite()){return Err("nonfinite member prediction".into())}Ok(p)
        }).collect()
    }
}
/// Matched evaluation origins, using training-fitted transforms only.
pub fn evaluation(data:&StationBins,start:i64,end:i64,scales:OriginScales)->Result<Vec<super::EvaluationRow>,String>{
    if start<data.first||end<=start||start%60!=0||end%60!=0||end>data.last+60{return Err("invalid assessment interval".into())}
    let mut rows=Vec::new();
    for i in super::H-1..data.bins.len().saturating_sub(Y){
        let origin=data.first+i as i64*60;
        if origin+60<start{continue}
        if origin+1440>=end{break}
        let Ok(raw)=data.origin(origin) else{continue};
        let mut target=[0.;Y];let mut valid=true;
        for (j,b) in data.bins[i+1..=i+Y].iter().enumerate(){
            let t=origin+60*(j+1)as i64;
            match b{
                Some(r) if r.minute>=t-59&&r.minute<=t&&r.temperature.is_some_and(|v|v.is_finite())=>target[j]=r.temperature.unwrap(),
                _=>{valid=false;break}
            }
        }
        if valid{rows.push(super::EvaluationRow{input:scales.encode(&raw)?,target});}
    }
    if rows.is_empty(){return Err("empty matched assessment cohort".into())}
    Ok(rows)
}
#[derive(Debug,Default)]
pub struct Coverage{pub possible:usize,pub temperature_complete:usize,pub matched:usize}
/// Count the natural and matched cohorts without fitting a model.
pub fn coverage(data:&StationBins,start:i64,end:i64)->Result<Coverage,String>{
    if start<data.first||end<=start||start%60!=0||end%60!=0||end>data.last+60{return Err("invalid coverage interval".into())}
    let mut c=Coverage::default();
    for i in super::H-1..data.bins.len().saturating_sub(Y){
        let origin=data.first+i as i64*60;
        if origin+60<start{continue}
        if origin+1440>=end{break}
        c.possible+=1;
        if !data.bins[i+1-super::H..=i+Y].iter().enumerate().all(|(j,b)|{
            let t=origin+(j as i64-(super::H as i64-1))*60;
            b.is_some_and(|r|r.minute>=t-59&&r.minute<=t&&r.temperature.is_some_and(|v|v.is_finite()))
        }){continue}
        c.temperature_complete+=1;
        if data.origin(origin).is_ok(){c.matched+=1;}
    }
    Ok(c)
}
/// Generate prior-only calibration from contiguous complete calendar-year folds.
/// The earliest eligible calibration year is 2020, matching C015. CLI chooses
/// an explicit finite range; no assessment rows may enter the calibrator.
pub fn calibrate(data:&StationBins,first_year:i64,last_year:i64,cutoff:i64)->Result<Vec<CalibrationOrigin>,String>{
    if first_year<2020||last_year<first_year||last_year-first_year>9||last_year>=crate::weather::MAX_YEAR{return Err("invalid calibration years".into())}
    let final_end=crate::weather::minute(&format!("{}-01-01 00:00",last_year+1))?;
    if final_end>cutoff||cutoff>data.last+60{return Err("calibration overlaps final fit cutoff".into())}
    let mut result=Vec::new();
    for year in first_year..=last_year{
        let start=crate::weather::minute(&format!("{year}-01-01 00:00"))?;
        let end=crate::weather::minute(&format!("{}-01-01 00:00",year+1))?;
        let (train,scales)=super::prepare_training(&data.bins,data.first,start)?;
        let eval=evaluation(data,start,end,scales)?;
        println!("calibration-fold,{year},{},{}",train.len(),eval.len());
        let model=Members::fit(&train,scales)?;
        let inputs:Vec<_>=eval.iter().map(|r|r.input.clone()).collect();
        let predictions=model.predict_batch(&inputs)?;
        if predictions.len()!=eval.len(){return Err("calibration prediction count mismatch".into())}
        for (r,members) in eval.into_iter().zip(predictions){
            result.push(CalibrationOrigin{origin:r.input.origin,fit_cutoff:start,target:r.target,members});
        }
    }
    Ok(result)
}
enum Model{Mlp(Incumbent),P13(Members,P13)}
pub struct ForecastBundle{pub station:Station,pub cutoff:i64,pub scales:OriginScales,calibration_years:Option<(i64,i64)>,model:Model}
pub struct Forecast{pub origin:i64,pub point:[f64;Y],pub support:Option<[[f64;SUPPORT];Y]>}
impl ForecastBundle{
    /// Fits from station-bound completed training labels only.
    /// P13 calibration inputs must be separately generated out of fold.
    pub fn train(data:&StationBins,cutoff:i64,recipe:Recipe,calibration:&[CalibrationOrigin])->Result<Self,String>{
        if cutoff<=data.first||cutoff>data.last+60{return Err("fit cutoff outside archive".into())}
        let (rows,scales)=super::prepare_training(&data.bins,data.first,cutoff)?;
        let v=scales.values();
        let model=match recipe{
            Recipe::Mlp=>{if !calibration.is_empty(){return Err("unexpected MLP calibration".into())}Model::Mlp(Incumbent::fit(&rows,v[0],v[1],42)?)}
            Recipe::P13=>{let p=P13::fit(calibration,cutoff)?;Model::P13(Members::fit(&rows,scales)?,p)}
        };
        Ok(Self{station:data.station.clone(),cutoff,scales,calibration_years:None,model})
    }
    pub fn artifact(&self,sources:Vec<SourceFingerprint>)->Result<Artifact,String>{
        let (recipe,components)=match &self.model{
            Model::Mlp(m)=>(Recipe::Mlp,vec![("I".into(),m.model_bytes()?)]),
            Model::P13(m,p)=>(Recipe::P13,vec![("I".into(),m.incumbent.model_bytes()?),("L".into(),m.light.model_bytes()?),("R".into(),m.ridge.model_bytes()?),("M".into(),m.adam.model_bytes()?),("P13".into(),p.model_bytes()?)]),
        };
        Ok(Artifact{station:self.station.clone(),cutoff:self.cutoff,scales:self.scales,recipe,calibration_years:self.calibration_years,sources,components})
    }
    pub fn from_artifact(a:Artifact)->Result<Self,String>{
        a.validate()?;
        let v=a.scales.values();let(mean,sd)=(v[0],v[1]);
        let mut c=a.components.into_iter();
        let mut take=|expected:&str|->Result<Vec<u8>,String>{
            let(id,p)=c.next().ok_or("missing component")?;
            if id!=expected{return Err("component order mismatch".into())}Ok(p)
        };
        let incumbent=Incumbent::from_model_bytes(&take("I")?,mean,sd)?;
        let model=match a.recipe{
            Recipe::Mlp=>Model::Mlp(incumbent),
            Recipe::P13=>{
                let light=LightGbm::from_model_bytes(&take("L")?,mean,sd)?;
                let ridge=LocalRidge::from_model_bytes(&take("R")?,mean,sd)?;
                let adam=AdamMlp::from_model_bytes(&take("M")?,mean,sd)?;
                let p=P13::from_model_bytes(&take("P13")?)?;
                Model::P13(Members{incumbent,light,ridge,adam},p)
            }
        };
        if c.next().is_some(){return Err("extra components".into())}
        Ok(Self{station:a.station,cutoff:a.cutoff,scales:a.scales,calibration_years:a.calibration_years,model})
    }
    pub fn train_chronological(data:&StationBins,cutoff:i64,recipe:Recipe,years:Option<(i64,i64)>)->Result<Self,String>{
        match (recipe,years){
            (Recipe::Mlp,None)=>Self::train(data,cutoff,recipe,&[]),
            (Recipe::P13,Some((a,b)))=>{
                let mut model=Self::train(data,cutoff,recipe,&calibrate(data,a,b,cutoff)?)?;
                model.calibration_years=Some((a,b));Ok(model)
            },
            _=>Err("calibration years required only for P13".into())
        }
    }
    pub fn load(path:&Path)->Result<Self,String>{Self::from_artifact(Artifact::load(path)?)}
    /// Evaluate a frozen cohort in one backend batch. Labels are exposed only
    /// to the scoring callback, after prediction. Explicitly opt into station transfer.
    /// C015's first eligible origin is cutoff-60 (all fit labels <= that origin).
    pub fn evaluate_batch(&self,data:&StationBins,start:i64,end:i64,transfer:bool,mut consume:impl FnMut(&super::EvaluationRow,&[f64;Y],Option<&[[f64;Y];4]>,Option<&[[f64;SUPPORT];Y]>)->Result<(),String>)->Result<usize,String>{
        if start<self.cutoff{return Err("assessment starts before fit cutoff".into())}
        if data.station!=self.station&&!transfer{return Err("assessment station mismatch".into())}
        let rows=evaluation(data,start,end,self.scales)?;
        match &self.model{
            Model::Mlp(m)=>{
                for r in &rows{let point=m.predict(&r.input);if point.iter().any(|v|!v.is_finite()){return Err("nonfinite assessment".into())}consume(r,&point,None,None)?;}
            }
            Model::P13(m,p)=>{
                let inputs:Vec<_>=rows.iter().map(|r|r.input.clone()).collect();
                let members=m.predict_batch(&inputs)?;
                if members.len()!=rows.len(){return Err("assessment prediction count mismatch".into())}
                for (r,b) in rows.iter().zip(&members){
                    let s=p.support(b)?;
                    let point=std::array::from_fn(|h|P13::median(&s[h]));
                    consume(r,&point,Some(b),Some(&s))?;
                }
            }
        }
        Ok(rows.len())
    }
    pub fn forecast(&self,data:&StationBins,issue:i64)->Result<Forecast,String>{
        if data.station!=self.station{return Err("forecast station mismatch".into())}
        self.forecast_inner(data,issue)
    }
    /// Explicit transfer evaluation only: never silently relabel station identity.
    pub fn transfer_forecast(&self,data:&StationBins,issue:i64)->Result<Forecast,String>{
        self.forecast_inner(data,issue)
    }
    fn forecast_inner(&self,data:&StationBins,issue:i64)->Result<Forecast,String>{
        if issue<self.cutoff{return Err("forecast precedes model cutoff".into())}
        let x=self.scales.encode(&data.origin(issue)?)?;
        let(point,support)=match &self.model{
            Model::Mlp(m)=>(m.predict(&x),None),
            Model::P13(m,p)=>{
                let members=m.predict_batch(std::slice::from_ref(&x))?;
                let s=p.support(&members[0])?;let point=std::array::from_fn(|h|P13::median(&s[h]));(point,Some(s))
            }
        };
        if point.iter().any(|v|!v.is_finite()){return Err("nonfinite forecast".into())}
        Ok(Forecast{origin:issue,point,support})
    }
}
/// Load only verified station observations, binding the returned fingerprints
/// to the exact inputs consumed by the station archive.
pub fn load_verified(station:Station,paths:&[std::path::PathBuf],first:i64,last:i64,policy:super::station::ConflictPolicy)->Result<(StationBins,Vec<SourceFingerprint>),String>{
    let before=fingerprints(paths)?;
    let data=StationBins::load(station,paths,first,last,policy)?;
    if fingerprints(paths)?!=before{return Err("source changed during ingestion".into())}
    Ok((data,before))
}
/// Fingerprint actual source bytes; callers verify these again after reading.
pub fn fingerprints(paths:&[std::path::PathBuf])->Result<Vec<SourceFingerprint>,String>{
    let mut out=Vec::with_capacity(paths.len());
    for p in paths{
        let name=p.file_name().and_then(|v|v.to_str()).ok_or("invalid source filename")?.to_owned();
        let bytes=std::fs::read(p).map_err(|e|format!("{}: {e}",p.display()))?;
        out.push(SourceFingerprint{name,sha256:artifact::sha256(&bytes)?});
    }
    out.sort_by(|a,b|a.name.cmp(&b.name));
    if out.is_empty()||out.windows(2).any(|w|w[0].name==w[1].name){return Err("empty or duplicate source names".into())}
    Ok(out)
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn incumbent_bundle_exact_reload(){
        let station=Station::new("LAX").unwrap();let first=60;let cutoff=180*60;
        let bins=(0..200).map(|i|Some(crate::covariates::Weather{minute:first+i*60,temperature:Some((i%23)as f64),dew_c:Some((i%17)as f64),pressure_hpa:Some(1000.+(i%13)as f64),wind_ms:Some([(i%11)as f64,(i%7)as f64])})).collect();
        let data=StationBins{station:station.clone(),first,last:200*60,bins,quarantined:vec![]};
        let (_,scales)=super::super::prepare_training(&data.bins,data.first,cutoff).unwrap();
        let eval=evaluation(&data,120*60,170*60,scales).unwrap();
        assert_eq!(eval.first().unwrap().input.origin,119*60);
        assert_eq!(eval.last().unwrap().input.origin,145*60);
        let counts=coverage(&data,120*60,170*60).unwrap();
        assert_eq!(counts.possible,eval.len());assert_eq!(counts.matched,eval.len());

        assert!(calibrate(&data,2020,2021,cutoff).is_err());
        let model=ForecastBundle::train(&data,cutoff,Recipe::Mlp,&[]).unwrap();
        let a=model.artifact(vec![SourceFingerprint{name:"synthetic.csv".into(),sha256:[0;32]}]).unwrap();
        let bytes=a.encode().unwrap();let reloaded=ForecastBundle::from_artifact(Artifact::decode(&bytes).unwrap()).unwrap();
        assert_eq!(model.forecast(&data,cutoff).unwrap().point,reloaded.forecast(&data,cutoff).unwrap().point);
        assert!(reloaded.forecast(&data,cutoff-60).is_err());
        let mut count=0;
        let n=reloaded.evaluate_batch(&data,cutoff,200*60,false,|r,p,b,s|{
            assert!(r.input.origin+60>=cutoff);
            assert!(p.iter().all(|v|v.is_finite()));assert!(b.is_none()&&s.is_none());count+=1;Ok(())
        });
        // This short fixture has no complete 24h assessment after the fit cutoff.
        assert!(n.is_err());assert_eq!(count,0);

        let mut other=data;other.station=Station::default();assert!(reloaded.forecast(&other,cutoff).is_err());
    }
}
