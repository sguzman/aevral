//! Station-bound observation loading for training and live forecasts.
use crate::{covariates::{self, Weather}, weather::{self, Station}};
use std::{fs::File, io::{BufRead, BufReader}, path::PathBuf};

/// Conflict handling is explicit. LegacyExposed retains C015's exact policy:
/// conflicting reports before 2023 fail, conflicts in later years are quarantined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictPolicy { Strict, LegacyExposed }

pub struct StationBins {
    pub station: Station,
    pub first: i64,
    pub last: i64,
    pub bins: Vec<Option<Weather>>,
    pub quarantined: Vec<i64>,
}
impl StationBins {
    /// The requested interval consists of completed UTC hour endpoints.
    /// Source files must already have been acquired with a verified station identity.
    pub fn load(station: Station, paths: &[PathBuf], first: i64, last: i64, policy: ConflictPolicy) -> Result<Self,String> {
        if paths.is_empty() { return Err("no source files".into()); }
        let mut a=weather::Archive::for_station(station.clone());
        let mut b=covariates::Archive::for_station(station.clone());
        for path in paths {
            let open=|| File::open(path).map(BufReader::new).map_err(|e|format!("{}: {e}",path.display()));
            ingest_pair(&mut a,&mut b,open()?,open()?,policy).map_err(|e|format!("{}: {e}",path.display()))?;
        }
        finish(station,a,b,first,last)
    }
    /// Get an exact completed origin. Never silently move to an earlier complete hour.
    pub fn origin(&self, issue: i64) -> Result<super::RawOrigin,String> {
        if issue<self.first || issue>self.last || (issue-self.first)%60!=0 {
            return Err("issue outside loaded completed hours".into());
        }
        let i=((issue-self.first)/60) as usize;
        if i+1<super::H {return Err("insufficient history".into());}
        super::RawOrigin::from_history(issue,&self.bins[i+1-super::H..=i])
    }
}
fn ingest_pair(a:&mut weather::Archive,b:&mut covariates::Archive,temp:impl BufRead,met:impl BufRead,policy:ConflictPolicy)->Result<(),String>{
    let quarantine=policy==ConflictPolicy::LegacyExposed;
    a.ingest_policy(temp,quarantine)?;
    b.ingest_policy(met,quarantine)
}
fn finish(station:Station,mut a:weather::Archive,mut b:covariates::Archive,first:i64,last:i64)->Result<StationBins,String>{
    // The historical parsers currently support calendar years 2000–2026.
    let end=weather::minute("2026-12-31 23:59")?;
    if first<60 || last<first || last>end || first%60!=0 || last%60!=0 {
        return Err("invalid completed-hour interval".into());
    }
    if a.is_empty() || a.station()!=&station || b.station()!=&station {return Err("empty or mismatched source archive".into());}
    let quarantined:Vec<_>=a.conflicts.union(&b.conflicts).copied().collect();
    for &t in &quarantined {a.quarantine(t); b.quarantine(t);}
    let reference=a.hourly(first,last);
    let bins=b.hourly(first,last);
    if reference.len()!=bins.len() || reference.iter().zip(&bins).any(|(x,y)|x.map(|r|(r.minute,r.celsius))!=y.map(|r|(r.minute,r.temperature))){
        return Err("temperature/covariate alignment mismatch".into());
    }
    Ok(StationBins{station,first,last,bins,quarantined})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_origin_and_conflict_union() {
        let station=Station::new("LAX").unwrap();
        let first=weather::minute("2024-01-01 00:00").unwrap()+60;
        let mut a=weather::Archive::for_station(station.clone());
        let mut b=covariates::Archive::for_station(station.clone());
        let mut csv=String::from("station,valid,tmpf,dwpf,alti,drct,sknt\n");
        for day in 1..=4 {
            for hour in 0..24 {
                csv.push_str(&format!("LAX,2024-01-{day:02} {hour:02}:59,68,50,30,90,10\n"));
            }
        }
        // A covariate-only conflict also quarantines the temperature archive.
        csv.push_str("LAX,2024-01-04 00:59,68,51,30,90,10\n");
        ingest_pair(&mut a,&mut b,csv.as_bytes(),csv.as_bytes(),ConflictPolicy::LegacyExposed).unwrap();
        let data=finish(station,a,b,first,first+95*60).unwrap();
        assert_eq!(data.quarantined.len(),1);
        assert!(data.origin(first+71*60).is_ok());
        assert!(data.origin(first+72*60).is_err());
        assert!(data.origin(first+96*60).is_err());
        assert!(data.origin(first+71*60+1).is_err());
    }
    #[test]
    #[ignore = "requires verified frozen KSEA archive"]
    fn frozen_ksea_parity() {
        use super::super::{collect,prepare_training,retrospective,OriginScales};
        let root=PathBuf::from(std::env::var("AEVRAL_PARITY_DATA").unwrap());
        let year=std::env::var("AEVRAL_PARITY_YEAR").unwrap();
        let (first, old, fold)=retrospective::load_bins(&root,&year).unwrap();
        let paths:Vec<_>=(2000..=year.parse::<i32>().unwrap()).map(|y|root.join(format!("SEA-{y}.csv"))).collect();
        let data=StationBins::load(Station::default(),&paths,first,fold.end-60,ConflictPolicy::LegacyExposed).unwrap();
        assert_eq!(old,data.bins);
        let historical=collect(&old,first,fold).unwrap();
        let (train, scales)=prepare_training(&data.bins,first,fold.start).unwrap();
        assert_eq!(scales.values(),historical.scales);
        assert_eq!(train.len(),historical.train.len());
        for (a,b) in train.iter().zip(&historical.train) {
            assert_eq!(a.input.origin,b.input.origin);
            assert_eq!(a.input.x,b.input.x);
            assert_eq!(a.input.recent,b.input.recent);
            assert_eq!(a.target,b.target);
            assert_eq!(a.normalized_target,b.normalized_target);
        }
        let scales=OriginScales::from_values(historical.scales).unwrap();
        for r in &historical.eval {
            let input=scales.encode(&data.origin(r.input.origin).unwrap()).unwrap();
            assert_eq!(input.x,r.input.x);
            assert_eq!(input.recent,r.input.recent);
        }
        println!("station-parity,{year},{},{},pass",train.len(),historical.eval.len());
    }
}
