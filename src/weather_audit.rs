//! C018 pre-fit archive audit; no model fitting.
use aevral::{weather::{self,Station},features::tournament::{deploy,station::ConflictPolicy}};
use std::path::Path;
pub fn run(dir:&Path)->Result<(),String>{
    let paths=(2000..=2025).map(|y|dir.join(format!("LAX-{y}-01-01-{}-01-01.csv",y+1))).collect::<Vec<_>>();
    let end=weather::minute("2026-01-01 00:00")?;
    let(data,hashes)=deploy::load_verified(Station::new("LAX")?,&paths,60,end-60,ConflictPolicy::Strict)?;
    println!("year,bins,absent,missing_temperature,missing_origin_covariates,possible,temperature_complete,matched");
    for y in 2000..=2025{
        let start=weather::minute(&format!("{y}-01-01 00:00"))?;
        let stop=weather::minute(&format!("{}-01-01 00:00",y+1))?;
        let a=((start-data.first).max(0)/60)as usize;
        let b=((stop-data.first)/60)as usize;
        let bins=&data.bins[a..b];
        let absent=bins.iter().filter(|r|r.is_none()).count();
        let missing=bins.iter().filter(|r|r.is_some_and(|w|w.temperature.is_none())).count();
        let cov=bins.iter().filter(|r|r.is_some_and(|w|w.dew_c.is_none()||w.pressure_hpa.is_none()||w.wind_ms.is_none())).count();
        let c=deploy::coverage(&data,start.max(data.first),stop)?;
        println!("{y},{},{absent},{missing},{cov},{},{},{}",bins.len(),c.possible,c.temperature_complete,c.matched);
    }
    eprintln!("quarantined,{}",data.quarantined.len());
    for s in hashes{eprintln!("source,{},{}",s.name,s.sha256.iter().map(|b|format!("{b:02x}")).collect::<String>());}
    Ok(())
}
