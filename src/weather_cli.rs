//! User-facing weather commands. No implicit network access or station relabeling.
use aevral::{weather::{self,Station},features::tournament::{artifact::{Artifact,Recipe},deploy::{self,ForecastBundle},station::ConflictPolicy,p13::P13}};
use std::{collections::BTreeMap,path::{Path,PathBuf},time::{SystemTime,UNIX_EPOCH},io::Write};
const HELP:&str="Usage:
  aevral weather train --station KLAX --model p13 [--data DIR] [--artifact FILE] [--as-of UTC] [--calibration-start YEAR] [--calibration-end YEAR]
  aevral weather forecast --station KLAX --hours 24 [--model p13|mlp] [--data DIR] [--artifact FILE] [--as-of UTC] [--output CSV]
UTC format: YYYY-MM-DDTHH:MM (completed hour, defaults to current UTC hour).
Stations: KLAX, KSEA. Models: p13 (default), mlp. Source: IEM routine UTC CSV.
Defaults: data/raw/lax or data/raw/ksea; models/KLAX-p13.aev or corresponding station/model.
Calibration defaults: last three complete calendar years, never before 2020.
Artifacts and output files are never overwritten. Acquire observations separately.";
fn now()->Result<i64,String>{
    let seconds=SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e|e.to_string())?.as_secs();
    let minutes=i64::try_from(seconds/60).map_err(|_|"clock overflow")?-15778080;
    if minutes<0||minutes>=weather::minute_limit(){return Err("unsupported clock".into())}Ok(minutes/60*60)
}
fn timestamp(s:&str)->Result<i64,String>{
    let t=weather::minute(&s.replace('T'," "))?;
    if t%60!=0||t<60{return Err("as-of must be a completed UTC hour".into())}Ok(t)
}
fn files(dir:&Path)->Result<Vec<PathBuf>,String>{
    let mut v=Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e|format!("{}: {e}",dir.display()))?{
        let p=entry.map_err(|e|e.to_string())?.path();
        if p.is_file()&&p.extension().and_then(|s|s.to_str())==Some("csv"){v.push(p);}
    }
    v.sort();if v.is_empty(){return Err("no CSV observations in data directory".into())}Ok(v)
}
fn options(args:&[String],allowed:&[&str])->Result<BTreeMap<String,String>,String>{
    if args.len()%2!=0{return Err("each option needs a value".into())}
    let mut o=BTreeMap::new();for p in args.chunks_exact(2){
        if !allowed.contains(&p[0].as_str())||p[1].starts_with("--")||o.insert(p[0].clone(),p[1].clone()).is_some(){return Err(format!("invalid or duplicate option {}",p[0]))}
    }Ok(o)
}
pub fn run(args:&[String])->Result<(),String>{
    if args.is_empty()||args==["--help"]||args==["help"]{println!("{HELP}");return Ok(())}
    let cmd=args[0].as_str();
    if args.get(1).is_some_and(|x|x=="--help"){println!("{HELP}");return Ok(())}
    let allowed=match cmd{
        "train"=>vec!["--station","--model","--data","--artifact","--as-of","--calibration-start","--calibration-end"],
        "forecast"=>vec!["--station","--model","--data","--artifact","--as-of","--hours","--output"],
        _=>return Err(format!("unknown weather command {cmd}")),
    };
    let o=options(&args[1..],&allowed)?;let get=|k:&str|o.get(k).map(String::as_str);
    let (icao,provider)=match get("--station"){Some("KLAX")=>("KLAX","LAX"),Some("KSEA")=>("KSEA","SEA"),_=>return Err("specify --station KLAX or KSEA".into())};
    let name=get("--model").unwrap_or("p13");
    let recipe=match name{"mlp"=>Recipe::Mlp,"p13"=>Recipe::P13,_=>return Err("model must be mlp or p13".into())};
    let station=Station::new(provider)?;
    let clock=now()?;let issue=get("--as-of").map(timestamp).transpose()?.unwrap_or(clock);
    if issue>clock{return Err("as-of is in the future".into())}
    let dir=get("--data").map(PathBuf::from).unwrap_or_else(||PathBuf::from(format!("data/raw/{}",provider.to_ascii_lowercase())));
    let path=get("--artifact").map(PathBuf::from).unwrap_or_else(||PathBuf::from(format!("models/{icao}-{name}.aev")));
    let sources=files(&dir)?;
    let policy=if icao=="KSEA"{ConflictPolicy::LegacyExposed}else{ConflictPolicy::Strict};
    match cmd{
        "train"=>{
            if path.exists(){return Err(format!("artifact already exists: {}",path.display()))}
            let year=weather::year_start(issue)?.0;
            let parse=|k:&str,default:i64|->Result<i64,String>{get(k).map(|s|s.parse::<i64>().map_err(|_|format!("invalid {k}"))).transpose().map(|x|x.unwrap_or(default))};
            let years=if recipe==Recipe::P13{Some((parse("--calibration-start",(year-3).max(2020))?,parse("--calibration-end",year-1)?))}else{
                if get("--calibration-start").is_some()||get("--calibration-end").is_some(){return Err("calibration applies only to p13".into())}None
            };
            let(data,hashes)=deploy::load_verified(station,&sources,60,issue,policy)?;
            let model=ForecastBundle::train_chronological(&data,issue,recipe,years)?;
            if let Some(parent)=path.parent().filter(|p|!p.as_os_str().is_empty()){std::fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
            model.artifact(hashes)?.save(&path)?;
            let loaded=Artifact::load(&path)?;
            if loaded.station!=data.station||loaded.cutoff!=issue||loaded.recipe!=recipe{return Err("artifact readback mismatch".into())}
            ForecastBundle::from_artifact(loaded)?;
            println!("saved,{},{}",path.display(),weather::format_minute(issue)?);
        }
        "forecast"=>{
            let hours=get("--hours").unwrap_or("24").parse::<usize>().map_err(|_|"invalid hours")?;
            if !(1..=24).contains(&hours){return Err("hours must be 1..24".into())}
            let a=Artifact::load(&path)?;
            if a.station!=station||a.recipe!=recipe{return Err("artifact station/model mismatch".into())}
            let model=ForecastBundle::from_artifact(a)?;
            let first=issue.checked_sub(71*60).filter(|&x|x>=60).ok_or("insufficient history")?;
            let(data,hashes)=deploy::load_verified(station,&sources,first,issue,policy)?;
            let p=model.forecast(&data,issue)?;
            let mut csv=String::from("station,origin_utc,valid_utc,lead_hours,point_c,lo50_c,hi50_c,lo80_c,hi80_c,lo90_c,hi90_c,lo95_c,hi95_c\n");
            for h in 0..hours{
                use std::fmt::Write as _;
                write!(&mut csv,"{icao},{},{},{},{}",weather::format_minute(issue)?,weather::format_minute(issue+60*(h+1)as i64)?,h+1,p.point[h]).map_err(|e|e.to_string())?;
                if let Some(s)=&p.support{for c in [0.5,0.8,0.9,0.95]{let(l,u)=P13::band(&s[h],c)?;write!(&mut csv,",{l},{u}").map_err(|e|e.to_string())?;}}else{csv.push_str(",,,,,,,,");}
                csv.push('\n');
            }
            eprintln!("forecast-source-count,{}",hashes.len());
            if let Some(p)=get("--output"){let mut f=std::fs::OpenOptions::new().write(true).create_new(true).open(p).map_err(|e|e.to_string())?;f.write_all(csv.as_bytes()).map_err(|e|e.to_string())?;f.sync_all().map_err(|e|e.to_string())?;}
            else{print!("{csv}");}
        }
        _=>unreachable!()
    }Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn command_validation(){
        assert!(timestamp("2025-01-01T00:00").is_ok());
        assert!(timestamp("2025-01-01T00:01").is_err());
        assert!(options(&["--station".into(),"KLAX".into(),"--station".into(),"KSEA".into()],&["--station"]).is_err());
        assert!(run(&["forecast".into(),"--station".into(),"FAKE".into()]).is_err());
    }
}
