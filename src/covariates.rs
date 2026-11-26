//! Origin-time weather covariates from the same routine report as temperature.
use std::{collections::{BTreeMap,BTreeSet},io::BufRead};
use crate::weather::{minute,Report,Split,split_for};
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct Weather {
    pub minute:i64,
    pub temperature:Option<f64>,
    pub dew_c:Option<f64>,
    pub pressure_hpa:Option<f64>,
    pub wind_ms:Option<[f64;2]>,
}
#[derive(Default)]
pub struct Archive {
    reports:BTreeMap<i64,Weather>,
    pub invalid:[usize;4],
    pub conflicts:BTreeSet<i64>,
}
fn numeric(s:&str,lo:f64,hi:f64,invalid:&mut usize)->Option<f64>{
    let s=s.trim();
    if s.is_empty()||s=="M"{return None}
    match s.parse::<f64>(){
        Ok(v) if v.is_finite()&&(lo..=hi).contains(&v)=>Some(v),
        _=>{*invalid+=1;None}
    }
}
impl Archive{
    pub fn ingest(&mut self,reader:impl BufRead)->Result<(),String>{self.ingest_policy(reader,false)}
    pub fn ingest_policy(&mut self,reader:impl BufRead,quarantine_test:bool)->Result<(),String>{
        let mut header=None;
        for(i,line)in reader.lines().enumerate(){
            let line=line.map_err(|e|e.to_string())?;
            let line=line.trim();
            if line.is_empty()||line.starts_with('#'){continue}
            let c:Vec<_>=line.split(',').collect();
            if header.is_none(){
                let pos=|key|c.iter().position(|v|*v==key).ok_or_else(||format!("missing column {key}"));
                header=Some((c.len(),pos("station")?,pos("valid")?,pos("tmpf")?,pos("dwpf")?,pos("alti")?,pos("drct")?,pos("sknt")?));continue
            }
            let(w,s,t,f,d,p,dir,speed)=header.unwrap();
            if c.len()!=w||c[s]!="SEA"{return Err(format!("bad row {}",i+1))}
            let t=minute(c[t])?;
            let temperature=if c[f].trim()=="M"||c[f].trim().is_empty(){None}else{
                let v=c[f].trim().parse::<f64>().map_err(|_|"bad temperature")?;
                if !v.is_finite()||!(-130.0..=150.0).contains(&v){return Err("invalid temperature".into())}
                Some((v-32.0)*5.0/9.0)
            };
            let dew_c=numeric(c[d],-130.,150.,&mut self.invalid[0]).map(|v|(v-32.)*5./9.);
            let pressure_hpa=numeric(c[p],25.,34.,&mut self.invalid[1]).map(|v|v*33.8638866667);
            let angle=numeric(c[dir],0.,360.,&mut self.invalid[2]);
            let speed=numeric(c[speed],0.,200.,&mut self.invalid[3]);
            let wind_ms=match(speed,angle){
                (Some(0.),_)=>Some([0.,0.]),
                (Some(v),Some(a))=>{let a=a.to_radians();let v=v*0.514444444444;Some([-v*a.sin(),-v*a.cos()])},
                _=>None
            };
            let report=Weather{minute:t,temperature,dew_c,pressure_hpa,wind_ms};
            if self.conflicts.contains(&t){continue}
            if let Some(old)=self.reports.get(&t){if *old!=report{
                if quarantine_test&&t>=minute("2023-01-01 00:00")?{self.quarantine(t);continue}
                return Err(format!("conflicting covariates at {t}"))
            }}
            else{self.reports.insert(t,report);}
        }
        if header.is_none(){return Err("missing CSV header".into())}Ok(())
    }
    pub fn quarantine(&mut self,t:i64){
        self.conflicts.insert(t);
        self.reports.insert(t,Weather{minute:t,temperature:None,dew_c:None,pressure_hpa:None,wind_ms:None});
    }
    pub fn hourly(&self,first:i64,last:i64)->Vec<Option<Weather>>{
        assert!(first%60==0&&last%60==0&&first<=last);
        (first..=last).step_by(60).map(|end|self.reports.range((end-59)..=end).next_back().map(|(_,v)|*v)).collect()
    }
}
/// Compare coverage of covariates at issue time among complete temperature windows.
/// Counts only validation/training cases; no future covariate is required.
pub fn audit(bins:&[Option<Weather>],first:i64,reference:&[Option<Report>])->Result<(),String>{
    if bins.len()!=reference.len(){return Err("bin count mismatch".into())}
    for (a,b) in bins.iter().zip(reference){
        if a.map(|v|(v.minute,v.temperature))!=b.map(|v|(v.minute,v.celsius)){return Err("temperature alignment mismatch".into())}
    }
    let mut counts=[[0usize;5];2];
    for i in 71..bins.len().saturating_sub(24){
        let Some(split)=split_for(first+i as i64*60)else{continue};
        let k=match split{Split::Train=>0,Split::Validation=>1,Split::Test=>continue};
        if !bins[i-71..=i+24].iter().all(|r|r.is_some_and(|v|v.temperature.is_some())){continue}
        let r=bins[i].unwrap();
        let flags=[r.dew_c.is_some(),r.pressure_hpa.is_some(),r.wind_ms.is_some()];
        counts[k][0]+=1;
        for(j,flag)in flags.iter().enumerate(){counts[k][j+1]+=usize::from(*flag)}
        counts[k][4]+=usize::from(flags.iter().all(|v|*v));
    }
    println!("split,temperature,dew_point,pressure,wind,all");
    for (name,c) in ["train","validation"].iter().zip(counts){
        println!("{name},{},{},{},{},{}",c[0],c[1],c[2],c[3],c[4]);
        if c[0]==0||c[4]==0{return Err("no joint complete cases".into())}
    }Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn latest_and_wind(){
        let mut a=Archive::default();
        a.ingest("station,valid,tmpf,dwpf,alti,drct,sknt\nSEA,2000-01-01 00:01,50,41,30,90,10\nSEA,2000-01-01 00:59,50,M,30,M,0\nSEA,2000-01-01 01:59,50,41,30,90,10\n".as_bytes()).unwrap();
        let b=a.hourly(60,120);let r=b[0].unwrap();
        assert_eq!(r.minute,59);assert_eq!(r.dew_c,None);assert_eq!(r.wind_ms,Some([0.,0.]));
        let r=b[1].unwrap();assert_eq!(r.dew_c,Some(5.));
        let wind=r.wind_ms.unwrap();assert!((wind[0]+5.14444444444).abs()<1e-9&&wind[1].abs()<1e-9);
        assert!((r.pressure_hpa.unwrap()-1015.916600001).abs()<1e-8);
    }
    #[test]fn invalid_values_remain_missing(){
        let mut a=Archive::default();
        a.ingest("station,valid,tmpf,dwpf,alti,drct,sknt\nSEA,2000-01-01 00:59,50,NaN,0,999,10\n".as_bytes()).unwrap();
        let r=a.hourly(60,60)[0].unwrap();
        assert_eq!(a.invalid,[1,1,1,0]);assert_eq!(r.dew_c,None);assert_eq!(r.pressure_hpa,None);assert_eq!(r.wind_ms,None);
    }
}

#[cfg(test)]
mod quarantine_tests{
    use super::*;
    #[test]fn ambiguous_test_report_never_recovers_or_falls_back(){
        let s="station,valid,tmpf,dwpf,alti,drct,sknt\nSEA,2024-01-01 00:01,50,41,30,90,10\nSEA,2024-01-01 00:59,50,41,30,90,10\nSEA,2024-01-01 00:59,51,41,30,90,10\nSEA,2024-01-01 00:59,50,41,30,90,10\n";
        let mut a=Archive::default();let mut b=crate::weather::Archive::default();
        a.ingest_policy(s.as_bytes(),true).unwrap();b.ingest_policy(s.as_bytes(),true).unwrap();
        let end=minute("2024-01-01 01:00").unwrap();
        assert_eq!(a.conflicts.len(),1);assert_eq!(b.conflicts.len(),1);
        let r=a.hourly(end,end)[0].unwrap();
        assert_eq!(r.minute,end-1);assert_eq!(r.temperature,None);assert_eq!(r.wind_ms,None);
        assert_eq!(b.hourly(end,end)[0].unwrap().celsius,None);
        let old=s.replace("2024","2018");
        assert!(Archive::default().ingest_policy(old.as_bytes(),true).is_err());
        assert!(crate::weather::Archive::default().ingest_policy(old.as_bytes(),true).is_err());
    }
}
