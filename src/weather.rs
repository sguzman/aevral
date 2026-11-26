//! UTC routine-report ingestion and causal forecast windows.
use std::{collections::{BTreeMap,BTreeSet},io::BufRead};
pub const WINDOW:usize=24;
pub const HORIZON:usize=24;
fn leap(y:i64)->bool{y%4==0&&(y%100!=0||y%400==0)}
pub fn minute(s:&str)->Result<i64,String>{
    let s=s.trim();
    if s.len()!=16||!s.is_ascii()||&s[4..5]!="-"||&s[7..8]!="-"||&s[10..11]!=" "||&s[13..14]!=":"{return Err(format!("bad UTC minute {s}"))}
    let p=|a,b|s[a..b].parse::<i64>().map_err(|_|format!("bad UTC minute {s}"));
    let(y,m,d,h,n)=(p(0,4)?,p(5,7)?,p(8,10)?,p(11,13)?,p(14,16)?);
    if !(2000..=2026).contains(&y)||!(1..=12).contains(&m)||!(0..24).contains(&h)||!(0..60).contains(&n){return Err(format!("invalid UTC minute {s}"))}
    let lengths=[31,if leap(y){29}else{28},31,30,31,30,31,31,30,31,30,31];
    if d<1||d>lengths[(m-1)as usize]{return Err(format!("invalid day {s}"))}
    Ok(((2000..y).map(|y|if leap(y){366}else{365}).sum::<i64>()+lengths[..(m-1)as usize].iter().sum::<i64>()+d-1)*1440+h*60+n)
}
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct Report{pub minute:i64,pub celsius:Option<f64>}
#[derive(Default,Debug)]
pub struct Archive{reports:BTreeMap<i64,Option<f64>>,pub duplicates:usize,pub conflicts:BTreeSet<i64>}
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
                header=Some((c.len(),pos("station")?,pos("valid")?,pos("tmpf")?));continue
            }
            let(w,s,t,f)=header.unwrap();
            if c.len()!=w||c[s]!="SEA"{return Err(format!("invalid row {}",i+1))}
            let t=minute(c[t])?;
            let v=c[f].trim();
            let value=if v=="M"||v.is_empty(){None}else{
                let f=v.parse::<f64>().map_err(|_|format!("bad temperature row {}",i+1))?;
                if !f.is_finite()||!(-130.0..=150.0).contains(&f){return Err(format!("temperature out of range row {}",i+1))}
                Some((f-32.0)*5.0/9.0)
            };
            if self.conflicts.contains(&t){continue}
            if let Some(old)=self.reports.get(&t){
                if *old!=value{
                    if quarantine_test&&t>=minute("2023-01-01 00:00")?{self.quarantine(t);continue}
                    return Err(format!("conflicting report at minute {t}"))
                }
                self.duplicates+=1;
            }else{self.reports.insert(t,value);}
        }
        if header.is_none(){return Err("missing CSV header".into())}Ok(())
    }
    pub fn quarantine(&mut self,t:i64){self.conflicts.insert(t);self.reports.insert(t,None);}
    pub fn len(&self)->usize{self.reports.len()}
    pub fn is_empty(&self)->bool{self.reports.is_empty()}
    /// Bins (end-60,end]; choose latest report even if missing temperature.
    pub fn hourly(&self,first:i64,last:i64)->Vec<Option<Report>>{
        assert!(first%60==0&&last%60==0&&first<=last);
        (first..=last).step_by(60).map(|end|self.reports.range((end-59)..=end).next_back().map(|(&minute,&celsius)|Report{minute,celsius})).collect()
    }
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Split{Train,Validation,Test}
pub fn split_for(origin:i64)->Option<Split>{
    let first=origin+60;let last=origin+60*HORIZON as i64;
    let v=minute("2019-01-01 00:00").unwrap();
    let t=minute("2023-01-01 00:00").unwrap();
    let stop=minute("2026-01-01 00:00").unwrap();
    if first<0||last>=stop{None}else if last<v{Some(Split::Train)}else if first>=v&&last<t{Some(Split::Validation)}else if first>=t{Some(Split::Test)}else{None}
}
#[derive(Debug)]
pub struct Example{pub origin:i64,pub split:Split,pub x:[f64;WINDOW],pub y:[f64;HORIZON]}
#[derive(Default,Debug)]
pub struct Audit{pub bins:usize,pub absent:usize,pub missing_temperature:usize,pub eligible:usize,pub excluded_boundary:usize,pub excluded_missing:usize,pub train:usize,pub validation:usize,pub test:usize}
pub fn examples(bins:&[Option<Report>],first:i64)->(Vec<Example>,Audit){
    assert_eq!(first%60,0);
    let mut a=Audit{bins:bins.len(),absent:bins.iter().filter(|r|r.is_none()).count(),missing_temperature:bins.iter().filter(|r|matches!(r,Some(Report{celsius:None,..}))).count(),..Audit::default()};
    let mut rows=Vec::new();
    for i in (WINDOW-1)..bins.len().saturating_sub(HORIZON){
        let origin=first+i as i64*60;
        let Some(split)=split_for(origin)else{a.excluded_boundary+=1;continue};
        let values:Option<Vec<_>>=bins[i+1-WINDOW..=i+HORIZON].iter().map(|r|r.and_then(|v|v.celsius)).collect();
        let Some(v)=values else{a.excluded_missing+=1;continue};
        rows.push(Example{origin,split,x:v[..WINDOW].try_into().unwrap(),y:v[WINDOW..].try_into().unwrap()});
        a.eligible+=1;
        match split{Split::Train=>a.train+=1,Split::Validation=>a.validation+=1,Split::Test=>a.test+=1}
    }(rows,a)
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn dates_and_split(){
        assert_eq!(minute("2000-01-01 00:00").unwrap(),0);
        assert_eq!(minute("2000-03-01 00:00").unwrap(),60*24*60);
        assert!(minute("2001-02-29 00:00").is_err());
        let v=minute("2019-01-01 00:00").unwrap();
        assert_eq!(split_for(v-60),Some(Split::Validation));
        assert_eq!(split_for(v-120),None);
        assert_eq!(split_for(v-25*60),Some(Split::Train));
    }
    #[test]fn reports(){
        let mut a=Archive::default();
        a.ingest("station,valid,tmpf\nSEA,2000-01-01 00:59,68\nSEA,2000-01-01 01:00,M\nSEA,2000-01-01 01:01,50\nSEA,2000-01-01 01:01,50\n".as_bytes()).unwrap();
        assert_eq!(a.duplicates,1);
        let b=a.hourly(60,180);
        assert_eq!(b[0],Some(Report{minute:60,celsius:None}));
        assert_eq!(b[1],Some(Report{minute:61,celsius:Some(10.0)}));
        assert_eq!(b[2],None);
        assert!(a.ingest("station,valid,tmpf\nSEA,2000-01-01 01:01,51\n".as_bytes()).is_err());
    }
    #[test]fn windows(){
        let mut b:Vec<_>=(0..50).map(|i|Some(Report{minute:i*60,celsius:Some(i as f64)})).collect();
        let(r,a)=examples(&b,0);
        assert_eq!(a.eligible,3);assert_eq!(r[0].origin,23*60);assert_eq!(r[0].x[23],23.0);assert_eq!(r[0].y[0],24.0);assert_eq!(r[0].y[23],47.0);
        b[24]=None;let(_,a)=examples(&b,0);assert_eq!(a.eligible,0);assert_eq!(a.excluded_missing,3);
    }
}
