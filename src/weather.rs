//! UTC routine-report ingestion and causal forecast windows.
use std::{collections::{BTreeMap,BTreeSet},io::BufRead};
/// Exact source station identifier. ICAO-to-provider aliases belong in the
/// source adapter, never in the parser; mixed station rows are rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Station(String);
impl Station {
    pub fn new(id: &str) -> Result<Self, String> {
        if !(3..=8).contains(&id.len()) || !id.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()) {
            return Err("invalid source station identifier".into());
        }
        Ok(Self(id.to_owned()))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}
impl Default for Station {
    fn default() -> Self { Self("SEA".into()) }
}
pub const WINDOW:usize=24;
pub const HORIZON:usize=24;
pub(crate) fn leap(y:i64)->bool{y%4==0&&(y%100!=0||y%400==0)}
pub const MIN_YEAR:i64=2000;
pub const MAX_YEAR:i64=9999;
// Gregorian day count relative to 2000-01-01, without an O(year) loop.
fn days_before(y:i64)->i64{
    let n=y-1;
    let base=MIN_YEAR-1;
    365*(y-MIN_YEAR)+n/4-base/4-n/100+base/100+n/400-base/400
}
pub fn minute(s:&str)->Result<i64,String>{
    let s=s.trim();
    if s.len()!=16||!s.is_ascii()||&s[4..5]!="-"||&s[7..8]!="-"||&s[10..11]!=" "||&s[13..14]!=":"{return Err(format!("bad UTC minute {s}"))}
    let p=|a,b|s[a..b].parse::<i64>().map_err(|_|format!("bad UTC minute {s}"));
    let(y,m,d,h,n)=(p(0,4)?,p(5,7)?,p(8,10)?,p(11,13)?,p(14,16)?);
    if !(MIN_YEAR..=MAX_YEAR).contains(&y)||!(1..=12).contains(&m)||!(0..24).contains(&h)||!(0..60).contains(&n){return Err(format!("invalid UTC minute {s}"))}
    let lengths=[31,if leap(y){29}else{28},31,30,31,30,31,31,30,31,30,31];
    if d<1||d>lengths[(m-1)as usize]{return Err(format!("invalid day {s}"))}
    Ok((days_before(y)+lengths[..(m-1)as usize].iter().sum::<i64>()+d-1)*1440+h*60+n)
}
/// Exclusive upper bound of supported UTC minutes.
pub fn minute_limit()->i64{days_before(MAX_YEAR+1)*1440}
/// Resolve year and UTC minute of year in O(log years).
pub fn year_start(t:i64)->Result<(i64,i64),String>{
    if !(0..minute_limit()).contains(&t){return Err("UTC minute outside supported range".into())}
    let mut lo=MIN_YEAR;let mut hi=MAX_YEAR+1;
    while hi-lo>1{let mid=lo+(hi-lo)/2;if days_before(mid)*1440<=t{lo=mid}else{hi=mid}}
    Ok((lo,days_before(lo)*1440))
}
pub fn format_minute(t:i64)->Result<String,String>{
    let(y,start)=year_start(t)?;
    let mut day=(t-start)/1440;
    let lengths=[31,if leap(y){29}else{28},31,30,31,30,31,31,30,31,30,31];
    let mut m=0;while day>=lengths[m]{day-=lengths[m];m+=1}
    let n=t.rem_euclid(1440);
    Ok(format!("{y:04}-{:02}-{:02} {:02}:{:02}",m+1,day+1,n/60,n%60))
}
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct Report{pub minute:i64,pub celsius:Option<f64>}
#[derive(Default,Debug)]
pub struct Archive{station:Station,reports:BTreeMap<i64,Option<f64>>,pub duplicates:usize,pub conflicts:BTreeSet<i64>}
impl Archive{
    pub fn for_station(station: Station) -> Self { Self { station, ..Self::default() } }
    pub fn station(&self) -> &Station { &self.station }
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
            if c.len()!=w||c[s]!=self.station.as_str(){return Err(format!("invalid row {}",i+1))}
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

#[cfg(test)]
mod station_tests {
    use super::*;
    #[test]
    fn exact_station_identity_and_default_compatibility() {
        let sea = "station,valid,tmpf\nSEA,2024-01-01 00:59,68\n";
        let lax = sea.replace("SEA,", "LAX,");
        let mut a = Archive::default();
        let mut b = Archive::for_station(Station::new("LAX").unwrap());
        a.ingest(sea.as_bytes()).unwrap();
        b.ingest(lax.as_bytes()).unwrap();
        let t = minute("2024-01-01 01:00").unwrap();
        assert_eq!(a.hourly(t,t), b.hourly(t,t));
        assert!(b.ingest(sea.as_bytes()).is_err());
        assert!(a.ingest(lax.as_bytes()).is_err());
        assert!(b.ingest(lax.replace("LAX,", "KLAX,").as_bytes()).is_err());
        assert_eq!(b.station().as_str(), "LAX");
        assert!(Station::new(" lax").is_err());
        assert!(Station::new("../LAX").is_err());
    }
}

#[cfg(test)]
mod date_range_tests{
    use super::*;
    #[test]
    fn extended_gregorian_roundtrips(){
        for s in ["2000-01-01 00:00","2026-12-31 23:59","2027-01-01 00:00","2028-02-29 12:34","2100-03-01 00:00","2400-02-29 23:59","9999-12-31 23:59"]{
            assert_eq!(format_minute(minute(s).unwrap()).unwrap(),s);
        }
        assert!(minute("2100-02-29 00:00").is_err());
        assert!(format_minute(minute_limit()).is_err());
        assert!(format_minute(-1).is_err());
        for y in 2000..=2026{
            let legacy=(2000..y).map(|v|if leap(v){366}else{365}).sum::<i64>()*1440;
            assert_eq!(minute(&format!("{y}-01-01 00:00")).unwrap(),legacy);
        }
    }
}
