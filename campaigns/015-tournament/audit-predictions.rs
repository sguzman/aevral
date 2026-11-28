//! Independent C015 point forecast audit; std only, no model or scoring-engine imports.
use std::{collections::BTreeMap, fs::File, io::{BufRead, BufReader, BufWriter, Write}, path::Path};

const TARGET_HEADER: &str = "origin,lead,target_c";
const PRED_HEADER: &str = "fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c";
const TOL: f64 = 1e-7;

#[derive(Clone, Copy, Debug, Default)]
struct Metric { n: usize, abs: f64, sq: f64 }
impl Metric {
    fn add(&mut self, y: f64, p: f64) -> Result<(), String> {
        let d = p-y;
        self.n += 1; self.abs += d.abs(); self.sq += d*d;
        if !self.abs.is_finite() || !self.sq.is_finite() { return Err("nonfinite error sum".into()); }
        Ok(())
    }
    fn mae(self) -> f64 { self.abs / self.n as f64 }
    fn rmse(self) -> f64 { (self.sq / self.n as f64).sqrt() }
}
#[derive(Default)]
struct Metrics { all: Metric, lead: [Metric; 24], month: [Metric; 12], season: [Metric; 4] }
fn number(s: &str) -> Result<f64, String> {
    let x=s.parse::<f64>().map_err(|_| format!("invalid number: {s}"))?;
    if !x.is_finite() { return Err("nonfinite number".into()); }
    Ok(x)
}
fn integer<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("invalid integer: {s}"))
}
fn leap(y: i64) -> bool { y%4==0 && (y%100!=0 || y%400==0) }
fn year_start(y: i64) -> i64 { (2000..y).map(|k| if leap(k){366}else{365}).sum::<i64>()*1440 }
fn month(y: i64, origin: i64) -> Result<usize, String> {
    let start=year_start(y);
    let end=if y==2026 { start+304*1440 } else { year_start(y+1) };
    if origin+60<start || origin+1440>=end || origin%60!=0 { return Err("origin outside frozen assessment window".into()); }
    let mut day=(origin+60-start)/1440;
    for (i,n) in [31,if leap(y){29}else{28},31,30,31,30,31,31,30,31,30,31].iter().enumerate() {
        if day<*n { return Ok(i); } day-=n;
    }
    Err("invalid UTC month".into())
}
fn season(m:usize)->usize { match m { 11|0|1=>0,2..=4=>1,5..=7=>2,_=>3 } }
fn scan(target:impl BufRead,pred:impl BufRead,y:i64,recipe:&str)->Result<Metrics,String>{
    let mut t=target.lines();let mut p=pred.lines();
    let next=|x:Option<std::io::Result<String>>|x.ok_or("missing row".to_string())?.map_err(|e|e.to_string());
    if next(t.next())?!=TARGET_HEADER || next(p.next())?!=PRED_HEADER {return Err("bad CSV header".into());}
    let mut m=Metrics::default();let mut prior=None;let mut lead=1usize;
    loop {
        let (a,b)=match(t.next(),p.next()) {
            (None,None)=>break,
            (Some(a),Some(b))=>(a.map_err(|e|e.to_string())?,b.map_err(|e|e.to_string())?),
            _=>return Err("prediction/target length mismatch".into()),
        };
        let a:Vec<_>=a.split(',').collect();let b:Vec<_>=b.split(',').collect();
        if a.len()!=3 || b.len()!=6 || integer::<i64>(b[0])?!=y || b[1]!=recipe {return Err("prediction schema or identity mismatch".into());}
        let origin=integer::<i64>(a[0])?;
        if origin!=integer::<i64>(b[2])? || integer::<usize>(a[1])?!=lead || integer::<usize>(b[3])?!=lead {return Err("origin/lead mismatch".into());}
        if lead==1 {
            if prior.is_some_and(|v|origin<=v){return Err("duplicate or unordered origin".into());}
            prior=Some(origin);
        }else if prior!=Some(origin){return Err("origin changed within trajectory".into());}
        let target=number(a[2])?;
        if target!=number(b[4])? {return Err("target differs from frozen cohort".into());}
        let pred=number(b[5])?;
        let mo=month(y,origin)?;
        m.all.add(target,pred)?;m.lead[lead-1].add(target,pred)?;m.month[mo].add(target,pred)?;m.season[season(mo)].add(target,pred)?;
        lead=lead%24+1;
    }
    if lead!=1 || m.all.n==0 {return Err("empty or incomplete trajectory".into());}
    Ok(m)
}
fn expected(m:&Metrics)->BTreeMap<(&'static str,usize),Metric>{
    let mut rows=BTreeMap::new();rows.insert(("all",0),m.all);
    for (kind,ms) in [("lead",&m.lead[..]),("month",&m.month[..])] {
        for (i,&v) in ms.iter().enumerate() {if v.n>0 {rows.insert((kind,i+1),v);}}
    }
    rows
}
fn validate(log:impl BufRead,y:i64,recipe:&str,prefix:&str,m:&Metrics)->Result<(),String>{
    let mut rows=expected(m);
    let tags=[format!("{prefix}score"),format!("{prefix}lead"),format!("{prefix}month")];
    for l in log.lines() {
        let l=l.map_err(|e|e.to_string())?;let p:Vec<_>=l.split(',').collect();
        if p.len()<3 || p[1]!=y.to_string() || p[2]!=recipe {continue;}
        let (key,n,at)=if p[0]==tags[0] {
            if p.len()<6 {return Err("malformed score".into());}
            (("all",0),Some(integer::<usize>(p[3])?),4)
        }else if p[0]==tags[1] {
            if p.len()!=6 {return Err("malformed lead".into());}
            (("lead",integer::<usize>(p[3])?),None,4)
        }else if p[0]==tags[2] {
            if p.len()!=7 {return Err("malformed month".into());}
            (("month",integer::<usize>(p[3])?),Some(integer::<usize>(p[4])?),5)
        }else{continue;};
        let v=rows.remove(&key).ok_or("duplicate or unexpected metric")?;
        if n.is_some_and(|n|n!=v.n/24) || (number(p[at])?-v.mae()).abs()>TOL || (number(p[at+1])?-v.rmse()).abs()>TOL {
            return Err(format!("metric mismatch {key:?}"));
        }
    }
    if !rows.is_empty(){return Err(format!("missing producer metrics: {:?}",rows.keys()));}
    Ok(())
}
fn report(out:impl Write,m:&Metrics)->Result<(),String>{
    let mut out=BufWriter::new(out);
    writeln!(out,"group,index,targets,absolute_sum,squared_sum,mae_c,rmse_c").map_err(|e|e.to_string())?;
    let mut rows=expected(m);
    for(i,&v)in m.season.iter().enumerate(){if v.n>0{rows.insert(("season",i+1),v);}}
    for ((kind,i),v) in rows {
        writeln!(out,"{kind},{i},{},{:.17e},{:.17e},{:.12},{:.12}",v.n,v.abs,v.sq,v.mae(),v.rmse()).map_err(|e|e.to_string())?;
    }
    out.flush().map_err(|e|e.to_string())
}
fn open(p:&str)->Result<BufReader<File>,String>{File::open(Path::new(p)).map(BufReader::new).map_err(|e|format!("{p}: {e}"))}
fn run()->Result<(),String>{
    let a:Vec<_>=std::env::args().collect();
    if a.len()!=8{return Err("usage: audit TARGET PRED LOG YEAR RECIPE matched|natural OUTPUT".into());}
    let y=integer::<i64>(&a[4])?;
    if !(2023..=2026).contains(&y){return Err("outside C015".into());}
    let prefix=match a[6].as_str(){"matched"=>"","natural"=>"natural-",_=>return Err("bad cohort".into())};
    let m=scan(open(&a[1])?,open(&a[2])?,y,&a[5])?;
    validate(open(&a[3])?,y,&a[5],prefix,&m)?;
    report(File::create(&a[7]).map_err(|e|e.to_string())?,&m)?;
    println!("c015-independent-audit,{y},{},{},{},pass",a[5],a[6],m.all.n);
    Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("{e}");std::process::exit(1)}}
#[cfg(test)]
mod tests{
    use super::*;
    fn fixture()->(String,String,String){
        let mut t=format!("{TARGET_HEADER}\n");let mut p=format!("{PRED_HEADER}\n");let mut l="score,2026,m,1,1.0,1.0\nmonth,2026,m,1,1,1.0,1.0\n".to_string();
        let o=year_start(2026)-60;
        for i in 1..=24{t+=&format!("{o},{i},2\n");p+=&format!("2026,m,{o},{i},2,3\n");l+=&format!("lead,2026,m,{i},1,1\n");}
        (t,p,l)
    }
    #[test]fn parity(){let(t,p,l)=fixture();let m=scan(t.as_bytes(),p.as_bytes(),2026,"m").unwrap();validate(l.as_bytes(),2026,"m","",&m).unwrap();assert_eq!(m.all.n,24);assert_eq!(m.season[0].n,24);}
    #[test]fn rejects_bad_rows(){let(t,p,_)=fixture();for q in [p.replace(",2,3\n",",2,NaN\n"),p.replace(",2,3\n",",4,3\n"),p.replacen(",2,2,3\n",",1,2,3\n",1),p.replace("2026,m,","2025,m,")]{assert!(scan(t.as_bytes(),q.as_bytes(),2026,"m").is_err());}assert!(scan(t.as_bytes(),p.trim_end().rsplit_once('\n').unwrap().0.as_bytes(),2026,"m").is_err());}
    #[test]fn rejects_bad_metrics(){let(t,p,l)=fixture();let m=scan(t.as_bytes(),p.as_bytes(),2026,"m").unwrap();assert!(validate(l.replace("1.0,1.0","0.9,1.0").as_bytes(),2026,"m","",&m).is_err());assert!(validate(format!("{l}month,2026,m,11,0,0,0\n").as_bytes(),2026,"m","",&m).is_err());}
    #[test]fn calendar(){assert_eq!(month(2024,year_start(2024)+59*1440-60).unwrap(),1);assert_eq!(month(2024,year_start(2024)+60*1440-60).unwrap(),2);let end=year_start(2026)+304*1440;assert_eq!(month(2026,end-1500).unwrap(),9);assert!(month(2026,end-1440).is_err());}
}
