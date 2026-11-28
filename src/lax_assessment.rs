//! Fixed C018 2025 matched assessment, with saved model reload before scoring.
use aevral::{weather::{self,Station},features::tournament::{deploy::{self,ForecastBundle},artifact::{Recipe,Artifact},station::ConflictPolicy,p13::{P13,SUPPORT}}};
const Y:usize=24;
use std::{path::Path,fs::{self,File},io::{BufWriter,Write}};
#[derive(Default)]
struct Metric{n:usize,ae:f64,se:f64,crps:f64,covered:[usize;4],width:[f64;4]}
impl Metric{
    fn add(&mut self,y:f64,p:f64,s:Option<&[f64;SUPPORT]>)->Result<(),String>{
        if !y.is_finite()||!p.is_finite(){return Err("nonfinite score".into())}
        let e=p-y;self.n+=1;self.ae+=e.abs();self.se+=e*e;
        if let Some(v)=s{
            let n=SUPPORT as f64;
            let a=v.iter().map(|x|(x-y).abs()).sum::<f64>()/n;
            let b=v.iter().enumerate().map(|(i,x)|(2.*(i+1)as f64-n-1.)*(x-y)).sum::<f64>()/(n*n);
            self.crps+=(a-b).max(0.);
            for (i,c) in [0.5,0.8,0.9,0.95].into_iter().enumerate(){
                let(l,u)=P13::band(v,c)?;self.covered[i]+=usize::from(l<=y&&y<=u);self.width[i]+=u-l;
            }
        }Ok(())
    }
    fn write(&self,out:&mut impl Write,id:&str,h:usize,prob:bool)->Result<(),String>{
        if self.n==0{return Err("empty metric".into())}
        let n=self.n as f64;
        write!(out,"{id},{h},{},{:.12},{:.12}",self.n,self.ae/n,(self.se/n).sqrt()).map_err(|e|e.to_string())?;
        if prob{
            write!(out,",{:.12}",self.crps/n).map_err(|e|e.to_string())?;
            for i in 0..4{write!(out,",{:.12},{:.12}",self.covered[i]as f64/n,self.width[i]/n).map_err(|e|e.to_string())?;}
        }else{write!(out,",,,,,,,,,").map_err(|e|e.to_string())?;}
        writeln!(out).map_err(|e|e.to_string())
    }
}
pub fn run(lax:&Path,sea:&Path,out:&Path,id:&str)->Result<(),String>{
    let start=weather::minute("2025-01-01 00:00")?;let end=weather::minute("2026-01-01 00:00")?;
    if !matches!(id,"transfer"|"mlp"|"p13"){return Err("unknown frozen assessment model".into())}
    fs::create_dir(out).map_err(|e|e.to_string())?;
    let paths=(2000..=2025).map(|y|lax.join(format!("LAX-{y}-01-01-{}-01-01.csv",y+1))).collect::<Vec<_>>();
    let(data,lax_hashes)=deploy::load_verified(Station::new("LAX")?,&paths,60,end-60,ConflictPolicy::Strict)?;
    let c=deploy::coverage(&data,start,end)?;
    if c.matched==0{return Err("empty matched LAX cohort".into())}
    println!("lax-cohort,{},{},{}",c.possible,c.temperature_complete,c.matched);
    let model=if id=="transfer"{
        let paths=(2000..=2024).map(|y|sea.join(format!("SEA-{y}.csv"))).collect::<Vec<_>>();
        let(train,hashes)=deploy::load_verified(Station::new("SEA")?,&paths,60,start-60,ConflictPolicy::LegacyExposed)?;
        let m=ForecastBundle::train_chronological(&train,start,Recipe::Mlp,None)?;
        m.artifact(hashes)?.save(&out.join("model.aev"))?;
        ForecastBundle::from_artifact(Artifact::load(&out.join("model.aev"))?)?
    }else{
        let recipe=if id=="p13"{Recipe::P13}else{Recipe::Mlp};
        let m=ForecastBundle::train_chronological(&data,start,recipe,if id=="p13"{Some((2022,2024))}else{None})?;
        m.artifact(lax_hashes)?.save(&out.join("model.aev"))?;
        ForecastBundle::from_artifact(Artifact::load(&out.join("model.aev"))?)?
    };
    let mut scores:[Metric;25]=std::array::from_fn(|_|Metric::default());
    let mut daily:[Metric;25]=std::array::from_fn(|_|Metric::default());
    let mut predictions=BufWriter::new(File::create(out.join("predictions.csv")).map_err(|e|e.to_string())?);
    writeln!(predictions,"origin,lead,target,daily,point,I,L,R,M").map_err(|e|e.to_string())?;
    let mut dist=if id=="p13"{Some(BufWriter::new(File::create(out.join("distribution.csv")).map_err(|e|e.to_string())?))}else{None};
    if let Some(d)=&mut dist{write!(d,"origin,lead").map_err(|e|e.to_string())?;for j in 0..SUPPORT{write!(d,",s{j}").map_err(|e|e.to_string())?;}writeln!(d).map_err(|e|e.to_string())?;}
    let n=model.evaluate_batch(&data,start,end,id=="transfer",|r,p,b,s|{
        for h in 0..Y{
            let y=r.target[h];let d=r.input.recent[h];let v=s.map(|x|&x[h]);
            scores[0].add(y,p[h],v)?;scores[h+1].add(y,p[h],v)?;
            daily[0].add(y,d,None)?;daily[h+1].add(y,d,None)?;
            write!(predictions,"{},{},{y},{d},{}",r.input.origin,h+1,p[h]).map_err(|e|e.to_string())?;
            if let Some(b)=b{for m in b{write!(predictions,",{}",m[h]).map_err(|e|e.to_string())?;}}else{write!(predictions,",,,,").map_err(|e|e.to_string())?;}
            writeln!(predictions).map_err(|e|e.to_string())?;
            if let (Some(d),Some(v))=(&mut dist,v){
                write!(d,"{},{}",r.input.origin,h+1).map_err(|e|e.to_string())?;
                for x in v{write!(d,",{x}").map_err(|e|e.to_string())?;}
                writeln!(d).map_err(|e|e.to_string())?;
            }
        }Ok(())
    })?;
    if n!=c.matched||scores[0].n!=n*Y{return Err("matched count mismatch".into())}
    predictions.flush().map_err(|e|e.to_string())?;if let Some(d)=&mut dist{d.flush().map_err(|e|e.to_string())?;}
    let mut metrics=BufWriter::new(File::create(out.join("metrics.csv")).map_err(|e|e.to_string())?);
    writeln!(metrics,"model,lead,n,mae,rmse,crps,coverage50,width50,coverage80,width80,coverage90,width90,coverage95,width95").map_err(|e|e.to_string())?;
    for h in 0..=Y{scores[h].write(&mut metrics,id,h,id=="p13")?;daily[h].write(&mut metrics,"daily",h,false)?;}
    metrics.flush().map_err(|e|e.to_string())?;
    println!("lax-assessment,{id},{n},pass");Ok(())
}
