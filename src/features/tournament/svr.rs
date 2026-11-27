//! C007 K07: official LIBSVM epsilon-SVR via pinned CLI, independent hourly leads.
use super::{load,score,Fold,D,Y};
use super::neighbors::{subset,Cached};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::{Path,PathBuf},process::Command,time::Instant};
const GUARD:f64=1400.;
const BUDGET:usize=2048;
#[derive(Clone,Copy)]
struct Recipe{length:f64,c:f64}
fn config(label:&str)->Result<Recipe,String>{let(length,c)=match label{
    "svr-s05-c1"=>(0.5,1.),"svr-s05-c10"=>(0.5,10.),"svr-s1-c1"=>(1.,1.),"svr-s1-c10"=>(1.,10.),_=>return Err("unknown SVR configuration".into())
};Ok(Recipe{length,c})}
impl Recipe{fn gamma(self)->f64{1./(2.*D as f64*self.length*self.length)}}
struct Scratch(PathBuf);
impl Scratch{fn new()->Result<Self,String>{let p=std::env::temp_dir().join(format!("aevral-svr-{}",std::process::id()));std::fs::create_dir(&p).map_err(|e|e.to_string())?;Ok(Self(p))}}
impl Drop for Scratch{fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
fn write_rows(path:&Path,x:&[[f64;D]],y:Option<&[f64]>)->Result<(),String>{
    if y.is_some_and(|v|v.len()!=x.len())||x.iter().flatten().any(|v|!v.is_finite())||y.is_some_and(|v|v.iter().any(|z|!z.is_finite())){return Err("invalid LIBSVM rows".into())}
    let mut w=BufWriter::new(std::fs::File::create(path).map_err(|e|e.to_string())?);
    for(i,r)in x.iter().enumerate(){write!(w,"{}",y.map_or(0.,|v|v[i])).map_err(|e|e.to_string())?;
        for(j,v)in r.iter().enumerate(){write!(w," {}:{v}",j+1).map_err(|e|e.to_string())?;}
        writeln!(w).map_err(|e|e.to_string())?;
    }w.flush().map_err(|e|e.to_string())
}
fn model_count(s:&str,n:usize,gamma:f64)->Result<usize,String>{
    let(header,body)=s.split_once("\nSV\n").ok_or("missing LIBSVM support vector section")?;
    let field=|k:&str|header.lines().find_map(|l|l.strip_prefix(k));
    if field("svm_type ")!=Some("epsilon_svr")||field("kernel_type ")!=Some("rbf"){return Err("unexpected LIBSVM model type".into())}
    let g=field("gamma ").and_then(|s|s.parse::<f64>().ok()).ok_or("missing LIBSVM gamma")?;
    if !g.is_finite()||(g-gamma).abs()>1e-10*gamma{return Err("LIBSVM gamma mismatch".into())}
    let total=field("total_sv ").and_then(|s|s.parse::<usize>().ok()).ok_or("missing LIBSVM support vector count")?;
    if total>n||body.lines().filter(|s|!s.trim().is_empty()).count()!=total{return Err("LIBSVM support vector count mismatch".into())}
    Ok(total)
}
fn predictions(s:&str,n:usize)->Result<Vec<f64>,String>{
    let v:Vec<f64>=s.lines().map(|l|l.trim().parse::<f64>().map_err(|_|"invalid LIBSVM prediction".to_string())).collect::<Result<_,_>>()?;
    if v.len()!=n||v.iter().any(|v|!v.is_finite()){return Err("LIBSVM prediction count/finite mismatch".into())}Ok(v)
}
fn command(bin:&str,args:&[String],dir:&Path,start:Instant)->Result<(),String>{
    let remaining=GUARD-start.elapsed().as_secs_f64();
    if remaining<2.{return Err("runtime guard".into())}
    let out=Command::new("timeout").arg("--kill-after=1s").arg(format!("{:.0}s",remaining.floor()-1.)).arg(bin).args(args).current_dir(dir).output().map_err(|e|format!("execute {bin}: {e}"))?;
    if !out.status.success(){if matches!(out.status.code(),Some(124|137)){return Err("runtime guard".into())}
        let message=String::from_utf8_lossy(&out.stderr);return Err(format!("{bin} failed: {} {}",out.status,message.chars().take(2000).collect::<String>()))}
    Ok(())
}
fn backend()->Result<(String,String),String>{
    let version=std::env::var("AEVRAL_LIBSVM_VERSION").map_err(|_|"use locked .#svr Nix shell")?;
    if version!="333"{return Err(format!("unexpected LIBSVM version {version}"))}
    let train=std::env::var("AEVRAL_SVM_TRAIN").map_err(|_|"missing pinned svm-train path")?;
    let predict=std::env::var("AEVRAL_SVM_PREDICT").map_err(|_|"missing pinned svm-predict path")?;
    if !Path::new(&train).is_file()||!Path::new(&predict).is_file(){return Err("missing official LIBSVM executables".into())}Ok((train,predict))
}
fn lead(x:&[[f64;D]],y:&[f64],m:usize,recipe:Recipe,dir:&Path,start:Instant,bins:&(String,String))->Result<(Vec<f64>,usize,f64),String>{
    let now=Instant::now();write_rows(&dir.join("train.svm"),x,Some(y))?;
    let args=vec!["-s".into(),"3".into(),"-t".into(),"2".into(),"-g".into(),recipe.gamma().to_string(),"-c".into(),recipe.c.to_string(),"-p".into(),"0.05".into(),"-e".into(),"0.001".into(),"-h".into(),"1".into(),"-m".into(),"128".into(),"-b".into(),"0".into(),"-q".into(),"train.svm".into(),"model.svm".into()];
    command(&bins.0,&args,dir,start)?;
    let model=std::fs::read_to_string(dir.join("model.svm")).map_err(|e|e.to_string())?;
    let count=model_count(&model,x.len(),recipe.gamma())?;
    command(&bins.1,&["-b","0","-q","eval.svm","model.svm","pred.txt"].map(String::from),dir,start)?;
    let p=predictions(&std::fs::read_to_string(dir.join("pred.txt")).map_err(|e|e.to_string())?,m)?;
    Ok((p,count,now.elapsed().as_secs_f64()))
}
pub fn smoke()->Result<(),String>{
    let bins=backend()?;let dir=Scratch::new()?;let point=|v:f64|{let mut x=[0.;D];x[0]=v*(D as f64).sqrt();x};
    let y=[-2.,-1.,0.,1.,2.];let x=y.map(point);let ev=[point(-1.),point(1.)];write_rows(&dir.0.join("eval.svm"),&ev,None)?;
    let(p,n,_)=lead(&x,&y,2,Recipe{length:1.,c:10.},&dir.0,Instant::now(),&bins)?;
    if n==0||(p[0]+1.).abs()>0.1||(p[1]-1.).abs()>0.1{return Err(format!("synthetic epsilon-SVR failed: {p:?}"))}
    println!("svr-smoke,333,{n},{:.8},{:.8},pass",p[0],p[1]);Ok(())
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let recipe=config(label)?;let bins=backend()?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let rows=subset(&f.train,BUDGET);if rows.len()!=BUDGET{return Err("invalid SVR training subset".into())}
    let x:Vec<_>=rows.iter().map(|r|r.input.x).collect();let ev:Vec<_>=f.eval.iter().map(|r|r.input.x).collect();
    let workspace=128*1024*1024+8*(x.len()*(D+Y+1)+m*(D+Y+1)+4*x.len()*D);
    if workspace>2_000_000_000{println!("blocked,{id},{label},workspace guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
    let now=Instant::now();let dir=Scratch::new()?;write_rows(&dir.0.join("eval.svm"),&ev,None)?;
    let mut pred=vec![[0.;Y];m];let mut total_sv=0;
    for h in 0..Y{let y:Vec<_>=rows.iter().map(|r|r.normalized_target[h]).collect();
        let(p,count,secs)=match lead(&x,&y,m,recipe,&dir.0,now,&bins){Ok(v)=>v,Err(e)if e=="runtime guard"=>{println!("blocked,{id},{label},{e}");out.flush().map_err(|e|e.to_string())?;return Ok(())},Err(e)=>return Err(e)};
        println!("svr-lead,{id},{label},{},{count},{secs:.6}",h+1);
        if h==0{let estimate=secs*Y as f64+now.elapsed().as_secs_f64()-secs;
            println!("svr-preflight,{id},{label},{BUDGET},{workspace},{secs:.6},{estimate:.6}");
            if 2.*estimate+180.>GUARD{println!("blocked,{id},{label},preflight runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
        }
        total_sv+=count;for(i,v)in p.into_iter().enumerate(){pred[i][h]=v;}
    }
    let predictions=f.eval.iter().zip(pred).map(|(r,p)|(r.input.origin,f.restore(p))).collect();
    println!("svr-config,{id},{label},333,{BUDGET},{},{},{},0.05,0.001,128,{Y},{total_sv},{:.6}",recipe.length,recipe.c,recipe.gamma(),now.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn model_validation(){let s="svm_type epsilon_svr\nkernel_type rbf\ngamma 0.00625\ntotal_sv 1\nSV\n1 1:2\n";assert_eq!(model_count(s,2,0.00625).unwrap(),1);assert!(model_count(s,2,0.025).is_err());assert!(model_count(s,0,0.00625).is_err());assert!(model_count(&s.replace("total_sv 1","total_sv 2"),2,0.00625).is_err());}
    #[test]fn finite_prediction_validation(){assert_eq!(predictions("1\n-2\n",2).unwrap(),vec![1.,-2.]);assert!(predictions("NaN\n",1).is_err());assert!(predictions("1\n",2).is_err());}
    #[test]fn dummy_eval_labels(){let p=std::env::temp_dir().join(format!("aevral-svr-export-test-{}",std::process::id()));write_rows(&p,&[[1.;D];2],None).unwrap();let s=std::fs::read_to_string(&p).unwrap();assert!(s.lines().all(|l|l.starts_with("0 1:1 ")&&l.split_whitespace().count()==D+1));std::fs::remove_file(p).unwrap();}
}
