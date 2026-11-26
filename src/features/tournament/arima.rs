//! Bounded conditional ARIMA/ARMAX and multiplicative seasonal ARIMA.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow};
use crate::{benchmark::Metrics,windows::Calendar};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const H:usize=72;
const Y:usize=24;
const N:usize=H+Y;
const SAMPLE:usize=2048;
#[derive(Clone,Copy,Debug)]
struct Spec{p:usize,d:usize,seasonal_ar:bool,seasonal_diff:bool,x:bool}
impl Spec{
    fn parse(s:&str)->Result<Self,String>{
        let(p,d,seasonal_ar,seasonal_diff,x)=match s{
            "arima-101"=>(1,0,false,false,false),"arima-201"=>(2,0,false,false,false),
            "arima-111"=>(1,1,false,false,false),"arima-211"=>(2,1,false,false,false),
            "arimax-101"=>(1,0,false,false,true),"arimax-111"=>(1,1,false,false,true),
            "sarima-101-100"=>(1,0,true,false,false),"sarima-111-100"=>(1,1,true,false,false),
            "sarima-101-010"=>(1,0,false,true,false),_=>return Err("unknown ARIMA config".into())};
        Ok(Self{p,d,seasonal_ar,seasonal_diff,x})
    }
    fn diff(self)->usize{if self.seasonal_diff{24}else{self.d}}
    fn start(self)->usize{self.diff()+self.p+if self.seasonal_ar{24}else{0}}
    fn count(self)->usize{2+self.p+usize::from(self.seasonal_ar)+if self.x{8}else{0}}
}
struct Params{c:f64,theta:f64,lags:Vec<(usize,f64)>,beta:[f64;8]}
fn params(s:Spec,v:&[f64])->Params{
    let k2=if s.p==2{v[3]}else{0.};
    let phi=v[2]*(1.-k2);
    let mut lags=vec![(1,phi)];if s.p==2{lags.push((2,k2))}
    let mut j=2+s.p;
    if s.seasonal_ar{let a=v[j];j+=1;let extra:Vec<_>=lags.iter().map(|&(i,b)|(i+24,-a*b)).collect();lags.push((24,a));lags.extend(extra)}
    let mut beta=[0.;8];if s.x{beta.copy_from_slice(&v[j..j+8])}
    Params{c:v[0],theta:v[1],lags,beta}
}
fn difference(y:&[f64],lag:usize,z:&mut[f64]){for i in lag..y.len(){z[i]=if lag==0{y[i]}else{y[i]-y[i-lag]}}}
fn features(origin:i64,w:&[f64],n:usize)->Vec<[f64;8]>{
    (0..n).map(|i|{let mut v=[0.;8];let cal=Calendar::Both.features(origin+(i as i64-H as i64)*60);
        v[..4].copy_from_slice(&cal);v[4..].copy_from_slice(w);v}).collect()
}
fn step(p:&Params,z:&[f64],e:&[f64],i:usize,x:Option<&[f64;8]>)->f64{
    p.c+p.lags.iter().map(|&(lag,b)|b*z[i-lag]).sum::<f64>()+p.theta*e[i-1]+x.map_or(0.,|v|p.beta.iter().zip(v).map(|(b,x)|b*x).sum::<f64>())
}
struct Window{z:[f64;H+1],x:Vec<[f64;8]>}
fn windows(rows:&[TrainingRow],s:Spec)->Vec<Window>{
    let n=rows.len().min(SAMPLE);(0..n).map(|i|{
        let r=&rows[i*rows.len()/n];let mut y=[0.;H+1];y[..H].copy_from_slice(&r.input.x[..H]);y[H]=r.normalized_target[0];
        let mut z=[0.;H+1];difference(&y,s.diff(),&mut z);
        let x=if s.x{features(r.input.origin,&r.input.x[76..80],H+1)}else{vec![]};
        Window{z,x}
    }).collect()
}
fn objective(s:Spec,v:&[f64],windows:&[Window])->f64{
    let p=params(s,v);let mut loss=0.;
    for w in windows{let mut e=[0.;H+1];
        for i in s.start()..=H{e[i]=w.z[i]-step(&p,&w.z,&e,i,w.x.get(i))}
        loss+=e[H]*e[H];
    }loss/windows.len()as f64
}
fn bound(s:Spec,j:usize)->f64{if j==0{4.}else if j<2+s.p+usize::from(s.seasonal_ar){0.98}else{4.}}
fn optimize(s:Spec,w:&[Window])->Result<(Vec<f64>,f64,usize,f64,bool),String>{
    if w.is_empty(){return Err("empty training sample".into())}
    let mut v=vec![0.;s.count()];v[2]=0.5;
    let mut best=objective(s,&v,w);let mut delta=0.25;let mut evals=1;
    let now=Instant::now();
    // Fixed deterministic coordinate pattern search; finite budget, no evaluation-dependent tuning.
    for _ in 0..96{
        if now.elapsed().as_secs_f64()>1000.{return Err("conditional fit exceeded 1000s guard".into())}
        let before=best;
        for j in 0..v.len(){
            let old=v[j];let mut chosen=old;
            for sign in [-1.,1.]{
                let a=(old+sign*delta).clamp(-bound(s,j),bound(s,j));if a==old{continue}
                v[j]=a;let f=objective(s,&v,w);evals+=1;
                if f.is_finite()&&f<best-1e-12{best=f;chosen=a}
            }
            v[j]=chosen;
        }
        if before-best<=1e-9*before.max(1e-6){delta*=0.5}
        if delta<=1e-4{return Ok((v,best,evals,delta,true))}
    }
    if !best.is_finite(){return Err("nonfinite conditional objective".into())}
    Ok((v,best,evals,delta,false))
}
struct Model{s:Spec,p:Params,mean:f64,sd:f64}
impl Predictor for Model{
    fn predict(&self,x:&OriginInput)->[f64;Y]{
        let mut y=[0.;N];let mut z=[0.;N];let mut e=[0.;N];
        y[..H].copy_from_slice(&x.x[..H]);difference(&y[..H],self.s.diff(),&mut z);
        let fx=if self.s.x{features(x.origin,&x.x[76..80],N)}else{vec![]};
        for i in self.s.start()..H{e[i]=z[i]-step(&self.p,&z,&e,i,fx.get(i))}
        for i in H..N{
            z[i]=step(&self.p,&z,&e,i,fx.get(i));
            y[i]=z[i]+if self.s.diff()==0{0.}else{y[i-self.s.diff()]};
            // Future innovations remain exactly zero.
        }
        std::array::from_fn(|h|y[H+h]*self.sd+self.mean)
    }
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let s=Spec::parse(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);
    println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();let w=windows(&f.train,s);
    match optimize(s,&w){
        Ok((v,obj,evals,delta,converged))=>{
            println!("arima-config,{id},{label},{},{},{evals},{obj:.12e},{delta:.8},{converged},{:.6}",w.len(),s.count(),now.elapsed().as_secs_f64());
            println!("arima-coef,{id},{label},{v:?}");
            let model=Model{s,p:params(s,&v),mean:f.target_mean,sd:f.target_sd};
            score(&model,label,&f,&mut out)?;
        },Err(e)=>println!("blocked,{id},{label},{e}"),
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn seasonal_cross_term(){let s=Spec::parse("sarima-101-100").unwrap();let p=params(s,&[0.,0.1,0.5,0.4]);assert_eq!(p.lags,vec![(1,0.5),(24,0.4),(25,-0.2)]);}
    #[test]fn differences_roundtrip(){let y:Vec<_>=(0..96).map(|i|(i as f64).sin()).collect();for lag in [0,1,24]{let mut z=vec![0.;96];difference(&y,lag,&mut z);for i in lag..96{let v=z[i]+if lag==0{0.}else{y[i-lag]};assert!((v-y[i]).abs()<1e-12)}}}
    #[test]fn innovations_affect_first_forecast(){let mut x=[0.;80];x[71]=2.;let input=OriginInput{origin:0,x,recent:[0.;24]};let s=Spec::parse("arima-101").unwrap();let m=Model{s,p:params(s,&[0.,0.5,0.]),mean:0.,sd:1.};let y=m.predict(&input);assert_eq!(y[0],1.);assert_eq!(y[1],0.);}
    #[test]fn differenced_zero_drift_persistence(){let mut x=[0.;80];x[71]=2.;let input=OriginInput{origin:0,x,recent:[0.;24]};let s=Spec::parse("arima-111").unwrap();let m=Model{s,p:params(s,&[0.,0.,0.]),mean:0.,sd:1.};assert_eq!(m.predict(&input),[2.;24]);}
    #[test]fn conditional_objective_reduces(){let s=Spec::parse("arima-101").unwrap();let w=vec![Window{z:[1.;H+1],x:vec![]}];let before=objective(s,&[0.,0.,0.5],&w);let(_,after,_,_,_)=optimize(s,&w).unwrap();assert!(after<before);}
}
