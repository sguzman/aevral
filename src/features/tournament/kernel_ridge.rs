//! C007 K05/K06: exact primal linear and bounded dense RBF kernel ridge.
use super::{load,score,Fold,TrainingRow,D,Y};
use super::neighbors::{subset,distance,Cached};
use super::kernel_linalg::Cholesky;
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const GUARD:f64=1400.;
#[derive(Clone,Copy)]
enum Recipe{Linear(f64),Rbf(f64,f64)}
impl Recipe{
    fn budget(self)->usize{match self{Self::Linear(_)=>8192,Self::Rbf(..)=>2048}}
    fn label_params(self)->(&'static str,f64,f64){match self{Self::Linear(l)=>("linear",0.,l),Self::Rbf(s,l)=>("rbf",s,l)}}
}
fn config(label:&str)->Result<Recipe,String>{Ok(match label{
    "linear-krr-l001"=>Recipe::Linear(0.001),"linear-krr-l01"=>Recipe::Linear(0.01),"linear-krr-l1"=>Recipe::Linear(0.1),
    "rbf-krr-s05-l01"=>Recipe::Rbf(0.5,0.01),"rbf-krr-s05-l1"=>Recipe::Rbf(0.5,0.1),
    "rbf-krr-s1-l01"=>Recipe::Rbf(1.,0.01),"rbf-krr-s1-l1"=>Recipe::Rbf(1.,0.1),
    _=>return Err("unknown kernel ridge configuration".into())
})}
pub(super) fn rbf(a:&[f64;D],b:&[f64;D],length:f64)->f64{(-distance(a,b)/(2.*length*length)).exp()}
pub(super) fn target_mean(rows:&[&TrainingRow])->[f64;Y]{
    let mut m=[0.;Y];for r in rows{for h in 0..Y{m[h]+=r.normalized_target[h];}}m.map(|v|v/rows.len() as f64)
}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>GUARD{Err("runtime guard".into())}else{Ok(())}}
pub(super) fn checked_solve(a:&[f64],n:usize,c:&Cholesky,b:&[f64])->Result<(Vec<f64>,f64),String>{
    let x=c.solve(b)?;let mut err: f64=0.;
    for i in 0..n{let v=a[i*n..(i+1)*n].iter().zip(&x).map(|(a,b)|a*b).sum::<f64>();err=err.max((v-b[i]).abs()/(1.+b[i].abs()));}
    if !err.is_finite()||err>1e-8{return Err(format!("kernel solve residual failure: {err}"))}Ok((x,err))
}
struct Linear{mean_x:[f64;D],mean_y:[f64;Y],coef:Vec<Vec<f64>>}
impl Linear{
    fn fit(rows:&[&TrainingRow],lambda:f64,now:Instant)->Result<(Self,f64),String>{
        if rows.is_empty()||lambda<=0.||!lambda.is_finite(){return Err("invalid linear fit".into())}
        let n=rows.len() as f64;let scale=(D as f64).sqrt();let mean_y=target_mean(rows);let mut mean_x=[0.;D];
        for r in rows{for j in 0..D{mean_x[j]+=r.input.x[j]/scale/n;}}
        let mut a=vec![0.;D*D];let mut rhs=vec![vec![0.;D];Y];
        for r in rows{guard(now)?;let z:[f64;D]=std::array::from_fn(|j|r.input.x[j]/scale-mean_x[j]);
            for i in 0..D{for j in 0..=i{a[i*D+j]+=z[i]*z[j]/n;}for h in 0..Y{rhs[h][i]+=z[i]*(r.normalized_target[h]-mean_y[h])/n;}}
        }
        for i in 0..D{for j in 0..i{a[j*D+i]=a[i*D+j];}a[i*D+i]+=lambda;}
        let c=Cholesky::factor_guarded(&a,D,Some(now))?;let mut coef=Vec::with_capacity(Y);let mut residual: f64=0.;
        for b in rhs{let(x,e)=checked_solve(&a,D,&c,&b)?;coef.push(x);residual=residual.max(e);}
        Ok((Self{mean_x,mean_y,coef},residual))
    }
    fn predict(&self,q:&[f64;D])->[f64;Y]{
        let z:[f64;D]=std::array::from_fn(|j|q[j]/(D as f64).sqrt()-self.mean_x[j]);
        std::array::from_fn(|h|self.mean_y[h]+z.iter().zip(&self.coef[h]).map(|(a,b)|a*b).sum::<f64>())
    }
}
pub(super) struct DenseRbf{pub(super) x:Vec<[f64;D]>,pub(super) mean:[f64;Y],pub(super) alpha:Vec<[f64;Y]>,pub(super) factor:Cholesky,pub(super) length:f64}
impl DenseRbf{
    pub(super) fn fit(rows:&[&TrainingRow],length:f64,lambda:f64,now:Instant)->Result<(Self,f64),String>{
        if rows.is_empty()||length<=0.||lambda<=0.||!length.is_finite()||!lambda.is_finite(){return Err("invalid RBF fit".into())}
        let n=rows.len();let x:Vec<_>=rows.iter().map(|r|r.input.x).collect();let mean=target_mean(rows);
        let mut a=vec![0.;n*n];
        for i in 0..n{guard(now)?;a[i*n+i]=1.+lambda;for j in 0..i{let k=rbf(&x[i],&x[j],length);a[i*n+j]=k;a[j*n+i]=k;}}
        let factor=Cholesky::factor_guarded(&a,n,Some(now))?;let mut alpha=vec![[0.;Y];n];let mut residual: f64=0.;
        for h in 0..Y{guard(now)?;let b:Vec<_>=rows.iter().map(|r|r.normalized_target[h]-mean[h]).collect();let(v,e)=checked_solve(&a,n,&factor,&b)?;residual=residual.max(e);for i in 0..n{alpha[i][h]=v[i];}}
        Ok((Self{x,mean,alpha,factor,length},residual))
    }
    pub(super) fn predict(&self,q:&[f64;D])->[f64;Y]{
        let mut p=self.mean;for(i,x)in self.x.iter().enumerate(){let k=rbf(x,q,self.length);for h in 0..Y{p[h]+=k*self.alpha[i][h];}}p
    }
}
enum Model{Linear(Linear),Rbf(DenseRbf)}
impl Model{
    fn fit(rows:&[&TrainingRow],recipe:Recipe,now:Instant)->Result<(Self,f64),String>{match recipe{
        Recipe::Linear(l)=>Linear::fit(rows,l,now).map(|(m,e)|(Self::Linear(m),e)),
        Recipe::Rbf(s,l)=>DenseRbf::fit(rows,s,l,now).map(|(m,e)|(Self::Rbf(m),e))
    }}
    fn predict(&self,q:&[f64;D])->[f64;Y]{match self{Self::Linear(m)=>m.predict(q),Self::Rbf(m)=>m.predict(q)}}
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let recipe=config(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let rows=subset(&f.train,recipe.budget());let n=rows.len();
    if n!=recipe.budget()||rows.iter().any(|r|r.input.x.iter().chain(&r.normalized_target).any(|v|!v.is_finite())){return Err("invalid kernel training subset".into())}
    let dim=match recipe{Recipe::Linear(_)=>D,Recipe::Rbf(..)=>n};
    let workspace=8*(2*dim*dim+2*dim*Y+n*(D+Y+1)+m*(Y+1)+4*dim);
    let now=Instant::now();let p=n.min(512);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    let timer=Instant::now();let(sm,_)=Model::fit(&small,recipe,now)?;let fit_probe=timer.elapsed().as_secs_f64();
    let q=m.min(32);let timer=Instant::now();for i in 0..q{let _=std::hint::black_box(sm.predict(&f.eval[i*m/q].input.x));}let pred_probe=timer.elapsed().as_secs_f64();drop(sm);
    let ratio=n as f64/p as f64;
    let(fit_scale,pred_scale)=match recipe{Recipe::Linear(_)=>(ratio,1.),Recipe::Rbf(..)=>(ratio.powi(3),ratio)};
    let estimate=fit_probe*fit_scale+pred_probe*pred_scale*m as f64/q as f64;
    println!("krr-preflight,{id},{label},{n},{p},{q},{workspace},{fit_probe:.6},{pred_probe:.6},{estimate:.6}");
    if workspace>2_000_000_000||2.*estimate+180.>GUARD{println!("blocked,{id},{label},preflight workspace or runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
    let timer=Instant::now();let(model,residual)=match Model::fit(&rows,recipe,now){Ok(v)=>v,Err(e) if e=="runtime guard"=>{println!("blocked,{id},{label},{e}");out.flush().map_err(|e|e.to_string())?;return Ok(())},Err(e)=>return Err(e)};
    let fit=timer.elapsed().as_secs_f64();let timer=Instant::now();let mut predictions=Vec::with_capacity(m);
    for r in &f.eval{if guard(now).is_err(){println!("blocked,{id},{label},runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
        let pred=model.predict(&r.input.x);if pred.iter().any(|v|!v.is_finite()){return Err("nonfinite kernel prediction".into())}
        predictions.push((r.input.origin,f.restore(pred)));
    }
    let pred=timer.elapsed().as_secs_f64();let(kind,length,lambda)=recipe.label_params();
    println!("krr-config,{id},{label},{n},{kind},{length},{lambda},{dim},{residual:.12e},{fit:.6},{pred:.6}");
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;use super::super::OriginInput;
    fn row(v:f64,y:f64)->TrainingRow{let mut x=[0.;D];x[0]=v;TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:[y;Y],normalized_target:[y;Y]}}
    #[test]fn linear_primal_dual(){
        let d=[row(-2.,1.),row(1.,4.),row(3.,8.)];let rows:Vec<_>=d.iter().collect();let n=rows.len();let lambda=0.01;
        let (m,e)=Linear::fit(&rows,lambda,Instant::now()).unwrap();assert!(e<1e-12);
        let mut a=vec![0.;n*n];let mean_x=d.iter().map(|r|r.input.x[0]).sum::<f64>()/n as f64;let mean_y=target_mean(&rows)[0];
        for i in 0..n{for j in 0..n{a[i*n+j]=(d[i].input.x[0]-mean_x)*(d[j].input.x[0]-mean_x)/D as f64;if i==j{a[i*n+j]+=n as f64*lambda;}}}
        let alpha=Cholesky::factor(&a,n).unwrap().solve(&d.iter().map(|r|r.normalized_target[0]-mean_y).collect::<Vec<_>>()).unwrap();
        let q=row(2.,0.).input.x;let dual=mean_y+(0..n).map(|i|(q[0]-mean_x)*(d[i].input.x[0]-mean_x)/D as f64*alpha[i]).sum::<f64>();assert!((m.predict(&q)[0]-dual).abs()<1e-10);
    }
    #[test]fn linear_constant_intercept(){let d=[row(1.,5.),row(2.,5.)];let rows:Vec<_>=d.iter().collect();let(m,_)=Linear::fit(&rows,0.1,Instant::now()).unwrap();assert_eq!(m.predict(&[0.;D]),[5.;Y]);}
    #[test]fn rbf_symmetry_and_pd(){let d=[row(-1.,1.),row(1.,2.),row(3.,4.)];let mut a=vec![0.;9];for i in 0..3{for j in 0..3{a[i*3+j]=rbf(&d[i].input.x,&d[j].input.x,0.5);}}assert_eq!(a[0],1.);assert_eq!(a[1],a[3]);assert!(a[1]>a[2]);assert!(Cholesky::factor(&a,3).is_ok());}
    #[test]fn rbf_two_point_analytic(){let d=[row(-2.,-1.),row(2.,1.)];let rows:Vec<_>=d.iter().collect();let(m,e)=DenseRbf::fit(&rows,0.5,0.1,Instant::now()).unwrap();let k=rbf(&d[0].input.x,&d[1].input.x,0.5);let expected=(1.-k)/(1.1-k);assert!((m.predict(&d[1].input.x)[0]-expected).abs()<1e-12);assert!(e<1e-12);}
}
