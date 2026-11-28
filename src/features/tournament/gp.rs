//! C007 K08/K09: exact subset GP and streaming FITC inducing GP.
use super::{load,score,Fold,TrainingRow,D,Y};
use super::neighbors::{subset,Cached};
use super::kernel_linalg::Cholesky;
use super::kernel_ridge::{rbf,target_mean,checked_solve,DenseRbf};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const JITTER:f64=1e-8;
const NOISE:f64=0.05;
const GUARD:f64=1400.;
#[derive(Clone,Copy)]
struct Recipe{length:f64,inducing:Option<usize>}
impl Recipe{fn budget(self)->usize{if self.inducing.is_some(){8192}else{1024}}}
fn config(label:&str)->Result<Recipe,String>{
    let(length,inducing)=match label{
        "gp-s05"=>(0.5,None),"gp-s1"=>(1.,None),
        "fitc-s05-m128"=>(0.5,Some(128)),"fitc-s05-m256"=>(0.5,Some(256)),
        "fitc-s1-m128"=>(1.,Some(128)),"fitc-s1-m256"=>(1.,Some(256)),
        _=>return Err("unknown GP configuration".into())
    };Ok(Recipe{length,inducing})
}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>GUARD{Err("runtime guard".into())}else{Ok(())}}
fn norm2(x:&[f64])->f64{x.iter().map(|v|v*v).sum()}
fn bounded_unit(v:f64)->Result<f64,String>{
    if !v.is_finite()||v< -1e-6||v>1.+1e-6{return Err(format!("GP variance/diagonal outside unit prior: {v}"))}
    Ok(v.clamp(0.,1.))
}
struct Fitc{x:Vec<[f64;D]>,mean:[f64;Y],alpha:Vec<[f64;Y]>,prior:Cholesky,posterior:Cholesky,length:f64,min_d:f64,max_d:f64}
impl Fitc{
    fn fit(rows:&[&TrainingRow],m:usize,length:f64,now:Instant)->Result<(Self,f64),String>{
        let n=rows.len();if m==0||m>n||length<=0.||!length.is_finite(){return Err("invalid FITC dimensions".into())}
        let x:Vec<_>=(0..m).map(|i|rows[i*n/m].input.x).collect();let mean=target_mean(rows);
        let mut a=vec![0.;m*m];
        for i in 0..m{guard(now)?;a[i*m+i]=1.+JITTER;for j in 0..i{let k=rbf(&x[i],&x[j],length);a[i*m+j]=k;a[j*m+i]=k;}}
        let prior=Cholesky::factor_guarded(&a,m,Some(now))?;
        let mut rhs=vec![vec![0.;m];Y];let(mut min_d,mut max_d)=(f64::INFINITY,f64::NEG_INFINITY);
        for r in rows{
            guard(now)?;let k:Vec<_>=x.iter().map(|z|rbf(z,&r.input.x,length)).collect();
            let q=bounded_unit(norm2(&prior.forward(&k)?))?;
            let d=1.-q+NOISE;min_d=min_d.min(d);max_d=max_d.max(d);
            for i in 0..m{let w=k[i]/d;for j in 0..=i{a[i*m+j]+=w*k[j];}
                for h in 0..Y{rhs[h][i]+=w*(r.normalized_target[h]-mean[h]);}}
        }
        for i in 0..m{for j in 0..i{a[j*m+i]=a[i*m+j];}}
        let posterior=Cholesky::factor_guarded(&a,m,Some(now))?;let mut alpha=vec![[0.;Y];m];let mut residual: f64=0.;
        for(h,b)in rhs.iter().enumerate(){guard(now)?;let(v,e)=checked_solve(&a,m,&posterior,b)?;residual=residual.max(e);for i in 0..m{alpha[i][h]=v[i];}}
        Ok((Self{x,mean,alpha,prior,posterior,length,min_d,max_d},residual))
    }
    fn predict(&self,q:&[f64;D])->[f64;Y]{
        let mut p=self.mean;for(i,x)in self.x.iter().enumerate(){let k=rbf(x,q,self.length);for h in 0..Y{p[h]+=k*self.alpha[i][h];}}p
    }
    fn variance(&self,q:&[f64;D])->Result<f64,String>{
        let k:Vec<_>=self.x.iter().map(|z|rbf(z,q,self.length)).collect();
        let a=bounded_unit(norm2(&self.prior.forward(&k)?))?;
        let b=norm2(&self.posterior.forward(&k)?);
        bounded_unit(1.-a+b)
    }
}
enum Model{Exact(DenseRbf),Sparse(Fitc)}
impl Model{
    fn fit(rows:&[&TrainingRow],r:Recipe,now:Instant)->Result<(Self,f64),String>{
        if let Some(m)=r.inducing{Fitc::fit(rows,m,r.length,now).map(|(m,e)|(Self::Sparse(m),e))}
        else{DenseRbf::fit(rows,r.length,NOISE+JITTER,now).map(|(m,e)|(Self::Exact(m),e))}
    }
    fn predict(&self,q:&[f64;D])->[f64;Y]{match self{Self::Exact(m)=>m.predict(q),Self::Sparse(m)=>m.predict(q)}}
    fn variance(&self,q:&[f64;D])->Result<f64,String>{match self{
        Self::Exact(m)=>{let k:Vec<_>=m.x.iter().map(|x|rbf(x,q,m.length)).collect();bounded_unit(1.-norm2(&m.factor.forward(&k)?))},
        Self::Sparse(m)=>m.variance(q)
    }}
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    config(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    run_prepared(&f,output,label)
}
pub(super) fn run_prepared(f:&super::PreparedFold,output:&Path,label:&str)->Result<(),String>{
    let id=f.fold.id;
    let recipe=config(label)?;
    let m=f.eval.len();
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let rows=subset(&f.train,recipe.budget());let n=rows.len();
    if super::Fold::retrospective(id).is_ok(){
        super::representation::origins(&output.parent().ok_or("output directory")?.join("backend-fit-origins.txt"),rows.iter().map(|r|r.input.origin))?;
    }
    
    if n!=recipe.budget()||rows.iter().any(|r|r.input.x.iter().chain(&r.normalized_target).any(|v|!v.is_finite())){return Err("invalid GP training subset".into())}
    let dim=recipe.inducing.unwrap_or(n);let workspace=8*(3*dim*dim+3*dim*Y+dim*(D+Y+5)+n+m*(Y+1));
    let now=Instant::now();let p=n.min(512);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    let timer=Instant::now();let(sm,_)=Model::fit(&small,recipe,now)?;let fit_probe=timer.elapsed().as_secs_f64();
    let q=m.min(32);let timer=Instant::now();for i in 0..q{let _=std::hint::black_box(sm.predict(&f.eval[i*m/q].input.x));}let pred_probe=timer.elapsed().as_secs_f64();
    let timer=Instant::now();for i in 0..q{let _=std::hint::black_box(sm.variance(&f.eval[i].input.x)?);}let var_probe=timer.elapsed().as_secs_f64();drop(sm);
    let ratio=n as f64/p as f64;
    let(fit_scale,pred_scale,var_scale)=if recipe.inducing.is_some(){(ratio,1.,1.)}else{(ratio.powi(3),ratio,ratio.powi(2))};
    let estimate=fit_probe*fit_scale+pred_probe*pred_scale*m as f64/q as f64+var_probe*var_scale;
    println!("gp-preflight,{id},{label},{n},{p},{q},{workspace},{fit_probe:.6},{pred_probe:.6},{var_probe:.6},{estimate:.6}");
    if workspace>2_000_000_000||2.*estimate+180.>GUARD{println!("blocked,{id},{label},preflight workspace or runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
    let timer=Instant::now();let(model,residual)=match Model::fit(&rows,recipe,now){Ok(v)=>v,Err(e) if e=="runtime guard"=>{println!("blocked,{id},{label},{e}");out.flush().map_err(|e|e.to_string())?;return Ok(())},Err(e)=>return Err(e)};
    let fit=timer.elapsed().as_secs_f64();let timer=Instant::now();let mut predictions=Vec::with_capacity(m);
    let(mut min_var,mut max_var,mut sum_var)=(f64::INFINITY,f64::NEG_INFINITY,0.);
    for(i,r)in f.eval.iter().enumerate(){
        if guard(now).is_err(){println!("blocked,{id},{label},runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
        let pred=model.predict(&r.input.x);if pred.iter().any(|v|!v.is_finite()){return Err("nonfinite GP prediction".into())}
        if i<q{let v=model.variance(&r.input.x)?;min_var=min_var.min(v);max_var=max_var.max(v);sum_var+=v;}
        predictions.push((r.input.origin,f.restore(pred)));
    }
    let pred=timer.elapsed().as_secs_f64();
    println!("gp-variance,{id},{label},{q},{min_var:.12e},{max_var:.12e},{:.12e}",sum_var/q as f64);
    if let Model::Sparse(s)=&model{println!("fitc-diagonal,{id},{label},{:.12e},{:.12e}",s.min_d,s.max_d);}
    println!("gp-config,{id},{label},{n},{dim},{},{},{NOISE},{JITTER},{residual:.12e},{fit:.6},{pred:.6}",if recipe.inducing.is_some(){"fitc"}else{"exact"},recipe.length);
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;use super::super::OriginInput;
    fn row(v:f64,y:f64)->TrainingRow{let mut x=[0.;D];x[0]=v*(D as f64).sqrt();TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:[y;Y],normalized_target:[y;Y]}}
    #[test]fn exact_one_point_variance(){let d=[row(0.,3.)];let rows:Vec<_>=d.iter().collect();let(m,_)=Model::fit(&rows,Recipe{length:1.,inducing:None},Instant::now()).unwrap();let q=row(1.,0.).input.x;let k=rbf(&q,&d[0].input.x,1.);assert_eq!(m.predict(&q),[3.;Y]);assert!((m.variance(&q).unwrap()-(1.-k*k/(1.+NOISE+JITTER))).abs()<1e-12);}
    #[test]fn exact_mean_matches_krr(){let d=[row(-1.,-2.),row(1.,2.)];let rows:Vec<_>=d.iter().collect();let(m,_)=Model::fit(&rows,Recipe{length:1.,inducing:None},Instant::now()).unwrap();let q=d[1].input.x;let k=rbf(&d[0].input.x,&q,1.);let expected=2.*(1.-k)/(1.+NOISE+JITTER-k);assert!((m.predict(&q)[0]-expected).abs()<1e-12);}
    #[test]fn all_inducing_matches_exact(){let d=[row(-1.,-2.),row(0.,1.),row(1.,2.)];let rows:Vec<_>=d.iter().collect();let(e,_)=Model::fit(&rows,Recipe{length:0.5,inducing:None},Instant::now()).unwrap();let(s,_)=Model::fit(&rows,Recipe{length:0.5,inducing:Some(3)},Instant::now()).unwrap();for v in [-2.,-1.,0.,0.5,1.,2.]{let q=row(v,0.).input.x;assert!((e.predict(&q)[0]-s.predict(&q)[0]).abs()<1e-6);assert!((e.variance(&q).unwrap()-s.variance(&q).unwrap()).abs()<1e-6);}}
    #[test]fn sparse_diagonal_and_variance(){let d=[row(-2.,1.),row(-1.,2.),row(1.,4.),row(2.,5.)];let rows:Vec<_>=d.iter().collect();let(s,_)=Fitc::fit(&rows,1,1.,Instant::now()).unwrap();assert!(s.min_d>=NOISE&&s.max_d>0.9);assert!(s.variance(&d[0].input.x).unwrap()<s.variance(&d[3].input.x).unwrap());assert!(bounded_unit(-0.01).is_err());assert!(bounded_unit(1.01).is_err());}
}
