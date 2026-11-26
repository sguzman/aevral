//! C004 L05/L06: exact Huber and median objectives via bounded ADMM.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow,D};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;
const P:usize=D+1;
const RHO:f64=1.;
const MAX_ITER:usize=1000;
const TOL:f64=1e-4;
#[derive(Clone,Copy)]
enum Loss{Huber(f64),Median}
impl Loss{
    fn value(self,r:f64)->f64{match self{Self::Huber(d)=>if r.abs()<=d{0.5*r*r}else{d*(r.abs()-0.5*d)},Self::Median=>0.5*r.abs()}}
    fn prox(self,v:f64)->f64{match self{Self::Huber(d)=>if v.abs()<=d*(1.+1./RHO){v*RHO/(RHO+1.)}else{v-v.signum()*d/RHO},Self::Median=>v.signum()*(v.abs()-0.5/RHO).max(0.)}}
}
const CONFIGS:[(&str,Loss,f64);7]=[
    ("huber-0.25",Loss::Huber(0.25),0.),("huber-0.5",Loss::Huber(0.5),0.),("huber-1",Loss::Huber(1.),0.),("huber-2",Loss::Huber(2.),0.),
    ("median-0",Loss::Median,0.),("median-1e-4",Loss::Median,1e-4),("median-1e-2",Loss::Median,1e-2)];
fn design(x:&[f64;D])->[f64;P]{let mut a=[1.;P];a[1..].copy_from_slice(x);a}
fn gram(xs:&[[f64;P]],lambda:f64)->Result<Vec<[f64;P]>,String>{
    let n=xs.len() as f64;let mut g=vec![[0.;P];P];
    for x in xs{for j in 0..P{for i in 0..=j{g[j][i]+=x[j]*x[i]}}}
    for j in 0..P{for i in 0..=j{g[j][i]/=n;}if j>0{g[j][j]+=lambda/RHO;}}
    // Cholesky of normalized Gram for the ADMM quadratic update; reject bad pivots.
    let max=(0..P).map(|j|g[j][j]).fold(0.,f64::max);
    for j in 0..P{for i in 0..=j{
        let mut v=g[j][i];for k in 0..i{v-=g[j][k]*g[i][k]}
        if i==j{if !v.is_finite()||v<=max*1e-12{return Err("ill-conditioned ADMM quadratic update".into())}g[j][i]=v.sqrt()}else{g[j][i]=v/g[i][i]}
    }}
    Ok(g)
}
fn solve(l:&[[f64;P]],mut b:Vec<[f64;Y]>)->Vec<[f64;Y]>{
    for j in 0..P{for k in 0..Y{let mut v=b[j][k];for i in 0..j{v-=l[j][i]*b[i][k]}b[j][k]=v/l[j][j]}}
    for j in (0..P).rev(){for k in 0..Y{let mut v=b[j][k];for i in j+1..P{v-=l[i][j]*b[i][k]}b[j][k]=v/l[j][j]}}b
}
fn multiply(x:&[f64;P],b:&[[f64;Y]])->[f64;Y]{
    let mut y=[0.;Y];for j in 0..P{for k in 0..Y{y[k]+=x[j]*b[j][k]}}y
}
struct Robust{beta:Vec<[f64;Y]>,mean:f64,sd:f64}
impl Predictor for Robust{fn predict(&self,x:&OriginInput)->[f64;Y]{multiply(&design(&x.x),&self.beta).map(|v|v*self.sd+self.mean)}}
struct Fit{model:Robust,iterations:usize,primal:f64,stationarity:f64,objective:f64}
fn fit(rows:&[TrainingRow],loss:Loss,lambda:f64,mean:f64,sd:f64)->Result<Fit,String>{
    if rows.is_empty(){return Err("empty training".into())}
    let xs:Vec<_>=rows.iter().map(|r|design(&r.input.x)).collect();
    let l=gram(&xs,lambda)?;let n=rows.len() as f64;
    let(mut z,mut u)=(vec![[0.;Y];rows.len()],vec![[0.;Y];rows.len()]);
    for iter in 1..=MAX_ITER{
        let mut rhs=vec![[0.;Y];P];
        for (i,x) in xs.iter().enumerate(){
            let a:[f64;Y]=std::array::from_fn(|k|rows[i].normalized_target[k]+z[i][k]-u[i][k]);
            for j in 0..P{for k in 0..Y{rhs[j][k]+=x[j]*a[k]}}
        }
        for r in &mut rhs{for v in r{*v/=n}}
        let beta=solve(&l,rhs);
        if beta.iter().flatten().any(|v|!v.is_finite()){return Err("nonfinite ADMM coefficients".into())}
        let mut sq=[0.;Y];let mut objective=0.;
        for(i,x)in xs.iter().enumerate(){
            let p=multiply(x,&beta);
            for k in 0..Y{
                let r=p[k]-rows[i].normalized_target[k];
                z[i][k]=loss.prox(r+u[i][k]);
                let e=r-z[i][k];u[i][k]+=e;sq[k]+=e*e;
                if iter%10==0||iter==MAX_ITER{objective+=loss.value(r);}
            }
        }
        let primal=sq.iter().map(|v|(v/n).sqrt()).fold(0.,f64::max);
        if !primal.is_finite(){return Err("nonfinite primal residual".into())}
        if iter%10==0||iter==MAX_ITER{
            let mut grad=vec![[0.;Y];P];
            for(i,x)in xs.iter().enumerate(){for j in 0..P{for k in 0..Y{grad[j][k]+=x[j]*u[i][k]}}}
            let mut stationarity=0.0f64;
            for j in 0..P{for k in 0..Y{
                let penalty=if j==0{0.}else{lambda*beta[j][k]};
                stationarity=stationarity.max((RHO*grad[j][k]/n+penalty).abs());
            }}
            objective=objective/(n*Y as f64)+lambda*0.5*beta[1..].iter().flatten().map(|v|v*v).sum::<f64>()/Y as f64;
            if !objective.is_finite()||!stationarity.is_finite(){return Err("nonfinite diagnostics".into())}
            if iter%100==0||primal<=TOL&&stationarity<=TOL{println!("admm,{iter},{primal:.8e},{stationarity:.8e},{objective:.10}");}
            if primal<=TOL&&stationarity<=TOL{return Ok(Fit{model:Robust{beta,mean,sd},iterations:iter,primal,stationarity,objective})}
            if iter==MAX_ITER{return Err(format!("ADMM cap {MAX_ITER}; primal={primal:.8e}; stationarity={stationarity:.8e}"))}
        }
    }unreachable!()
}
pub fn run(data:&Path,id:&str,output:&Path,selection:&str)->Result<(),String>{
    let &(label,loss,lambda)=CONFIGS.iter().find(|(label,_,_)|*label==selection).ok_or("unknown robust configuration")?;
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=crate::benchmark::Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);
    println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();match fit(&f.train,loss,lambda,f.target_mean,f.target_sd){
        Ok(v)=>{println!("robust-config,{id},{label},{lambda},{},{:.12e},{:.12e},{:.12e},{},{:.6}",v.iterations,v.primal,v.stationarity,v.objective,P*Y,now.elapsed().as_secs_f64());score(&v.model,label,&f,&mut out)?;},
        Err(e)=>println!("blocked,{id},{label},{e}"),
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn exact_prox(){
        let h=Loss::Huber(0.5);
        assert_eq!(h.prox(0.4),0.2);assert_eq!(h.prox(2.),1.5);assert_eq!(h.prox(-2.),-1.5);
        assert_eq!(Loss::Median.prox(0.3),0.);assert_eq!(Loss::Median.prox(-2.),-1.5);
        for loss in [h,Loss::Median]{for v in [-2.,-0.4,0.,0.4,2.]{
            let p=loss.prox(v);let obj=|z:f64|loss.value(z)+0.5*RHO*(z-v).powi(2);
            assert!(obj(p)<=obj(p+1e-5)&&obj(p)<=obj(p-1e-5));
        }}
    }
    fn fixture()->Vec<TrainingRow>{
        let mut rows=Vec::new();for j in 0..D{for s in [-1.,1.]{
            let mut x=[0.;D];x[j]=s*(D as f64).sqrt();
            let y=std::array::from_fn(|k|1.+x[j]*(k+1)as f64/100.);
            rows.push(TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:y,normalized_target:y});
        }}rows
    }
    #[test]fn noiseless_exact_fit(){
        let rows=fixture();for loss in [Loss::Huber(0.5),Loss::Median]{
            let v=fit(&rows,loss,0.,10.,2.).unwrap();assert!(v.primal<=TOL&&v.stationarity<=TOL);
            for r in &rows{let p=v.model.predict(&r.input);for k in 0..Y{assert!((p[k]-(10.+2.*r.target[k])).abs()<1e-6)}}
        }
    }
    #[test]fn quadratic_solution(){
        let rows=fixture();let v=fit(&rows,Loss::Huber(100.),0.1,0.,1.).unwrap();
        for k in 0..Y{assert!((v.model.beta[0][k]-1.).abs()<5e-4);for j in 1..P{assert!((v.model.beta[j][k]-(k+1)as f64/110.).abs()<5e-4)}}
    }
    #[test]fn median_resists_outlier(){
        let clean=fixture();let mut rows=Vec::new();for r in clean{
            for offset in [0.,0.,3.]{let y=r.target.map(|v|v+offset);
                rows.push(TrainingRow{input:r.input.clone(),target:y,normalized_target:y});
            }
        }
        let v=fit(&rows,Loss::Median,0.,0.,1.).unwrap();
        assert!(v.model.beta[0].iter().all(|b|(*b-1.).abs()<1e-3));
    }
}
