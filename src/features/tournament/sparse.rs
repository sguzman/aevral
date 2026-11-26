//! C004 L03/L04: centered sufficient statistics and cyclic coordinate descent.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow,D};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;
const MAX_SWEEPS:usize=10000;
const TOL:f64=1e-7;
const CONFIGS:[(&str,f64,f64);10]=[
    ("lasso-1e-4",1e-4,1.),("lasso-1e-3",1e-3,1.),("lasso-1e-2",1e-2,1.),("lasso-0.1",0.1,1.),
    ("enet-1e-3-a25",1e-3,0.25),("enet-1e-3-a75",1e-3,0.75),
    ("enet-1e-2-a25",1e-2,0.25),("enet-1e-2-a75",1e-2,0.75),
    ("enet-0.1-a25",0.1,0.25),("enet-0.1-a75",0.1,0.75)];
pub(super) struct Gram{pub(super) mx:[f64;D],pub(super) my:[f64;Y],pub(super) g:Vec<[f64;D]>,pub(super) xy:Vec<[f64;Y]>,yy:[f64;Y]}
impl Gram{
    pub(super) fn fit(rows:&[TrainingRow])->Result<Self,String>{
        if rows.is_empty(){return Err("empty training cohort".into())}
        let n=rows.len() as f64;
        let mut q=Self{mx:[0.;D],my:[0.;Y],g:vec![[0.;D];D],xy:vec![[0.;Y];D],yy:[0.;Y]};
        for r in rows{
            for j in 0..D{q.mx[j]+=r.input.x[j]}
            for k in 0..Y{q.my[k]+=r.normalized_target[k]}
        }
        q.mx.iter_mut().for_each(|v|*v/=n);q.my.iter_mut().for_each(|v|*v/=n);
        for r in rows{
            let x:[f64;D]=std::array::from_fn(|j|r.input.x[j]-q.mx[j]);
            let y:[f64;Y]=std::array::from_fn(|k|r.normalized_target[k]-q.my[k]);
            for j in 0..D{
                for i in 0..=j{q.g[j][i]+=x[j]*x[i]}
                for k in 0..Y{q.xy[j][k]+=x[j]*y[k]}
            }
            for k in 0..Y{q.yy[k]+=y[k]*y[k]}
        }
        for j in 0..D{
            for i in 0..=j{let v=q.g[j][i]/n;q.g[j][i]=v;q.g[i][j]=v;}
            for k in 0..Y{q.xy[j][k]/=n}
        }
        q.yy.iter_mut().for_each(|v|*v/=n);
        if q.g.iter().flatten().chain(q.xy.iter().flatten()).any(|v|!v.is_finite())||q.g.iter().enumerate().any(|(j,r)|r[j]<=0.){return Err("nonfinite or degenerate Gram design".into())}
        Ok(q)
    }
    fn gradient(&self,b:&[f64;D],k:usize)->[f64;D]{
        std::array::from_fn(|j|self.g[j].iter().zip(b).map(|(a,b)|a*b).sum::<f64>()-self.xy[j][k])
    }
    fn objective(&self,b:&[f64;D],k:usize,l1:f64,l2:f64)->f64{
        let g=self.gradient(b,k);
        let mse=(self.yy[k]+(0..D).map(|j|b[j]*(g[j]-self.xy[j][k])).sum::<f64>())/2.;
        mse+b.iter().map(|v|l1*v.abs()+0.5*l2*v*v).sum::<f64>()
    }
    fn solve_output(&self,k:usize,lambda:f64,alpha:f64)->Result<([f64;D],usize,f64,f64),String>{
        let(l1,l2)=(lambda*alpha,lambda*(1.-alpha));let mut b=[0.;D];let mut grad=self.gradient(&b,k);
        let mut prev=self.objective(&b,k,l1,l2);
        for sweep in 1..=MAX_SWEEPS{
            for j in 0..D{
                let z=self.g[j][j]*b[j]-grad[j];
                let new=z.signum()*(z.abs()-l1).max(0.)/(self.g[j][j]+l2);
                let delta=new-b[j];b[j]=new;
                if delta!=0.{for i in 0..D{grad[i]+=delta*self.g[j][i];}}
            }
            // Recompute gradient from scratch to bound incremental roundoff.
            grad=self.gradient(&b,k);
            let residual=(0..D).map(|j|{
                let g=grad[j]+l2*b[j];
                if b[j]==0.{(g.abs()-l1).max(0.)}else{(g+l1*b[j].signum()).abs()}
            }).fold(0.,f64::max);
            let obj=self.objective(&b,k,l1,l2);
            if !obj.is_finite()||!residual.is_finite()||obj>prev+1e-10*(1.+prev.abs()){return Err(format!("objective increased or nonfinite at output {} sweep {sweep}",k+1))}
            if residual<=TOL{return Ok((b,sweep,residual,obj))}
            prev=obj;
        }
        Err(format!("KKT tolerance not reached at output {} after {MAX_SWEEPS} sweeps",k+1))
    }
}
struct Sparse{beta:Vec<[f64;D]>,intercept:[f64;Y],mean:f64,sd:f64}
impl Predictor for Sparse{
    fn predict(&self,x:&OriginInput)->[f64;Y]{
        std::array::from_fn(|k|(self.intercept[k]+self.beta[k].iter().zip(x.x.iter()).map(|(a,b)|a*b).sum::<f64>())*self.sd+self.mean)
    }
}
fn fit(q:&Gram,lambda:f64,alpha:f64,mean:f64,sd:f64)->Result<(Sparse,usize,f64,f64,usize),String>{
    let mut beta=Vec::new();let mut intercept=[0.;Y];let(mut sweeps,mut residual,mut objective,mut nonzero)=(0,0.0f64,0.,0);
    for k in 0..Y{
        let(b,s,r,o)=q.solve_output(k,lambda,alpha)?;
        intercept[k]=q.my[k]-b.iter().zip(q.mx.iter()).map(|(a,b)|a*b).sum::<f64>();
        sweeps=sweeps.max(s);residual=residual.max(r);objective+=o;nonzero+=b.iter().filter(|&&v|v!=0.).count();beta.push(b);
    }
    Ok((Sparse{beta,intercept,mean,sd},sweeps,residual,objective/Y as f64,nonzero))
}
pub fn run(data:&Path,id:&str,output:&Path,selection:&str)->Result<(),String>{
    if selection!="all"&&!CONFIGS.iter().any(|(label,_,_)|*label==selection){return Err("unknown sparse configuration".into())}
    let f=load(data,Fold::development(id)?)?;
    let (n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if (f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=crate::benchmark::Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if (daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{},{},{},{}",n,m,f.possible,f.temperature_complete);
    println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();let q=Gram::fit(&f.train)?;println!("gram-seconds,{id},{:.6}",now.elapsed().as_secs_f64());
    for(label,lambda,alpha)in CONFIGS{
        if selection!="all"&&selection!=label{continue}
        let now=Instant::now();match fit(&q,lambda,alpha,f.target_mean,f.target_sd){
            Ok((model,sweeps,residual,obj,nonzero))=>{
                println!("sparse-config,{id},{label},{lambda},{alpha},{sweeps},{residual:.12e},{obj:.12e},{nonzero},{:.6}",now.elapsed().as_secs_f64());
                score(&model,label,&f,&mut out)?;
            },
            Err(e)=>println!("blocked,{id},{label},{e}"),
        }
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn fixture()->Vec<TrainingRow>{
        let mut rows=Vec::new();
        for j in 0..D{for s in [-1.,1.]{
            let mut x=[2.;D];x[j]+=s*(D as f64).sqrt();
            let y=std::array::from_fn(|k|3.+s*(D as f64).sqrt()*(j+1)as f64*(k+1)as f64/100.);
            rows.push(TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:y,normalized_target:y});
        }}rows
    }
    #[test]fn orthogonal_shrinkage_intercept(){
        let q=Gram::fit(&fixture()).unwrap();
        for alpha in [0.,0.25,0.75,1.]{
            let(m,_,r,_,_)=fit(&q,0.1,alpha,10.,2.).unwrap();assert!(r<=TOL);
            for k in 0..Y{
                for j in 0..D{let exact=(((j+1)*(k+1))as f64/100.-0.1*alpha).max(0.)/(1.+0.1*(1.-alpha));assert!((m.beta[k][j]-exact).abs()<1e-9);}
                assert!((m.intercept[k]+2.*m.beta[k].iter().sum::<f64>()-3.).abs()<1e-9);
            }
            let p=m.predict(&OriginInput{origin:0,x:[2.;D],recent:[0.;Y]});assert!(p.iter().all(|v|(v-16.).abs()<1e-9));
        }
    }
    #[test]fn objective_matches_direct_loss(){
        let rows=fixture();let q=Gram::fit(&rows).unwrap();let(b,_,r,o)=q.solve_output(0,0.01,0.75).unwrap();assert!(r<=TOL);
        let mse=rows.iter().map(|r|{let p=q.my[0]+(0..D).map(|j|b[j]*(r.input.x[j]-q.mx[j])).sum::<f64>();(p-r.normalized_target[0]).powi(2)}).sum::<f64>()/(2.*rows.len()as f64);
        let penalty=b.iter().map(|v|0.0075*v.abs()+0.00125*v*v).sum::<f64>();
        assert!((o-mse-penalty).abs()<1e-9);
    }
    #[test]fn exact_zero_coefficients(){let q=Gram::fit(&fixture()).unwrap();let(b,_,_,_)=q.solve_output(0,100.,1.).unwrap();assert_eq!(b,[0.;D]);}
}
