//! C004 L01/L02: streamed Givens QR, with shared factorization for ridge.
use super::{load,score,Fold,OriginInput,Predictor,D};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const P:usize=D+1;
const Y:usize=24;
const WIDTH:usize=P+Y;
#[derive(Clone)]
struct Qr{r:Vec<f64>,n:usize}
impl Qr{
    fn new()->Self{Self{r:vec![0.;P*WIDTH],n:0}}
    // Apply orthogonal plane rotations to the augmented [X | Y] row.
    fn append(&mut self,mut row:[f64;WIDTH]){
        for j in 0..P{
            let off=j*WIDTH;let a=self.r[off+j];let b=row[j];
            if b==0.{continue}
            let h=a.hypot(b);let c=a/h;let s=b/h;
            self.r[off+j]=h;
            for k in j+1..WIDTH{
                let v=self.r[off+k];let w=row[k];
                self.r[off+k]=c*v+s*w;row[k]=-s*v+c*w;
            }
        }
        self.n+=1;
    }
    fn observation(&mut self,x:&[f64;D],y:&[f64;Y]){
        let mut row=[0.;WIDTH];row[0]=1.;
        row[1..P].copy_from_slice(x);row[P..].copy_from_slice(y);self.append(row);
    }
    fn ridge(&self,lambda:f64)->Self{
        let mut q=self.clone();let penalty=(self.n as f64*lambda).sqrt();
        // Intercept excluded from penalty; objective 1/(2n)||XW-Y||² + lambda/2 ||W||².
        if lambda>0.{for j in 1..P{let mut row=[0.;WIDTH];row[j]=penalty;q.append(row);}}
        q
    }
    fn solve(&self)->Result<(Vec<[f64;Y]>,usize,f64),String>{
        let max=(0..P).map(|j|self.r[j*WIDTH+j].abs()).fold(0.,f64::max);
        let min=(0..P).map(|j|self.r[j*WIDTH+j].abs()).fold(f64::INFINITY,f64::min);
        let tol=max*1e-10;
        let rank=(0..P).filter(|&j|self.r[j*WIDTH+j].abs()>tol).count();
        if max==0.||!max.is_finite()||rank!=P{return Err(format!("QR diagonal rank check {rank}/{P}; no silent regularization"))}
        let mut beta=vec![[0.;Y];P];
        for j in (0..P).rev(){
            for k in 0..Y{
                let mut v=self.r[j*WIDTH+P+k];
                for i in j+1..P{v-=self.r[j*WIDTH+i]*beta[i][k];}
                beta[j][k]=v/self.r[j*WIDTH+j];
            }
        }
        if beta.iter().flatten().any(|v|!v.is_finite()){return Err("nonfinite coefficients".into())}
        Ok((beta,rank,min/max))
    }
}
struct Linear{beta:Vec<[f64;Y]>,mean:f64,sd:f64}
impl Linear{
    fn normalized(&self,x:&[f64;D])->[f64;Y]{
        let mut y=self.beta[0];
        for j in 0..D{for k in 0..Y{y[k]+=x[j]*self.beta[j+1][k]}}
        y
    }
}
impl Predictor for Linear{fn predict(&self,x:&OriginInput)->[f64;Y]{self.normalized(&x.x).map(|v|v*self.sd+self.mean)}}
const CONFIGS:[(&str,f64);7]=[("ols",0.),("ridge-1e-6",1e-6),("ridge-1e-4",1e-4),("ridge-1e-3",1e-3),("ridge-1e-2",1e-2),("ridge-0.1",0.1),("ridge-1",1.)];
pub fn run(data:&Path,id:&str,output:&Path,selection:&str)->Result<(),String>{
    if selection!="all"&&!CONFIGS.iter().any(|(label,_)|*label==selection){return Err("unknown linear configuration".into())}
    let f=load(data,Fold::development(id)?)?;
    let expected=match id{"2019"=>(121849,7455),"2020"=>(129327,7543),"2021"=>(136893,7501),"2022"=>(144417,7639),_=>return Err("annual folds only".into())};
    if (f.train.len(),f.eval.len())!=expected{return Err("frozen cohort mismatch".into())}
    let mut daily=crate::benchmark::Metrics::default();
    for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    let baseline=match id{"2019"=>1.88752763,"2020"=>2.07087759,"2021"=>2.14898785,"2022"=>2.10544725,_=>unreachable!()};
    if (daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{},{},{},{}",f.train.len(),f.eval.len(),f.possible,f.temperature_complete);
    println!("scales,{id},{:?}",f.scales);
    println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let start=Instant::now();let mut qr=Qr::new();
    for r in &f.train{qr.observation(&r.input.x,&r.normalized_target);}
    let qr_seconds=start.elapsed().as_secs_f64();println!("factor-seconds,{id},{qr_seconds:.6}");
    for (label,lambda) in CONFIGS{
        if selection!="all"&&selection!=label{continue}
        let now=Instant::now();
        let (beta,rank,ratio)=match qr.ridge(lambda).solve(){Ok(v)=>v,Err(e)=>{println!("blocked,{id},{label},{e}");continue}};
        let solve_seconds=now.elapsed().as_secs_f64();
        println!("linear-config,{id},{label},{lambda},{rank},{ratio:.12e},{},{solve_seconds:.6}",P*Y);
        let m=Linear{beta,mean:f.target_mean,sd:f.target_sd};
        score(&m,label,&f,&mut out)?;
    }
    out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn data()->Qr{
        let mut q=Qr::new();
        // Symmetric one-hot pairs: intercept orthogonal to all columns, exact known coefficients.
        for j in 0..D{for sign in [-1.,1.]{
            let mut x=[0.;D];x[j]=sign;
            let y=std::array::from_fn(|k|3.+sign*(j+1)as f64*(k+1)as f64/100.);
            q.observation(&x,&y);
        }}q
    }
    #[test]fn exact_coefficients(){
        let (b,rank,ratio)=data().solve().unwrap();assert_eq!(rank,P);assert!(ratio>0.);
        for k in 0..Y{assert!((b[0][k]-3.).abs()<1e-10);for j in 0..D{assert!((b[j+1][k]-(j+1)as f64*(k+1)as f64/100.).abs()<1e-10);}}
        let m=Linear{beta:b,mean:10.,sd:2.};let mut x=[0.;D];x[0]=2.;
        let p=m.predict(&OriginInput{origin:0,x,recent:[0.;Y]});assert!((p[0]-16.04).abs()<1e-10);
    }
    #[test]fn ridge_shrinkage_and_intercept(){
        let q=data();let(b,_,_)=q.ridge(0.1).solve().unwrap();
        let shrink=2./(2.+q.n as f64*0.1);
        for k in 0..Y{assert!((b[0][k]-3.).abs()<1e-10);for j in 0..D{assert!((b[j+1][k]-shrink*(j+1)as f64*(k+1)as f64/100.).abs()<1e-10);}}
    }
    #[test]fn rank_deficiency_rejected(){
        let mut q=Qr::new();for _ in 0..100{q.observation(&[1.;D],&[3.;Y])}
        assert!(q.solve().is_err());assert!(q.ridge(0.01).solve().is_ok());
    }
}
