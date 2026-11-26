//! C004 L10/L11: polynomial and additive hinge-spline ridge.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow,D};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;
const LAMBDAS:[(&str,f64);3]=[("1e-3",0.001),("1e-2",0.01),("0.1",0.1)];
#[derive(Clone,Copy,Debug)]
enum Term{Linear(usize),Product(usize,usize),Hinge(usize,f64)}
impl Term{fn value(self,x:&[f64;D])->f64{match self{Self::Linear(i)=>x[i],Self::Product(i,j)=>x[i]*x[j],Self::Hinge(i,k)=>(x[i]-k).max(0.)}}}
fn terms(rows:&[TrainingRow],family:&str)->Result<Vec<Term>,String>{
    let mut t:Vec<_>=(0..D).map(Term::Linear).collect();
    match family{
        "poly-full"|"poly-recent"=>{
            let first=if family=="poly-full"{0}else{48};
            for i in first..D{for j in i..D{t.push(Term::Product(i,j));}}
        },
        "gam-3"|"gam-7"=>{
            let k=if family=="gam-3"{3}else{7};
            for i in 0..D{
                let mut a:Vec<_>=rows.iter().map(|r|r.input.x[i]).collect();
                a.sort_by(f64::total_cmp);let mut prev=None;
                for j in 1..=k{
                    let v=a[(a.len()-1)*j/(k+1)];
                    if v>a[0]&&v<a[a.len()-1]&&prev!=Some(v){t.push(Term::Hinge(i,v));prev=Some(v);}
                }
            }
        },
        _=>return Err("unknown basis family".into())
    }Ok(t)
}
struct Basis{terms:Vec<Term>,mean:Vec<f64>,sd:Vec<f64>}
impl Basis{
    fn fit(rows:&[TrainingRow],terms:Vec<Term>)->Result<Self,String>{
        if rows.is_empty(){return Err("empty training".into())}
        let n=rows.len()as f64;let mut mean=vec![0.;terms.len()];let mut sq=mean.clone();
        for r in rows{for(j,t)in terms.iter().enumerate(){let x=t.value(&r.input.x);mean[j]+=x;sq[j]+=x*x}}
        for v in &mut mean{*v/=n}
        let sd:Vec<_>=sq.iter().enumerate().map(|(j,v)|if matches!(terms[j],Term::Linear(_)){1.}else{(v/n-mean[j]*mean[j]).max(0.).sqrt()}).collect();
        if mean.iter().chain(sd.iter()).any(|v|!v.is_finite()){return Err("nonfinite basis".into())}
        Ok(Self{terms,mean,sd})
    }
    fn fill(&self,x:&[f64;D],z:&mut[f64]){
        for(j,t)in self.terms.iter().enumerate(){z[j]=if self.sd[j]>1e-10{(t.value(x)-self.mean[j])/self.sd[j]}else{0.}}
    }
}
struct Stats{g:Vec<f64>,xy:Vec<[f64;Y]>,my:[f64;Y],p:usize}
fn accumulate(rows:&[TrainingRow],b:&Basis,limit:Option<f64>)->Result<Stats,String>{
    let p=b.terms.len();let n=rows.len()as f64;
    let mut g=vec![0.;p*p];let mut xy=vec![[0.;Y];p];let mut my=[0.;Y];
    for r in rows{for k in 0..Y{my[k]+=r.normalized_target[k]/n}}
    let mut z=vec![0.;p];let clock=Instant::now();
    for (i,r) in rows.iter().enumerate(){
        if i%512==0&&limit.is_some_and(|s|clock.elapsed().as_secs_f64()>s){return Err("Gram time budget exceeded".into())}
        b.fill(&r.input.x,&mut z);let y:[f64;Y]=std::array::from_fn(|k|r.normalized_target[k]-my[k]);
        for j in 0..p{
            let a=z[j];let row=&mut g[j*p..j*p+j+1];
            for (v,x) in row.iter_mut().zip(&z){*v+=a*x}
            for k in 0..Y{xy[j][k]+=a*y[k]}
        }
    }
    for v in &mut g{*v/=n}
    for r in &mut xy{for v in r{*v/=n}}
    Ok(Stats{g,xy,my,p})
}
fn solve(q:&Stats,lambda:f64)->Result<(Vec<[f64;Y]>,f64),String>{
    let p=q.p;let mut l=q.g.clone();
    let max=(0..p).map(|j|l[j*p+j]+lambda).fold(0.,f64::max);
    for j in 0..p{
        let (prev,rest)=l.split_at_mut(j*p);let row=&mut rest[..p];
        for i in 0..j{
            let other=&prev[i*p..i*p+i];
            let a=row[..i].iter().zip(other).map(|(x,y)|x*y).sum::<f64>();
            row[i]=(row[i]-a)/prev[i*p+i];
        }
        let v=row[j]+lambda-row[..j].iter().map(|v|v*v).sum::<f64>();
        if !v.is_finite()||v<=max*1e-12{return Err("bad ridge Cholesky pivot".into())}
        row[j]=v.sqrt();
    }
    let mut b=q.xy.clone();
    for j in 0..p{for i in 0..j{let a=l[j*p+i];for k in 0..Y{b[j][k]-=a*b[i][k]}}
        for v in &mut b[j]{*v/=l[j*p+j]}}
    for j in (0..p).rev(){for i in j+1..p{let a=l[i*p+j];for k in 0..Y{b[j][k]-=a*b[i][k]}}
        for v in &mut b[j]{*v/=l[j*p+j]}}
    let mut residual=0.0f64;
    for j in 0..p{
        let mut z=[0.;Y];for i in 0..p{
            let a=if j>=i{q.g[j*p+i]}else{q.g[i*p+j]};
            for k in 0..Y{z[k]+=a*b[i][k]}
        }
        for k in 0..Y{residual=residual.max((z[k]+lambda*b[j][k]-q.xy[j][k]).abs())}
    }
    if !residual.is_finite()||residual>1e-7||b.iter().flatten().any(|v|!v.is_finite()){return Err(format!("ridge stationarity residual {residual}"))}
    Ok((b,residual))
}
struct Model<'a>{basis:&'a Basis,b:Vec<[f64;Y]>,my:[f64;Y],mean:f64,sd:f64}
impl Predictor for Model<'_>{fn predict(&self,x:&OriginInput)->[f64;Y]{
    let mut z=vec![0.;self.basis.terms.len()];self.basis.fill(&x.x,&mut z);
    let mut y=self.my;for (a,b)in z.iter().zip(&self.b){for k in 0..Y{y[k]+=a*b[k]}}
    y.map(|v|v*self.sd+self.mean)
}}
pub fn run(data:&Path,id:&str,output:&Path,selection:&str)->Result<(),String>{
    let(family,only)=selection.split_once(':').unwrap_or((selection,"all"));
    if only!="all"&&only.split(',').any(|s|!LAMBDAS.iter().any(|r|r.0==s)){return Err("unknown lambda".into())}
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=crate::benchmark::Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);
    println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let clock=Instant::now();let t=terms(&f.train,family)?;let p=t.len();let b=Basis::fit(&f.train,t)?;
    println!("basis-seconds,{id},{family},{p},{},{:.6}",b.sd.iter().filter(|&&x|x<=1e-10).count(),clock.elapsed().as_secs_f64());
    // Two dense p*p matrices plus coefficient/cross-product buffers, not full expanded row matrix.
    let workspace=16*p*p+8*p*(3*Y+4);
    let q=if workspace>1_500_000_000{Err("basis workspace preflight exceeded".into())}else{
        let start=Instant::now();let sample=512.min(n);
        let probe=accumulate(&f.train[..sample],&b,Some(60.))?;
        drop(probe);
        let estimate=start.elapsed().as_secs_f64()*n as f64/sample as f64;
        println!("basis-preflight,{id},{family},{workspace},{sample},{estimate:.6}");
        if estimate*2.+180.>1400.{Err(format!("conservative Gram preflight estimate {:.3}s exceeds budget",estimate*2.+180.))}
        else{let start=Instant::now();let q=accumulate(&f.train,&b,Some(1100.));println!("gram-seconds,{id},{family},{:.6}",start.elapsed().as_secs_f64());q}
    };
    for(s,lambda)in LAMBDAS{
        if only!="all"&&!only.split(',').any(|v|v==s){continue}
        let label=format!("{family}-{s}");
        let start=Instant::now();match q.as_ref().map_err(|e|e.clone()).and_then(|q|solve(q,lambda)){
            Ok((beta,residual))=>{
                println!("basis-config,{id},{label},{lambda},{p},{},{residual:.12e},{:.6}",(p+1)*Y,start.elapsed().as_secs_f64());
                let model=Model{basis:&b,b:beta,my:q.as_ref().unwrap().my,mean:f.target_mean,sd:f.target_sd};
                score(&model,&label,&f,&mut out)?;
            },
            Err(e)=>println!("blocked,{id},{label},{e}"),
        }
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn fixture()->Vec<TrainingRow>{
        (-3..=3).map(|i|{let mut x=[0.;D];x[79]=i as f64;let y=[2.+x[79]*x[79];Y];TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:y,normalized_target:y}}).collect()
    }
    #[test]fn inventory(){let r=fixture();assert_eq!(terms(&r,"poly-full").unwrap().len(),3320);assert_eq!(terms(&r,"poly-recent").unwrap().len(),608);}
    #[test]fn nonlinear_fit(){
        let r=fixture();let b=Basis::fit(&r,vec![Term::Linear(79),Term::Product(79,79)]).unwrap();
        let q=accumulate(&r,&b,None).unwrap();let(beta,res)=solve(&q,1e-8).unwrap();assert!(res<1e-7);
        let m=Model{basis:&b,b:beta,my:q.my,mean:0.,sd:1.};for row in r{assert!((m.predict(&row.input)[0]-row.target[0]).abs()<1e-6)}
    }
    #[test]fn knots_and_constant_basis(){
        let r=fixture();let t=terms(&r,"gam-3").unwrap();assert_eq!(t.len(),83);
        let b=Basis::fit(&r,vec![Term::Hinge(0,0.),Term::Hinge(79,0.)]).unwrap();
        let mut z=[0.;2];b.fill(&[0.;D],&mut z);assert_eq!(z[0],0.);assert!(z[1]<0.);
    }
}
