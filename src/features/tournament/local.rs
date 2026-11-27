//! C007 K04: local ridge with unpenalized intercept evaluated at query offset zero.
use super::{load,score,Fold,TrainingRow,D,Y};
use super::neighbors::{nearest,subset,Neighbor,Cached};
use super::kernel_linalg::Cholesky;
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const P:usize=12;
const COORD:[usize;P]=[23,47,65,71,72,73,74,75,76,77,78,79];
const BUDGET:usize=8192;
const GUARD:f64=1400.;
fn local_fit(rows:&[&TrainingRow],ids:&[Neighbor],q:&[f64;D],lambda:f64)->Result<([f64;Y],f64),String>{
    if ids.is_empty()||!lambda.is_finite()||lambda<=0.{return Err("invalid local fit".into())}
    let n=ids.len() as f64;
    let mut mx=[0.;P];let mut my=[0.;Y];
    for id in ids{let r=rows[id.i];for j in 0..P{mx[j]+=(r.input.x[COORD[j]]-q[COORD[j]])/n;}for h in 0..Y{my[h]+=r.normalized_target[h]/n;}}
    let mut gram=[0.;P*P];let mut rhs=[[0.;P];Y];
    for id in ids{let r=rows[id.i];let z:[f64;P]=std::array::from_fn(|j|r.input.x[COORD[j]]-q[COORD[j]]-mx[j]);
        for i in 0..P{for j in 0..=i{gram[i*P+j]+=z[i]*z[j]/n;}for h in 0..Y{rhs[h][i]+=z[i]*(r.normalized_target[h]-my[h])/n;}}
    }
    for i in 0..P{for j in 0..i{gram[j*P+i]=gram[i*P+j];}gram[i*P+i]+=lambda;}
    let c=Cholesky::factor(&gram,P)?;let mut pred=my;let mut max_residual: f64=0.;
    for h in 0..Y{let b=c.solve(&rhs[h])?;
        for i in 0..P{let v=(0..P).map(|j|gram[i*P+j]*b[j]).sum::<f64>();max_residual=max_residual.max((v-rhs[h][i]).abs()/(1.+rhs[h][i].abs()));}
        pred[h]-=mx.iter().zip(&b).map(|(a,b)|a*b).sum::<f64>();
    }
    if max_residual>1e-8||pred.iter().any(|v|!v.is_finite()){return Err(format!("local solve residual/finite failure: {max_residual}"))}
    Ok((pred,max_residual))
}
fn predict(rows:&[&TrainingRow],q:&[f64;D],k:usize,lambda:f64)->Result<([f64;Y],f64),String>{
    let (ids,_,_)=nearest(rows,q,k,None);if ids.len()!=k{return Err("local neighbor count mismatch".into())}local_fit(rows,&ids,q,lambda)
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let(k,lambda)=match label{"local-k64-l01"=>(64,0.01),"local-k64-l1"=>(64,0.1),"local-k128-l01"=>(128,0.01),"local-k128-l1"=>(128,0.1),_=>return Err("unknown local ridge configuration".into())};
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let rows=subset(&f.train,BUDGET);if rows.len()<k||rows.iter().any(|r|r.input.x.iter().chain(&r.normalized_target).any(|v|!v.is_finite())){return Err("invalid reference subset".into())}
    let workspace=rows.len()*std::mem::size_of::<&TrainingRow>()+m*std::mem::size_of::<(i64,[f64;Y])>()+k*std::mem::size_of::<Neighbor>()+8*(3*P*P+Y*P+4*P+2*Y);
    let now=Instant::now();let sample=m.min(32);
    for i in 0..sample{let _=std::hint::black_box(predict(&rows,&f.eval[i*m/sample].input.x,k,lambda)?);}
    let secs=now.elapsed().as_secs_f64();let estimate=secs*m as f64/sample as f64;
    println!("local-preflight,{id},{label},{},{sample},{workspace},{secs:.6},{estimate:.6}",rows.len());
    if workspace>2_000_000_000||2.*estimate+180.>GUARD{println!("blocked,{id},{label},preflight workspace or runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
    let mut predictions=Vec::with_capacity(m);let mut residual: f64=0.;let timer=Instant::now();
    for r in &f.eval{
        if now.elapsed().as_secs_f64()>GUARD{println!("blocked,{id},{label},runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
        let(p,e)=predict(&rows,&r.input.x,k,lambda).map_err(|e|format!("origin {}: {e}",r.input.origin))?;residual=residual.max(e);
        predictions.push((r.input.origin,f.restore(p)));
    }
    if predictions.windows(2).any(|w|w[0].0>=w[1].0){return Err("unordered predictions".into())}
    println!("local-config,{id},{label},{},{k},{lambda},{P},{residual:.12e},{:.6}",rows.len(),timer.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
fn nearest_l1(rows:&[&TrainingRow],q:&[f64;D],k:usize)->Vec<Neighbor>{
    let mut heap=std::collections::BinaryHeap::with_capacity(k);
    for(i,row)in rows.iter().enumerate(){
        let n=Neighbor{d2:row.input.x.iter().zip(q).map(|(a,b)|(a-b).abs()).sum(),i};
        if heap.len()<k{heap.push(n)}else if n<*heap.peek().unwrap(){*heap.peek_mut().unwrap()=n;}
    }heap.into_sorted_vec()
}
fn predict_opt(rows:&[&TrainingRow],q:&[f64;D],k:usize,lambda:f64,l1:bool)->Result<([f64;Y],f64),String>{
    if !l1{return predict(rows,q,k,lambda)}
    let ids=nearest_l1(rows,q,k);if ids.len()!=k{return Err("local neighbor count mismatch".into())}local_fit(rows,&ids,q,lambda)
}
pub(super) fn optimize(f:&super::PreparedFold,rows:&[&TrainingRow],label:&str,k:usize,lambda:f64,l1:bool,out:&mut impl Write)->Result<(),String>{
    if ![64,128,256].contains(&k)||![0.001,0.01,0.1].contains(&lambda)||rows.len()<k{return Err("invalid C012 local ridge configuration".into())}
    if rows.iter().any(|r|r.input.x.iter().chain(&r.normalized_target).any(|v|!v.is_finite())){return Err("invalid reference subset".into())}
    let id=f.fold.id;let m=f.eval.len();
    let workspace=rows.len()*std::mem::size_of::<&TrainingRow>()+m*std::mem::size_of::<(i64,[f64;Y])>()+k*std::mem::size_of::<Neighbor>()+8*(3*P*P+Y*P+4*P+2*Y);
    let now=Instant::now();let sample=rows.len().min(32);
    for i in 0..sample{let _=std::hint::black_box(predict_opt(rows,&rows[i*rows.len()/sample].input.x,k,lambda,l1)?);}
    let secs=now.elapsed().as_secs_f64();let estimate=secs*m as f64/sample as f64;
    println!("local-preflight,{id},{label},{},{sample},{workspace},{secs:.6},{estimate:.6}",rows.len());
    if workspace>2_000_000_000||2.*estimate+180.>GUARD{println!("blocked,{id},{label},preflight workspace or runtime guard");return Ok(())}
    let timer=Instant::now();let mut predictions=Vec::with_capacity(m);let mut residual:f64=0.;
    for row in &f.eval{
        if now.elapsed().as_secs_f64()>GUARD{println!("blocked,{id},{label},runtime guard");return Ok(())}
        let(p,e)=predict_opt(rows,&row.input.x,k,lambda,l1)?;residual=residual.max(e);predictions.push((row.input.origin,f.restore(p)));
    }
    let pred=timer.elapsed().as_secs_f64();
    println!("local-config,{id},{label},{},{k},{lambda},{P},{residual:.12e},{pred:.6}",rows.len());
    score(&Cached{predictions},label,f,out)?;
    println!("optimization-cost,{id},{label},0.000000,{pred:.6},0");Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;use super::super::OriginInput;
    fn row(v:f64,y:f64)->TrainingRow{let mut x=[0.;D];x[23]=v;TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:[y;Y],normalized_target:[y;Y]}}
    #[test]fn local_affine_recovery(){let d=[row(-1.,1.),row(0.,3.),row(1.,5.)];let rows:Vec<_>=d.iter().collect();let mut q=[0.;D];q[23]=0.5;let(p,e)=predict(&rows,&q,3,1e-9).unwrap();assert!((p[0]-4.).abs()<1e-8);assert!(e<1e-12);}
    #[test]fn analytic_ridge_shrinkage(){let d=[row(-1.,1.),row(1.,5.)];let rows:Vec<_>=d.iter().collect();let mut q=[0.;D];q[23]=2.;let(p,_)=predict(&rows,&q,2,1.).unwrap();assert!((p[0]-5.).abs()<1e-12);}
    #[test]fn constant_rank_deficient(){let d=[row(1.,2.),row(1.,6.)];let rows:Vec<_>=d.iter().collect();let(p,_)=predict(&rows,&[0.;D],2,0.01).unwrap();assert_eq!(p,[4.;Y]);}
}
