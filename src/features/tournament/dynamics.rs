//! C008 D03-D05: pooled SSA, truncated delay DMD and polynomial EDMD.
use super::{load,score,Fold,TrainingRow,D,Y};
use super::neighbors::{subset,Cached};
use super::latent::eig;
use super::kernel_linalg::Cholesky;
use super::kernel_ridge::checked_solve;
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const GUARD:f64=1400.;
const BUDGET:usize=8192;
#[derive(Clone,Copy)]
enum Recipe{Ssa(usize,usize),Dmd(usize,usize),Edmd(usize,f64)}
fn config(s:&str)->Result<Recipe,String>{Ok(match s{
"ssa-l24-r4"=>Recipe::Ssa(24,4),"ssa-l24-r8"=>Recipe::Ssa(24,8),"ssa-l36-r4"=>Recipe::Ssa(36,4),"ssa-l36-r8"=>Recipe::Ssa(36,8),
"dmd-l12-r4"=>Recipe::Dmd(12,4),"dmd-l12-r8"=>Recipe::Dmd(12,8),"dmd-l24-r4"=>Recipe::Dmd(24,4),"dmd-l24-r8"=>Recipe::Dmd(24,8),
"edmd-l6-l001"=>Recipe::Edmd(6,0.001),"edmd-l6-l01"=>Recipe::Edmd(6,0.01),"edmd-l12-l001"=>Recipe::Edmd(12,0.001),"edmd-l12-l01"=>Recipe::Edmd(12,0.01),
_=>return Err("unknown dynamical recipe".into())})}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>GUARD{Err("runtime guard".into())}else{Ok(())}}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn norm(a:&[f64])->f64{dot(a,a).sqrt()}
fn finite(a:&[f64])->Result<(),String>{if a.iter().any(|v|!v.is_finite()||v.abs()>1e6){Err("numerical rollout bound".into())}else{Ok(())}}
fn leading<const L:usize>(g:&[[f64;L]],r:usize)->Result<(Vec<(f64,[f64;L])>,f64),String>{
    let e=eig(g)?;
    if r==0||r>=L||e[0].0<=0.||e[r-1].0<=1e-10*e[0].0||e[r-1].0-e[r].0<=1e-10*e[0].0{return Err("nonidentifiable retained subspace".into())}
    if e[L-1].0 < -1e-8*e[0].0{return Err("negative Gram eigenvalue".into())}
    let scale=g.iter().flatten().map(|x|x.abs()).fold(0.,f64::max).max(1e-30);
    let residual=e.iter().take(r).map(|(v,u)|(0..L).map(|i|(dot(&g[i],u)-v*u[i]).abs()/scale).fold(0.,f64::max)).fold(0.,f64::max);
    Ok((e.into_iter().take(r).collect(),residual))
}
fn recurrence<const L:usize>(u:&[(f64,[f64;L])])->Result<(Vec<f64>,f64),String>{
    let den=1.-u.iter().map(|(_,v)|v[L-1].powi(2)).sum::<f64>();
    if den<=1e-8{return Err("singular SSA recurrence".into())}
    Ok(((0..L-1).map(|i|u.iter().map(|(_,v)|v[i]*v[L-1]).sum::<f64>()/den).collect(),den))
}
enum Model{Ssa(Vec<f64>),Dmd{u:Vec<Vec<f64>>,a:Vec<Vec<f64>>},Edmd{l:usize,mean:Vec<f64>,out_mean:Vec<f64>,a:Vec<Vec<f64>>}}
struct Diag{pairs:usize,dim:usize,rank:usize,residual:f64,operator:f64,extra:f64}
impl Model{
    fn predict(&self,x:&[f64;D])->Result<([f64;Y],f64),String>{
        let mut p=[0.;Y];let mut max=0f64;
        match self{
            Self::Ssa(r)=>{let mut s=x[72-r.len()..72].to_vec();for y in &mut p{*y=dot(r,&s);finite(&[*y])?;max=max.max(y.abs());s.rotate_left(1);let n=s.len();s[n-1]=*y;}}
            Self::Dmd{u,a}=>{let l=u[0].len();let mut z:Vec<_>=u.iter().map(|v|dot(v,&x[72-l..72])).collect();
                for y in &mut p{z=a.iter().map(|v|dot(v,&z)).collect();finite(&z)?;max=max.max(z.iter().map(|v|v.abs()).fold(0.,f64::max));*y=u.iter().zip(&z).map(|(v,z)|v[l-1]*z).sum();finite(&[*y])?;}}
            Self::Edmd{l,mean,out_mean,a}=>{let mut z=lift(&x[72-l..72]);
                for y in &mut p{let q:Vec<_>=z.iter().zip(mean).map(|(x,m)|x-m).collect();z=a.iter().zip(out_mean).map(|(v,m)|m+dot(v,&q)).collect();finite(&z)?;max=max.max(z.iter().map(|v|v.abs()).fold(0.,f64::max));*y=z[l-1];}}
        }Ok((p,max))
    }
}
fn spectral<const L:usize>(rows:&[&TrainingRow],r:usize,dmd:bool,now:Instant)->Result<(Model,Diag),String>{
    let mut g=vec![[0.;L];L];let mut c=vec![[0.;L];L];let mut pairs=0;
    for row in rows{guard(now)?;let x=&row.input.x;
        for t in 0..(73-L-usize::from(dmd)){let s=&x[t..t+L];pairs+=1;
            for i in 0..L{for j in 0..=i{g[i][j]+=s[i]*s[j];}if dmd{for j in 0..L{c[i][j]+=x[t+i+1]*s[j];}}}
        }
    }
    for i in 0..L{for j in 0..=i{g[i][j]/=pairs as f64;g[j][i]=g[i][j];}for j in 0..L{c[i][j]/=pairs as f64;}}
    let(u,residual)=leading(&g,r)?;
    if !dmd{let(a,den)=recurrence(&u)?;let operator=norm(&a);return Ok((Model::Ssa(a),Diag{pairs,dim:L,rank:r,residual,operator,extra:den}))}
    let a=dmd_matrix(&u,&c);let operator=a.iter().map(|v|dot(v,v)).sum::<f64>().sqrt();
    let u=u.into_iter().map(|(_,v)|v.to_vec()).collect();
    Ok((Model::Dmd{u,a},Diag{pairs,dim:L,rank:r,residual,operator,extra:0.}))
}
fn dmd_matrix<const L:usize>(u:&[(f64,[f64;L])],c:&[[f64;L]])->Vec<Vec<f64>>{
    u.iter().map(|(_,v)|u.iter().map(|(lambda,w)|{let cw:Vec<_>=c.iter().map(|row|dot(row,w)).collect();dot(v,&cw)/lambda}).collect()).collect()
}
// The constant observable is implicit and exactly one; remaining dictionary coordinates preserve manifest order.
fn lift(s:&[f64])->Vec<f64>{let mut z=s.to_vec();for i in 0..s.len(){for j in i..s.len(){z.push(s[i]*s[j]);}}z}
fn edmd(rows:&[&TrainingRow],l:usize,lambda:f64,now:Instant)->Result<(Model,Diag),String>{
    let d=l+l*(l+1)/2;let mut g=vec![0.;d*d];let mut c=vec![0.;d*d];let mut mx=vec![0.;d];let mut my=vec![0.;d];let mut n=0usize;let mut yy=0.;
    // Stream paired moments, with a centered covariance and unpenalized intercept.
    for row in rows{guard(now)?;let x=&row.input.x;let mut s=lift(&x[..l]);for t in 0..72-l{let y=lift(&x[t+1..t+l+1]);n+=1;
        for i in 0..d{mx[i]+=s[i];my[i]+=y[i];yy+=y[i]*y[i];for j in 0..=i{g[i*d+j]+=s[i]*s[j];}for j in 0..d{c[i*d+j]+=y[i]*s[j];}}
        s=y;
    }}
    if n==0{return Err("empty EDMD pairs".into())}
    let nf=n as f64;for i in 0..d{mx[i]/=nf;my[i]/=nf;}
    for i in 0..d{for j in 0..=i{g[i*d+j]=g[i*d+j]/nf-mx[i]*mx[j];g[j*d+i]=g[i*d+j];}for j in 0..d{c[i*d+j]=c[i*d+j]/nf-my[i]*mx[j];}}
    for i in 0..d{g[i*d+i]+=lambda;}
    let factor=Cholesky::factor_guarded(&g,d,Some(now))?;let mut a=Vec::new();let mut solve=0f64;
    for i in 0..d{let(v,e)=checked_solve(&g,d,&factor,&c[i*d..(i+1)*d])?;solve=solve.max(e);a.push(v);}
    let mut mse=yy/nf-dot(&my,&my);
    for i in 0..d{mse-=2.*dot(&a[i],&c[i*d..(i+1)*d]);for j in 0..d{for k in 0..d{mse+=a[i][j]*a[i][k]*(g[j*d+k]-if j==k{lambda}else{0.});}}}
    if !mse.is_finite()||mse < -1e-7*(1.+yy/nf){return Err("invalid EDMD residual".into())}
    let operator=a.iter().map(|v|dot(v,v)).sum::<f64>().sqrt();
    Ok((Model::Edmd{l,mean:mx,out_mean:my,a},Diag{pairs:n,dim:d+1,rank:d,residual:solve,operator,extra:(mse.max(0.)/d as f64).sqrt()}))
}
fn fit(rows:&[&TrainingRow],r:Recipe,now:Instant)->Result<(Model,Diag),String>{match r{
Recipe::Ssa(24,r)=>spectral::<24>(rows,r,false,now),Recipe::Ssa(36,r)=>spectral::<36>(rows,r,false,now),
Recipe::Dmd(12,r)=>spectral::<12>(rows,r,true,now),Recipe::Dmd(24,r)=>spectral::<24>(rows,r,true,now),
Recipe::Edmd(l,v)=>edmd(rows,l,v,now),_=>Err("invalid recipe".into())}}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let r=config(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for row in &f.eval{daily.add(&row.target,&row.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    macro_rules! block{($reason:expr)=>{{println!("blocked,{id},{label},{}",$reason);out.flush().map_err(|e|e.to_string())?;return Ok(())}}}
    let rows=subset(&f.train,BUDGET);let n=rows.len();let d=match r{Recipe::Ssa(l,_)|Recipe::Dmd(l,_)=>l,Recipe::Edmd(l,_)=>1+l+l*(l+1)/2};
    let workspace=8*(10*d*d+16*d+n*4+m*(Y+1)+4096);
    if workspace>2_000_000_000{block!("preflight workspace guard")}
    let now=Instant::now();let p=n.min(256);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    // A small rank-deficient probe must not misclassify the full fit; measure actual moment construction via fit, flag a numerical probe block explicitly.
    let timer=Instant::now();let(sm,_)=match fit(&small,r,now){Ok(v)=>v,Err(e)=>{block!(format!("preflight: {e}"))}};
    let fp=timer.elapsed().as_secs_f64();let q=n.min(32);let timer=Instant::now();
    for i in 0..q{match sm.predict(&rows[i*n/q].input.x){Ok(v)=>{std::hint::black_box(v);},Err(e)=>{block!(format!("preflight: {e}"))}}}
    let pp=timer.elapsed().as_secs_f64();drop(sm);let estimate=fp*n as f64/p as f64+pp*(m+128) as f64/q as f64;
    println!("dyn-preflight,{id},{label},{n},{p},{q},{workspace},{fp:.6},{pp:.6},{estimate:.6}");
    if 2.*estimate+180.>GUARD{block!("preflight runtime guard")}
    let timer=Instant::now();let(model,diag)=match fit(&rows,r,now){Ok(v)=>v,Err(e)=>{block!(e)}};
    let fit_seconds=timer.elapsed().as_secs_f64();let timer=Instant::now();let mut predictions=Vec::with_capacity(m);let mut max=0f64;
    for row in &f.eval{if guard(now).is_err(){block!("runtime guard")}
        let(p,v)=match model.predict(&row.input.x){Ok(v)=>v,Err(e)=>{block!(e)}};max=max.max(v);predictions.push((row.input.origin,f.restore(p)));
    }
    println!("dyn-config,{id},{label},{n},{},{},{},{:.12e},{:.12e},{:.12e},{max:.12e},{fit_seconds:.6},{:.6}",diag.pairs,diag.dim,diag.rank,diag.residual,diag.operator,diag.extra,timer.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
use super::*;use super::super::OriginInput;
fn row(x:[f64;D])->TrainingRow{TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:[0.;Y],normalized_target:[0.;Y]}}
#[test]fn ssa_sine_recurrence(){let x=std::array::from_fn(|i|(i as f64*0.31).sin());let r=row(x);let(m,d)=spectral::<24>(&[&r],2,false,Instant::now()).unwrap();assert_eq!(d.pairs,49);let(p,_)=m.predict(&x).unwrap();for(i,v)in p.iter().enumerate(){assert!((v-((72+i)as f64*0.31).sin()).abs()<1e-8)}}
#[test]fn ssa_singular(){assert!(recurrence(&[(1.,[0.,0.,1.])]).is_err());}
#[test]fn dmd_exponential(){let x=std::array::from_fn(|i|0.98f64.powi(i as i32));let r=row(x);let(m,d)=spectral::<12>(&[&r],1,true,Instant::now()).unwrap();assert_eq!(d.pairs,60);let(p,_)=m.predict(&x).unwrap();for(i,v)in p.iter().enumerate(){assert!((v-0.98f64.powi((72+i)as i32)).abs()<1e-8)}}
#[test]fn dmd_oscillation(){let x=std::array::from_fn(|i|(i as f64*0.31).cos());let r=row(x);let(m,_)=spectral::<12>(&[&r],2,true,Instant::now()).unwrap();let(p,_)=m.predict(&x).unwrap();for(i,v)in p.iter().enumerate(){assert!((v-((72+i)as f64*0.31).cos()).abs()<1e-8)}}
#[test]fn lift_order(){assert_eq!(lift(&[2.,3.,4.]),vec![2.,3.,4.,4.,6.,8.,9.,12.,16.]);}
#[test]fn edmd_affine_identity(){let rr:Vec<_>=(0..5).map(|j|row(std::array::from_fn(|i|j as f64*0.5+i as f64*0.01))).collect();let refs:Vec<_>=rr.iter().collect();let(m,d)=edmd(&refs,1,1e-9,Instant::now()).unwrap();assert_eq!(d.dim,3);assert_eq!(d.pairs,355);let(p,_)=m.predict(&rr[2].input.x).unwrap();for(i,v)in p.iter().enumerate(){assert!((v-(1.+(72+i)as f64*0.01)).abs()<1e-5)}let rr:Vec<_>=(0..5).map(|j|row([j as f64;D])).collect();let refs:Vec<_>=rr.iter().collect();let(m,_)=edmd(&refs,1,1e-9,Instant::now()).unwrap();let(p,_)=m.predict(&rr[2].input.x).unwrap();assert!(p.iter().all(|v|(v-2.).abs()<1e-6));}
}
