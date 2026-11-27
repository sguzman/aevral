//! C008 D01/D02: causal Fourier and Haar projections with direct ridge.
use super::{load,score,Fold,TrainingRow,D,Y};
use super::neighbors::{subset,Cached};
use super::kernel_linalg::Cholesky;
use super::kernel_ridge::{checked_solve,target_mean};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const BUDGET:usize=8192;
const GUARD:f64=1400.;
#[derive(Clone,Copy)]
enum Kind{Fourier(usize),Haar(usize)}
#[derive(Clone,Copy)]
struct Recipe{kind:Kind,lambda:f64}
fn config(label:&str)->Result<Recipe,String>{
    let(kind,lambda)=match label{
        "fourier-k3-l001"=>(Kind::Fourier(3),0.001),
        "fourier-k3-l01"=>(Kind::Fourier(3),0.01),
        "fourier-k6-l001"=>(Kind::Fourier(6),0.001),
        "fourier-k6-l01"=>(Kind::Fourier(6),0.01),
        "haar-m16-l001"=>(Kind::Haar(16),0.001),
        "haar-m16-l01"=>(Kind::Haar(16),0.01),
        "haar-m32-l001"=>(Kind::Haar(32),0.001),
        "haar-m32-l01"=>(Kind::Haar(32),0.01),
        _=>return Err("unknown decomposition recipe".into())
    };Ok(Recipe{kind,lambda})
}
fn fourier_basis(k:usize)->Vec<[f64;72]>{
    let mut b=vec![[1./72f64.sqrt();72]];
    let scale=(2./72f64).sqrt();
    for j in 1..=k{
        b.push(std::array::from_fn(|i|scale*(std::f64::consts::TAU*j as f64*i as f64/72.).cos()));
        b.push(std::array::from_fn(|i|scale*(std::f64::consts::TAU*j as f64*i as f64/72.).sin()));
    }b
}
fn haar(x:&[f64])->Vec<f64>{
    debug_assert_eq!(x.len(),64);
    let mut a=x.to_vec();let mut details=Vec::new();
    while a.len()>1{
        let mut next=Vec::with_capacity(a.len()/2);let mut d=Vec::with_capacity(a.len()/2);
        for p in a.chunks_exact(2){next.push((p[0]+p[1])*std::f64::consts::FRAC_1_SQRT_2);d.push((p[0]-p[1])*std::f64::consts::FRAC_1_SQRT_2);}
        details.push(d);a=next;
    }
    for d in details.into_iter().rev(){a.extend(d);}a
}
enum Map{Fourier(Vec<[f64;72]>),Haar(usize)}
impl Map{
    fn new(kind:Kind)->Self{match kind{Kind::Fourier(k)=>Self::Fourier(fourier_basis(k)),Kind::Haar(m)=>Self::Haar(m)}}
    fn dim(&self)->usize{9+match self{Self::Fourier(b)=>b.len(),Self::Haar(m)=>*m}}
    fn transform(&self,x:&[f64;D])->Result<Vec<f64>,String>{
        let mut v:Vec<f64>=match self{
            Self::Fourier(b)=>b.iter().map(|row|row.iter().zip(&x[..72]).map(|(a,b)|a*b).sum()).collect(),
            Self::Haar(m)=>haar(&x[8..72])[..*m].to_vec()
        };
        v.push(x[71]);v.extend_from_slice(&x[72..80]);
        if v.iter().any(|v|!v.is_finite()){return Err("nonfinite decomposition".into())}Ok(v)
    }
}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>GUARD{Err("runtime guard".into())}else{Ok(())}}
// Shared direct readout for later C008 feature maps. No intercept penalty.
pub(super) struct Readout{mean_x:Vec<f64>,mean_y:[f64;Y],coef:Vec<Vec<f64>>}
impl Readout{
    pub(super) fn fit(rows:&[&TrainingRow],z:&[Vec<f64>],lambda:f64,now:Instant)->Result<(Self,f64),String>{
        if rows.is_empty()||rows.len()!=z.len()||!lambda.is_finite()||lambda<=0.{return Err("invalid readout data".into())}
        let n=rows.len();let d=z[0].len();
        if d==0||z.iter().any(|v|v.len()!=d||v.iter().any(|v|!v.is_finite()))||rows.iter().any(|r|r.normalized_target.iter().any(|v|!v.is_finite())){return Err("invalid readout feature/target".into())}
        let mut mean_x=vec![0.;d];let mean_y=target_mean(rows);
        for v in z{for j in 0..d{mean_x[j]+=v[j]/n as f64;}}
        let mut a=vec![0.;d*d];let mut rhs=vec![vec![0.;d];Y];let mut v=vec![0.;d];
        for(row,raw)in rows.iter().zip(z){
            guard(now)?;for j in 0..d{v[j]=raw[j]-mean_x[j];}
            for i in 0..d{for j in 0..=i{a[i*d+j]+=v[i]*v[j]/n as f64;}for h in 0..Y{rhs[h][i]+=v[i]*(row.normalized_target[h]-mean_y[h])/n as f64;}}
        }
        for i in 0..d{for j in 0..i{a[j*d+i]=a[i*d+j];}a[i*d+i]+=lambda;}
        let factor=Cholesky::factor_guarded(&a,d,Some(now))?;let mut residual:f64=0.;let mut coef=Vec::with_capacity(Y);
        for b in rhs{let(v,e)=checked_solve(&a,d,&factor,&b)?;coef.push(v);residual=residual.max(e);}
        Ok((Self{mean_x,mean_y,coef},residual))
    }
    pub(super) fn predict(&self,z:&[f64])->Result<[f64;Y],String>{
        if z.len()!=self.mean_x.len(){return Err("readout dimension mismatch".into())}
        let p:[f64;Y]=std::array::from_fn(|h|self.mean_y[h]+z.iter().zip(&self.mean_x).zip(&self.coef[h]).map(|((x,m),b)|(x-m)*b).sum::<f64>());
        if p.iter().any(|v|!v.is_finite()||v.abs()>1e6){return Err("numerical prediction bound".into())}Ok(p)
    }
}
struct Model{map:Map,readout:Readout}
impl Model{
    fn fit(rows:&[&TrainingRow],r:Recipe,now:Instant)->Result<(Self,f64),String>{
        let map=Map::new(r.kind);let mut z=Vec::with_capacity(rows.len());
        for row in rows{guard(now)?;z.push(map.transform(&row.input.x)?);}
        let(readout,residual)=Readout::fit(rows,&z,r.lambda,now)?;
        Ok((Self{map,readout},residual))
    }
    fn predict(&self,x:&[f64;D])->Result<[f64;Y],String>{self.readout.predict(&self.map.transform(x)?)}
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let r=config(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for row in &f.eval{daily.add(&row.target,&row.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let rows=subset(&f.train,BUDGET);let n=rows.len();let d=Map::new(r.kind).dim();
    let workspace=8*(n*(d+4)+3*d*d+3*d*Y+d*80+m*(Y+1)+1024);
    let now=Instant::now();
    macro_rules! block{($reason:expr)=>{{println!("blocked,{id},{label},{}",$reason);out.flush().map_err(|e|e.to_string())?;return Ok(())}}}
    if workspace>2_000_000_000{block!("preflight workspace guard")}
    let p=n.min(512);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    let timer=Instant::now();let(sm,_)=Model::fit(&small,r,now)?;let fit_probe=timer.elapsed().as_secs_f64();
    let q=n.min(32);let timer=Instant::now();for i in 0..q{std::hint::black_box(sm.predict(&rows[i*n/q].input.x)?);}
    let pred_probe=timer.elapsed().as_secs_f64();drop(sm);
    let estimate=fit_probe*n as f64/p as f64+pred_probe*(m+128) as f64/q as f64;
    println!("decomp-preflight,{id},{label},{n},{p},{q},{workspace},{fit_probe:.6},{pred_probe:.6},{estimate:.6}");
    if 2.*estimate+180.>GUARD{block!("preflight runtime guard")}
    let timer=Instant::now();let(model,residual)=match Model::fit(&rows,r,now){Ok(v)=>v,Err(e)if e=="runtime guard"=>{block!(e)},Err(e)=>return Err(e)};
    let fit=timer.elapsed().as_secs_f64();let timer=Instant::now();let mut predictions=Vec::with_capacity(m);
    for row in &f.eval{
        if guard(now).is_err(){block!("runtime guard")}
        let p=match model.predict(&row.input.x){Ok(p)=>p,Err(e)if e=="numerical prediction bound"=>{block!(e)},Err(e)=>return Err(e)};
        predictions.push((row.input.origin,f.restore(p)));
    }
    println!("decomp-config,{id},{label},{n},{d},{},{},{:.12e},{fit:.6},{:.6}",match r.kind{Kind::Fourier(_)=>"fourier",Kind::Haar(_)=>"haar"},r.lambda,residual,timer.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;use super::super::OriginInput;
    fn dot(x:&[f64],y:&[f64])->f64{x.iter().zip(y).map(|(a,b)|a*b).sum()}
    #[test]fn fourier_orthonormal_and_constant(){
        let b=fourier_basis(6);
        for(i,x)in b.iter().enumerate(){for(j,y)in b.iter().enumerate(){assert!((dot(x,y)-if i==j{1.}else{0.}).abs()<1e-12)}}
        let c=[2.;72];assert!((dot(&b[0],&c)-2.*72f64.sqrt()).abs()<1e-12);for x in &b[1..]{assert!(dot(x,&c).abs()<1e-12)}
    }
    #[test]fn fourier_known_sinusoid(){let b=fourier_basis(6);let x:[f64;72]=std::array::from_fn(|i|(std::f64::consts::TAU*3.*i as f64/72.).sin());for(i,v)in b.iter().enumerate(){assert!((dot(v,&x)-if i==6{6.}else{0.}).abs()<1e-12)}}
    #[test]fn haar_energy_and_reconstruction(){
        let x:Vec<_>=(0..64).map(|i|((i as f64)*0.31).sin()+i as f64/64.).collect();let h=haar(&x);
        assert!((dot(&x,&x)-dot(&h,&h)).abs()<1e-10);
        let mut a=vec![h[0]];let mut at=1;while a.len()<64{let mut next=Vec::new();for(i,v)in a.iter().enumerate(){let d=h[at+i];next.push((v+d)*std::f64::consts::FRAC_1_SQRT_2);next.push((v-d)*std::f64::consts::FRAC_1_SQRT_2);}at+=a.len();a=next;}
        for(i,v)in a.iter().enumerate(){assert!((v-x[i]).abs()<1e-12)}
    }
    #[test]fn haar_step_and_causal_coordinates(){
        let mut x=[0.;D];x[40..72].fill(2.);let map=Map::new(Kind::Haar(16));let v=map.transform(&x).unwrap();
        assert!((v[0]-8.).abs()<1e-12);assert!((v[1]+8.).abs()<1e-12);assert!(v[2..16].iter().all(|a|a.abs()<1e-12));assert_eq!(v[16],2.);
        x[..8].fill(999.);assert_eq!(v,map.transform(&x).unwrap());x[79]=3.;let z=map.transform(&x).unwrap();assert_eq!(&z[..24],&v[..24]);assert_eq!(z[24],3.);
    }
    #[test]fn ridge_centered_intercept_and_shrinkage(){
        let rows:Vec<_>=[-1.,0.,1.].into_iter().map(|v|TrainingRow{input:OriginInput{origin:0,x:[0.;D],recent:[0.;Y]},target:[3.+2.*v;Y],normalized_target:[3.+2.*v;Y]}).collect();
        let refs:Vec<_>=rows.iter().collect();let z=vec![vec![-1.],vec![0.],vec![1.]];
        let(m,e)=Readout::fit(&refs,&z,1./3.,Instant::now()).unwrap();assert!(e<1e-12);assert!((m.predict(&[0.]).unwrap()[0]-3.).abs()<1e-12);assert!((m.predict(&[1.]).unwrap()[0]-(3.+4./3.)).abs()<1e-12);
    }
}
