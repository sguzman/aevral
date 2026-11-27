//! C008 D06/D07: frozen echo-state and extreme-learning feature maps.
use super::{load,score,Fold,TrainingRow,D,Y};
use super::neighbors::{subset,Cached};
use super::decomposition::Readout;
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const BUDGET:usize=8192;
const GUARD:f64=1400.;
#[derive(Clone,Copy)]
enum Recipe{Esn(usize,f64),Elm(usize,f64)}
fn config(s:&str)->Result<Recipe,String>{Ok(match s{
"esn-m64-leak03"=>Recipe::Esn(64,0.3),"esn-m64-leak1"=>Recipe::Esn(64,1.),
"esn-m128-leak03"=>Recipe::Esn(128,0.3),"esn-m128-leak1"=>Recipe::Esn(128,1.),
"elm-m64-l001"=>Recipe::Elm(64,0.001),"elm-m64-l01"=>Recipe::Elm(64,0.01),
"elm-m128-l001"=>Recipe::Elm(128,0.001),"elm-m128-l01"=>Recipe::Elm(128,0.01),
_=>return Err("unknown random learner recipe".into())})}
struct Rng(u64);
impl Rng{
    fn next(&mut self)->u64{self.0=self.0.wrapping_add(0x9e3779b97f4a7c15);let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);z^(z>>31)}
    fn uniform(&mut self)->f64{(self.next()>>11) as f64/9007199254740992.}
    fn signed(&mut self)->f64{2.*self.uniform()-1.}
}
#[derive(Clone)]
enum Map{Esn{units:Vec<[f64;3]>,leak:f64},Elm{w:Vec<[f64;D]>,b:Vec<f64>}}
impl Map{
    fn new(r:Recipe,seed:u64)->Self{let mut rng=Rng(seed);match r{
        Recipe::Esn(m,leak)=>Self::Esn{units:(0..m).map(|_|[if rng.uniform()<0.5{-1.}else{1.},rng.signed(),rng.signed()]).collect(),leak},
        Recipe::Elm(m,_)=>{let scale=(3./D as f64).sqrt();let mut w=Vec::with_capacity(m);let mut b=Vec::with_capacity(m);for _ in 0..m{w.push(std::array::from_fn(|_|scale*rng.signed()));b.push(rng.signed());}Self::Elm{w,b}}
    }}
    fn dim(&self)->usize{match self{Self::Esn{units,..}=>units.len()+9,Self::Elm{w,..}=>w.len()}}
    fn hidden(&self,x:&[f64;D])->Vec<f64>{match self{
        Self::Esn{units,leak}=>{let m=units.len();let mut h=vec![0.;m];let mut next=vec![0.;m];for &u in &x[..72]{step(units,*leak,&h,u,&mut next);std::mem::swap(&mut h,&mut next);}h}
        Self::Elm{w,b}=>w.iter().zip(b).map(|(w,b)|(w.iter().zip(x).map(|(a,b)|a*b).sum::<f64>()+b).tanh()).collect()
    }}
    fn transform(&self,x:&[f64;D])->Result<(Vec<f64>,f64),String>{
        let mut z=self.hidden(x);let norm=z.iter().map(|v|v*v).sum::<f64>().sqrt();
        if z.iter().any(|v|!v.is_finite()||v.abs()>1.+1e-12){return Err("invalid hidden activation".into())}
        if matches!(self,Self::Esn{..}){z.push(x[71]);z.extend_from_slice(&x[72..80]);}
        Ok((z,norm))
    }
}
fn step(units:&[[f64;3]],leak:f64,h:&[f64],u:f64,out:&mut[f64]){
    let m=units.len();for(i,&[sign,w,b])in units.iter().enumerate(){out[i]=(1.-leak)*h[i]+leak*(0.8*sign*h[(i+m-1)%m]+w*u+b).tanh();}
}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>GUARD{Err("runtime guard".into())}else{Ok(())}}
struct Model{map:Map,readout:Readout}
struct Diag{residual:f64,mean_norm:f64,max_norm:f64,variance:f64}
impl Model{
    fn fit(rows:&[&TrainingRow],r:Recipe,seed:u64,now:Instant)->Result<(Self,Diag),String>{
        let map=Map::new(r,seed);let lambda=match r{Recipe::Esn(..)=>0.01,Recipe::Elm(_,l)=>l};
        let m=match r{Recipe::Esn(m,_)|Recipe::Elm(m,_)=>m};let mut z=Vec::with_capacity(rows.len());
        let(mut mean_norm,mut max_norm)=(0.,0f64);let mut sums=vec![0.;m];let mut sq=vec![0.;m];
        for row in rows{guard(now)?;let(v,norm)=map.transform(&row.input.x)?;mean_norm+=norm;max_norm=max_norm.max(norm);for j in 0..m{sums[j]+=v[j];sq[j]+=v[j]*v[j];}z.push(v);}
        let n=rows.len()as f64;let variance=(0..m).map(|j|(sq[j]/n-(sums[j]/n).powi(2)).max(0.)).sum::<f64>()/m as f64;
        let(readout,residual)=Readout::fit(rows,&z,lambda,now)?;
        Ok((Self{map,readout},Diag{residual,mean_norm:mean_norm/n,max_norm,variance}))
    }
    fn predict(&self,x:&[f64;D])->Result<([f64;Y],f64),String>{let(z,n)=self.map.transform(x)?;Ok((self.readout.predict(&z)?,n))}
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str,seed:u64)->Result<(),String>{
    let r=config(label)?;if ![42,123,2026].contains(&seed){return Err("seed outside frozen recipe".into())}
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for row in &f.eval{daily.add(&row.target,&row.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    macro_rules! block{($reason:expr)=>{{println!("blocked,{id},{label},{}",$reason);out.flush().map_err(|e|e.to_string())?;return Ok(())}}}
    let rows=subset(&f.train,BUDGET);let n=rows.len();let d=Map::new(r,seed).dim();let workspace=8*(n*(d+4)+3*d*d+3*d*Y+d*84+m*(Y+1)+4096);
    if workspace>2_000_000_000{block!("preflight workspace guard")}
    let now=Instant::now();let p=n.min(512);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    let timer=Instant::now();let(sm,_)=Model::fit(&small,r,seed,now)?;let fp=timer.elapsed().as_secs_f64();
    let q=n.min(32);let timer=Instant::now();for i in 0..q{std::hint::black_box(sm.predict(&rows[i*n/q].input.x)?);}
    let pp=timer.elapsed().as_secs_f64();drop(sm);let estimate=fp*n as f64/p as f64+pp*(m+128) as f64/q as f64;
    println!("random-preflight,{id},{label},{seed},{n},{p},{q},{workspace},{fp:.6},{pp:.6},{estimate:.6}");
    if 2.*estimate+180.>GUARD{block!("preflight runtime guard")}
    let timer=Instant::now();let(model,diag)=match Model::fit(&rows,r,seed,now){Ok(v)=>v,Err(e)if e=="runtime guard"=>{block!(e)},Err(e)=>return Err(e)};
    let fit=timer.elapsed().as_secs_f64();let timer=Instant::now();let mut predictions=Vec::with_capacity(m);let mut max=0f64;
    for row in &f.eval{if guard(now).is_err(){block!("runtime guard")}
        let(p,v)=match model.predict(&row.input.x){Ok(v)=>v,Err(e)if e=="numerical prediction bound"=>{block!(e)},Err(e)=>return Err(e)};max=max.max(v);predictions.push((row.input.origin,f.restore(p)));}
    let(hidden,lambda,leak,contraction)=match r{Recipe::Esn(m,l)=>(m,0.01,l,1.-0.2*l),Recipe::Elm(m,l)=>(m,l,0.,0.)};
    println!("random-config,{id},{label},{seed},{n},{hidden},{d},{lambda},{leak},{contraction},{:.12e},{:.12e},{:.12e},{:.12e},{max:.12e},{fit:.6},{:.6}",diag.residual,diag.mean_norm,diag.max_norm,diag.variance,timer.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;use super::super::OriginInput;
    #[test]fn splitmix_high53(){let mut r=Rng(0);assert_eq!(r.next(),0xe220a8397b1dcdaf);assert_eq!(r.next(),0x6e789e6aa1b965f4);let mut r=Rng(0);assert_eq!(r.uniform(),(0xe220a8397b1dcdafu64>>11)as f64/9007199254740992.);}
    #[test]fn esn_reset_reproducibility(){let x=[0.25;D];let a=Map::new(Recipe::Esn(64,0.3),42);let b=Map::new(Recipe::Esn(64,0.3),42);let c=Map::new(Recipe::Esn(64,0.3),123);let h=a.hidden(&x);a.hidden(&[2.;D]);assert_eq!(h,a.hidden(&x));assert_eq!(h,b.hidden(&x));assert_ne!(h,c.hidden(&x));}
    #[test]fn simultaneous_signed_ring(){let units=[[1.,0.,0.],[-1.,0.,0.],[1.,0.,0.]];let h=[0.1,0.2,0.3];let mut out=[0.;3];step(&units,1.,&h,0.,&mut out);assert!((out[0]-0.24f64.tanh()).abs()<1e-12);assert!((out[1]-(-0.08f64).tanh()).abs()<1e-12);assert!((out[2]-0.16f64.tanh()).abs()<1e-12);}
    #[test]fn ring_contraction_and_norm(){let Map::Esn{units,leak}=Map::new(Recipe::Esn(128,0.3),42)else{panic!()};let a:Vec<_>=(0..128).map(|i|(i as f64).sin()).collect();let b=vec![0.;128];let(mut x,mut y)=(vec![0.;128],vec![0.;128]);step(&units,leak,&a,0.2,&mut x);step(&units,leak,&b,0.2,&mut y);let d=x.iter().zip(y).map(|(x,y)|(x-y).powi(2)).sum::<f64>().sqrt();let n=a.iter().map(|v|v*v).sum::<f64>().sqrt();assert!(d<=(1.-0.2*leak)*n+1e-12);assert!(x.iter().all(|v|v.abs()<=1.));}
    #[test]fn elm_formula_and_seed(){let x=[0.2;D];let a=Map::new(Recipe::Elm(64,0.01),42);let b=Map::new(Recipe::Elm(64,0.01),42);assert_eq!(a.hidden(&x),b.hidden(&x));assert_ne!(a.hidden(&x),Map::new(Recipe::Elm(64,0.01),123).hidden(&x));let Map::Elm{w,b}=&a else{panic!()};assert!((a.hidden(&x)[0]-(w[0].iter().sum::<f64>()*0.2+b[0]).tanh()).abs()<1e-12);}
    #[test]fn readout_recovery_and_frozen_maps(){for r in [Recipe::Esn(64,0.3),Recipe::Elm(64,0.001)]{let map=Map::new(r,42);let rows:Vec<_>=(0..80).map(|i|{let x=std::array::from_fn(|j|((i*17+j*13)as f64*0.07).sin());let y=2.+3.*map.hidden(&x)[0];TrainingRow{input:OriginInput{origin:i as i64,x,recent:[0.;Y]},target:[y;Y],normalized_target:[y;Y]}}).collect();let refs:Vec<_>=rows.iter().collect();let z:Vec<_>=refs.iter().map(|r|map.transform(&r.input.x).unwrap().0).collect();let(rd,e)=Readout::fit(&refs,&z,1e-9,Instant::now()).unwrap();assert!(e<1e-8);for(row,v)in refs.iter().zip(z){assert!((rd.predict(&v).unwrap()[0]-row.target[0]).abs()<1e-4);assert_eq!(v,map.transform(&row.input.x).unwrap().0);}}}
}
