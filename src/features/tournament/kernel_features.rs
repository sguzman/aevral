//! C007 K10/K11: random Fourier and Nystrom feature ridge.
use super::{load,score,Fold,TrainingRow,D,Y};
use super::neighbors::{subset,Cached};
use super::kernel_linalg::Cholesky;
use super::kernel_ridge::{rbf,target_mean,checked_solve};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const BUDGET:usize=8192;
const JITTER:f64=1e-8;
const LAMBDA:f64=0.01;
const GUARD:f64=1400.;
#[derive(Clone,Copy)]
struct Recipe{random:bool,length:f64,m:usize}
fn config(label:&str)->Result<Recipe,String>{
    let(random,length,m)=match label{
        "rff-s05-m128"=>(true,0.5,128),"rff-s05-m256"=>(true,0.5,256),"rff-s1-m128"=>(true,1.,128),"rff-s1-m256"=>(true,1.,256),
        "nystrom-s05-m128"=>(false,0.5,128),"nystrom-s05-m256"=>(false,0.5,256),"nystrom-s1-m128"=>(false,1.,128),"nystrom-s1-m256"=>(false,1.,256),
        _=>return Err("unknown kernel feature configuration".into())
    };Ok(Recipe{random,length,m})
}
struct Rng(u64);
impl Rng{
    fn next(&mut self)->u64{
        self.0=self.0.wrapping_add(0x9e3779b97f4a7c15);let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);z^(z>>31)
    }
    fn uniform(&mut self)->f64{((self.next()>>12) as f64+0.5)/4503599627370496.}
    fn normal(&mut self)->f64{(-2.*self.uniform().ln()).sqrt()*(std::f64::consts::TAU*self.uniform()).cos()}
}
enum Map{
    Rff{w:Vec<[f64;D]>,b:Vec<f64>},
    Nystrom{x:Vec<[f64;D]>,factor:Cholesky,length:f64}
}
impl Map{
    fn fit(rows:&[&TrainingRow],r:Recipe,seed:u64,now:Instant)->Result<Self,String>{
        if r.m==0||r.m>rows.len()||r.length<=0.||!r.length.is_finite(){return Err("invalid feature map dimensions".into())}
        if r.random{
            let mut rng=Rng(seed);let sd=1./((D as f64).sqrt()*r.length);let mut w=Vec::with_capacity(r.m);let mut b=Vec::with_capacity(r.m);
            // Frozen draw order: each feature's 80 normal draws (two uniforms each), then phase.
            for _ in 0..r.m{w.push(std::array::from_fn(|_|sd*rng.normal()));b.push(std::f64::consts::TAU*rng.uniform());}
            Ok(Self::Rff{w,b})
        }else{
            let n=rows.len();let m=r.m;let x:Vec<_>=(0..m).map(|i|rows[i*n/m].input.x).collect();let mut a=vec![0.;m*m];
            for i in 0..m{a[i*m+i]=1.+JITTER;for j in 0..i{let k=rbf(&x[i],&x[j],r.length);a[i*m+j]=k;a[j*m+i]=k;}}
            let factor=Cholesky::factor_guarded(&a,m,Some(now))?;
            Ok(Self::Nystrom{x,factor,length:r.length})
        }
    }
    fn transform(&self,q:&[f64;D])->Result<Vec<f64>,String>{
        let z:Vec<f64>=match self{
            Self::Rff{w,b}=>{let scale=(2./w.len() as f64).sqrt();w.iter().zip(b).map(|(w,b)|scale*(w.iter().zip(q).map(|(a,b)|a*b).sum::<f64>()+b).cos()).collect()},
            Self::Nystrom{x,factor,length}=>{let k:Vec<_>=x.iter().map(|v|rbf(v,q,*length)).collect();factor.forward(&k)?}
        };
        if z.iter().any(|v|!v.is_finite()){return Err("nonfinite mapped feature".into())}Ok(z)
    }
}
struct Model{map:Map,mean_x:Vec<f64>,mean_y:[f64;Y],coef:Vec<Vec<f64>>}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>GUARD{Err("runtime guard".into())}else{Ok(())}}
impl Model{
    fn fit(rows:&[&TrainingRow],r:Recipe,seed:u64,now:Instant)->Result<(Self,f64),String>{
        let map=Map::fit(rows,r,seed,now)?;let n=rows.len();let m=r.m;let mean_y=target_mean(rows);let mut mean_x=vec![0.;m];
        let mut z=Vec::with_capacity(n);
        for row in rows{guard(now)?;let v=map.transform(&row.input.x)?;for j in 0..m{mean_x[j]+=v[j]/n as f64;}z.push(v);}
        let mut a=vec![0.;m*m];let mut rhs=vec![vec![0.;m];Y];
        for(row,v)in rows.iter().zip(z.iter_mut()){
            guard(now)?;for j in 0..m{v[j]-=mean_x[j];}
            for i in 0..m{for j in 0..=i{a[i*m+j]+=v[i]*v[j]/n as f64;}for h in 0..Y{rhs[h][i]+=v[i]*(row.normalized_target[h]-mean_y[h])/n as f64;}}
        }
        for i in 0..m{for j in 0..i{a[j*m+i]=a[i*m+j];}a[i*m+i]+=LAMBDA;}
        let factor=Cholesky::factor_guarded(&a,m,Some(now))?;let mut residual:f64=0.;let mut coef=Vec::with_capacity(Y);
        for b in rhs{let(v,e)=checked_solve(&a,m,&factor,&b)?;coef.push(v);residual=residual.max(e);}
        Ok((Self{map,mean_x,mean_y,coef},residual))
    }
    fn predict(&self,q:&[f64;D])->Result<[f64;Y],String>{
        let mut z=self.map.transform(q)?;for(j,v)in z.iter_mut().enumerate(){*v-=self.mean_x[j];}
        let p=std::array::from_fn(|h|self.mean_y[h]+z.iter().zip(&self.coef[h]).map(|(a,b)|a*b).sum::<f64>());
        if p.iter().any(|v|!v.is_finite()){return Err("nonfinite mapped ridge prediction".into())}Ok(p)
    }
    fn diagnostic(&self,rows:&[&TrainingRow],length:f64)->Result<(usize,f64,f64),String>{
        // 64 deterministic pairs, evenly spaced anchors and modular offset; no evaluation labels.
        let n=rows.len();let pairs=n.min(64);let mut ss=0.;let mut max:f64=0.;
        for i in 0..pairs{let a=i*n/pairs;let b=(a+n/3+i*17)%n;
            let x=&rows[a].input.x;let y=&rows[b].input.x;let z=self.map.transform(x)?;let w=self.map.transform(y)?;
            let error=(z.iter().zip(&w).map(|(u,v)|u*v).sum::<f64>()-rbf(x,y,length)).abs();ss+=error*error;max=max.max(error);
        }Ok((pairs,(ss/pairs as f64).sqrt(),max))
    }
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str,seed:u64)->Result<(),String>{
    let r=config(label)?;if ![42,123,2026].contains(&seed)||(!r.random&&seed!=42){return Err("seed outside frozen recipe".into())}
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for row in &f.eval{daily.add(&row.target,&row.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let rows=subset(&f.train,BUDGET);let n=rows.len();let d=r.m;
    if n!=BUDGET||rows.iter().any(|r|r.input.x.iter().chain(&r.normalized_target).any(|v|!v.is_finite())){return Err("invalid training subset".into())}
    let workspace=8*(n*d+3*d*d+3*d*Y+d*(D+8)+m*(Y+1)+n*4);
    let now=Instant::now();let p=n.min(512);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    let timer=Instant::now();let(sm,_)=Model::fit(&small,r,seed,now)?;let fit_probe=timer.elapsed().as_secs_f64();
    let q=m.min(32);let timer=Instant::now();for i in 0..q{let _=std::hint::black_box(sm.predict(&f.eval[i*m/q].input.x)?);}let pred_probe=timer.elapsed().as_secs_f64();drop(sm);
    let estimate=fit_probe*n as f64/p as f64+pred_probe*(m+128) as f64/q as f64;
    println!("kfeat-preflight,{id},{label},{seed},{n},{p},{q},{workspace},{fit_probe:.6},{pred_probe:.6},{estimate:.6}");
    if workspace>2_000_000_000||2.*estimate+180.>GUARD{println!("blocked,{id},{label},preflight workspace or runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
    let timer=Instant::now();let(model,residual)=match Model::fit(&rows,r,seed,now){Ok(v)=>v,Err(e)if e=="runtime guard"=>{println!("blocked,{id},{label},{e}");out.flush().map_err(|e|e.to_string())?;return Ok(())},Err(e)=>return Err(e)};
    let fit=timer.elapsed().as_secs_f64();let(pairs,rmse,max)=model.diagnostic(&rows,r.length)?;let timer=Instant::now();let mut predictions=Vec::with_capacity(m);
    for row in &f.eval{if guard(now).is_err(){println!("blocked,{id},{label},runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}predictions.push((row.input.origin,f.restore(model.predict(&row.input.x)?)));}
    println!("kfeat-kernel-error,{id},{label},{seed},{pairs},{rmse:.12e},{max:.12e}");
    println!("kfeat-config,{id},{label},{seed},{n},{d},{},{},{LAMBDA},{residual:.12e},{fit:.6},{:.6}",if r.random{"rff"}else{"nystrom"},r.length,timer.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;use super::super::OriginInput;
    fn row(v:f64,y:f64)->TrainingRow{let mut x=[0.;D];x[0]=v*(D as f64).sqrt();TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:[y;Y],normalized_target:[y;Y]}}
    #[test]fn splitmix_reference(){let mut r=Rng(0);assert_eq!(r.next(),0xe220a8397b1dcdaf);assert_eq!(r.next(),0x6e789e6aa1b965f4);}
    #[test]fn rff_reproducible(){let d=[row(-1.,1.),row(0.,2.),row(1.,3.)];let rows:Vec<_>=d.iter().collect();let r=Recipe{random:true,length:1.,m:3};let(a,_)=Model::fit(&rows,r,42,Instant::now()).unwrap();let(b,_)=Model::fit(&rows,r,42,Instant::now()).unwrap();let(c,_)=Model::fit(&rows,r,123,Instant::now()).unwrap();let q=row(0.5,0.).input.x;assert_eq!(a.predict(&q).unwrap(),b.predict(&q).unwrap());assert_ne!(a.predict(&q).unwrap(),c.predict(&q).unwrap());}
    #[test]fn nystrom_full_landmark_gram(){let d=[row(-1.,1.),row(0.,2.),row(1.,3.)];let rows:Vec<_>=d.iter().collect();let map=Map::fit(&rows,Recipe{random:false,length:0.5,m:3},42,Instant::now()).unwrap();for a in &d{for b in &d{let z=map.transform(&a.input.x).unwrap();let w=map.transform(&b.input.x).unwrap();assert!((z.iter().zip(w).map(|(x,y)|x*y).sum::<f64>()-rbf(&a.input.x,&b.input.x,0.5)).abs()<2e-8);}}}
    #[test]fn constant_target_intercept(){let d=[row(-1.,3.),row(0.,3.),row(1.,3.)];let rows:Vec<_>=d.iter().collect();for random in [true,false]{let(m,e)=Model::fit(&rows,Recipe{random,length:1.,m:3},42,Instant::now()).unwrap();assert!(e<1e-12);assert_eq!(m.predict(&row(9.,0.).input.x).unwrap(),[3.;Y]);}}
}
