//! C006 T02-T04, shared-output bootstrapped CART and randomized trees.
use super::{load,score,Fold,OriginInput,Predictor,D};
use super::cart::{Data,Node,Outputs,Split,norm,sums,split_features};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;const DEPTH:usize=8;const LEAF:usize=32;const GUARD:f64=1200.;
#[derive(Clone)]
struct Rng(u64);
impl Rng{
    fn next(&mut self)->u64{
        self.0=self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);z^(z>>31)
    }
    fn index(&mut self,n:usize)->usize{
        let n=n as u64;let threshold=n.wrapping_neg()%n;
        loop{let x=self.next();if x>=threshold{return (x%n)as usize}}
    }
    fn uniform(&mut self)->f64{(self.next()>>11)as f64*(1./9007199254740992.)}
}
#[derive(Clone,Copy,Debug)]
struct Config{trees:usize,mtry:usize,bootstrap:bool,random_threshold:bool}
fn config(label:&str)->Result<Config,String>{
    let parts:Vec<_>=label.split('-').collect();
    let(trees,mtry,bootstrap,random_threshold)=match parts.as_slice(){
        ["bag",n]=>{let n=n.parse::<usize>().map_err(|_|"invalid trees")?;(n,D,true,false)},
        [family,n,m] if *family=="rf"||*family=="extra"=>{let n=n.parse::<usize>().map_err(|_|"invalid trees")?;let m=m.parse::<usize>().map_err(|_|"invalid features")?;if m!=9&&m!=27{return Err("invalid mtry".into())}(n,m,*family=="rf",*family=="extra")},
        _=>return Err("unknown forest configuration".into())
    };
    if trees!=16&&trees!=32{return Err("invalid tree count".into())}
    Ok(Config{trees,mtry,bootstrap,random_threshold})
}
fn features(m:usize,rng:&mut Rng)->Vec<usize>{
    let mut f:Vec<_>=(0..D).collect();if m<D{for i in 0..m{let j=i+rng.index(D-i);f.swap(i,j)}f.truncate(m);f.sort_unstable()}f
}
fn samples(n:usize,bootstrap:bool,rng:&mut Rng)->Vec<usize>{
    if bootstrap{(0..n).map(|_|rng.index(n)).collect()}else{(0..n).collect()}
}
fn random_split(data:&Data,ids:&[usize],total:&[f64],fs:&[usize],leaf:usize,rng:&mut Rng,now:Instant)->Result<Option<Split>,String>{
    let n=ids.len();let parent=norm(total)/n as f64;let mut best:Option<Split>=None;
    for &f in fs{
        if now.elapsed().as_secs_f64()>GUARD{return Err("fit elapsed guard".into())}
        let(lo,hi)=ids.iter().fold((f64::INFINITY,f64::NEG_INFINITY),|(lo,hi),&i|(lo.min(data.x[i][f]),hi.max(data.x[i][f])));
        if lo>=hi{continue}
        let threshold=lo+(hi-lo)*rng.uniform();
        let mut l=[0.;Y];let mut k=0;
        for &i in ids{if data.x[i][f]<=threshold{k+=1;for h in 0..Y{l[h]+=data.y[i][h]}}}
        if k<leaf||n-k<leaf{continue}
        let right=l.iter().zip(total).map(|(a,b)|(b-a).powi(2)).sum::<f64>();
        let gain=norm(&l)/k as f64+right/(n-k)as f64-parent;
        if gain>1e-10&&best.as_ref().is_none_or(|s|gain>s.gain+1e-12){best=Some(Split{feature:f,threshold,gain})}
    }Ok(best)
}
fn grow(data:&Data,ids:Vec<usize>,cfg:Config,depth:usize,leaf:usize,rng:&mut Rng,now:Instant)->Result<Node,String>{
    if now.elapsed().as_secs_f64()>GUARD{return Err("fit elapsed guard".into())}
    let n=ids.len();if n==0{return Err("empty node".into())}
    let(total,sq)=sums(data,&ids,Outputs::Shared);
    let mean=||Node::Leaf(total.iter().map(|v|v/n as f64).collect());
    if depth==0||n<2*leaf||sq-norm(&total)/n as f64<=1e-10{return Ok(mean())}
    let fs=features(cfg.mtry,rng);
    let s=if cfg.random_threshold{random_split(data,&ids,&total,&fs,leaf,rng,now)?}else{split_features(data,&ids,Outputs::Shared,leaf,&total,now,&fs)?};
    let Some(s)=s else{return Ok(mean())};
    let(l,r):(Vec<_>,Vec<_>)=ids.into_iter().partition(|&i|data.x[i][s.feature]<=s.threshold);
    if l.len()<leaf||r.len()<leaf{return Err("split violates leaf size".into())}
    let left=grow(data,l,cfg,depth-1,leaf,rng,now)?;let right=grow(data,r,cfg,depth-1,leaf,rng,now)?;
    Ok(Node::Split{feature:s.feature,threshold:s.threshold,left:Box::new(left),right:Box::new(right)})
}
struct Forest{trees:Vec<Node>,mean:f64,sd:f64}
impl Predictor for Forest{
    fn predict(&self,x:&OriginInput)->[f64;Y]{
        let mut v=[0.;Y];for t in &self.trees{for(h,p)in t.predict(&x.x).iter().enumerate(){v[h]+=p}}
        for p in &mut v{*p=*p/self.trees.len()as f64*self.sd+self.mean}v
    }
}
fn fit(d:&Data,cfg:Config,seed:u64,now:Instant)->Result<Forest,String>{
    if d.x.len()<LEAF{return Err("sample too small".into())}
    let mut rng=Rng(seed);let mut trees=Vec::with_capacity(cfg.trees);
    for i in 0..cfg.trees{
        let timer=Instant::now();let ids=samples(d.x.len(),cfg.bootstrap,&mut rng);
        trees.push(grow(d,ids,cfg,DEPTH,LEAF,&mut rng,now)?);
        if i==0{let t=timer.elapsed().as_secs_f64();let est=now.elapsed().as_secs_f64()+2.*t*(cfg.trees-1)as f64;
            println!("forest-preflight,{},{},{},{:.6},{:.6}",d.x.len(),cfg.trees,cfg.mtry,t,est);
            if est>GUARD{return Err("preflight estimated fit exceeds 1200s".into())}
        }
    }Ok(Forest{trees,mean:0.,sd:1.})
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str,seed:u64)->Result<(),String>{
    if ![42,123,2026].contains(&seed){return Err("seed not in manifest".into())}
    let cfg=config(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();let d=Data::from_rows(&f.train);
    match fit(&d,cfg,seed,now){
        Ok(mut model)=>{model.mean=f.target_mean;model.sd=f.target_sd;let(nodes,leaves)=model.trees.iter().map(Node::counts).fold((0,0),|(a,b),(c,d)|(a+c,b+d));
        println!("forest-config,{id},{label},{seed},{},{},{},{},{},{},{},{},{},{},{:.6}",d.x.len(),cfg.trees,cfg.mtry,cfg.bootstrap,cfg.random_threshold,DEPTH,LEAF,nodes,leaves,leaves*Y,now.elapsed().as_secs_f64());
        score(&model,label,&f,&mut out)?;},
        Err(e) if e.contains("guard")||e.contains("preflight")=>println!("blocked,{id},{label},{e}"),
        Err(e)=>return Err(e)
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn sample_reproducible(){let a=samples(100,true,&mut Rng(42));let b=samples(100,true,&mut Rng(42));assert_eq!(a,b);assert_ne!(a,samples(100,true,&mut Rng(123)));assert!(a.iter().all(|&i|i<100));assert!(a.iter().copied().collect::<std::collections::BTreeSet<_>>().len()<100);assert_eq!(samples(100,false,&mut Rng(42)),(0..100).collect::<Vec<_>>());}
    #[test]fn features_unique(){let f=features(9,&mut Rng(42));assert_eq!(f.len(),9);assert!(f.windows(2).all(|w|w[0]<w[1]));assert!(f[8]<D);}
    fn synthetic()->Data{let mut x=vec![[0.;D];100];let mut y=vec![[0.;Y];100];for i in 0..100{for f in 0..D{x[i][f]=i as f64}y[i].fill(if i<50{1.}else{9.})}Data{x,y}}
    #[test]fn extra_reproducible(){let d=synthetic();let c=config("extra-16-9").unwrap();let a=grow(&d,(0..100).collect(),c,2,5,&mut Rng(42),Instant::now()).unwrap();let b=grow(&d,(0..100).collect(),c,2,5,&mut Rng(42),Instant::now()).unwrap();assert_eq!(format!("{a:?}"),format!("{b:?}"));assert!(a.counts().1>1);}
    #[test]fn extra_legal_threshold(){let d=synthetic();let(total,_)=sums(&d,&(0..100).collect::<Vec<_>>(),Outputs::Shared);let s=random_split(&d,&(0..100).collect::<Vec<_>>(),&total,&(0..D).collect::<Vec<_>>(),20,&mut Rng(42),Instant::now()).unwrap().unwrap();let l=d.x.iter().filter(|x|x[s.feature]<=s.threshold).count();assert!((20..=80).contains(&l));assert!(s.threshold>=0.&&s.threshold<99.);}
    #[test]fn bootstrap_exact_split(){let d=synthetic();let c=config("bag-16").unwrap();let t=grow(&d,(0..100).collect(),c,1,10,&mut Rng(42),Instant::now()).unwrap();assert_eq!(t.predict(&d.x[0])[0],1.);assert_eq!(t.predict(&d.x[99])[0],9.);}
}
