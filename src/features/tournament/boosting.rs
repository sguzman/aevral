//! C006 T05/T09: deterministic histogram gradient boosting, shared topology.
use super::{load,score,Fold,OriginInput,Predictor,D};
use super::cart::{Data,Node};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;const B:usize=32;const LEAF:usize=32;const RATE:f64=0.1;const GUARD:f64=1200.;
type V=[f64;Y];
struct Bins{cuts:Vec<Vec<f64>>,rows:Vec<[u8;D]>}
impl Bins{
    fn new(x:&[[f64;D]])->Self{
        let mut cuts=Vec::with_capacity(D);
        for f in 0..D{
            let mut v:Vec<_>=x.iter().map(|r|r[f]).collect();v.sort_unstable_by(f64::total_cmp);
            let mut c=Vec::new();
            for b in 1..B{let k=b*v.len()/B;if k>0&&k<v.len()&&v[k-1]<v[v.len()-1]&&c.last().is_none_or(|p|*p<v[k-1]){c.push(v[k-1])}}
            cuts.push(c);
        }
        let rows=x.iter().map(|r|std::array::from_fn(|f|cuts[f].partition_point(|p|*p<r[f])as u8)).collect();
        Self{cuts,rows}
    }
}
fn median(v:&mut[f64])->f64{v.sort_unstable_by(f64::total_cmp);let n=v.len();if n%2==1{v[n/2]}else{0.5*(v[n/2-1]+v[n/2])}}
fn leaf_value(res:&[V],ids:&[usize],quantile:bool)->Vec<f64>{
    if quantile{let mut v=Vec::with_capacity(ids.len());(0..Y).map(|h|{v.clear();v.extend(ids.iter().map(|&i|res[i][h]));median(&mut v)}).collect()}
    else{let mut v=vec![0.;Y];for &i in ids{for h in 0..Y{v[h]+=res[i][h]}}for p in &mut v{*p/=ids.len()as f64}v}
}
fn norm(v:&V)->f64{v.iter().map(|x|x*x).sum()}
fn grow(b:&Bins,res:&[V],grad:&[V],ids:Vec<usize>,depth:usize,leaf:usize,q:bool,now:Instant)->Result<Node,String>{
    if now.elapsed().as_secs_f64()>GUARD{return Err("fit elapsed guard".into())}
    let n=ids.len();if n==0{return Err("empty boosting leaf".into())}
    let terminal=||Node::Leaf(leaf_value(res,&ids,q));
    if depth==0||n<2*leaf{return Ok(terminal())}
    let mut total=[0.;Y];for &i in &ids{for h in 0..Y{total[h]+=grad[i][h]}}
    let parent=norm(&total)/n as f64;
    let mut best:Option<(usize,usize,f64)>=None;
    for f in 0..D{
        if now.elapsed().as_secs_f64()>GUARD{return Err("fit elapsed guard".into())}
        if b.cuts[f].is_empty(){continue}
        let mut cnt=[0usize;B];let mut sums=[[0.;Y];B];
        for &i in &ids{let k=b.rows[i][f]as usize;cnt[k]+=1;for h in 0..Y{sums[k][h]+=grad[i][h]}}
        let mut l=[0.;Y];let mut k=0;
        for t in 0..b.cuts[f].len(){
            k+=cnt[t];for h in 0..Y{l[h]+=sums[t][h]}
            if k<leaf||n-k<leaf{continue}
            let r=l.iter().zip(total).map(|(a,b)|(b-a).powi(2)).sum::<f64>();
            let gain=norm(&l)/k as f64+r/(n-k)as f64-parent;
            if gain>1e-10&&best.as_ref().is_none_or(|p|gain>p.2+1e-12){best=Some((f,t,gain))}
        }
    }
    let Some((f,t,_))=best else{return Ok(terminal())};
    let(l,r):(Vec<_>,Vec<_>)=ids.into_iter().partition(|&i|b.rows[i][f]as usize<=t);
    if l.len()<leaf||r.len()<leaf{return Err("illegal histogram split".into())}
    let left=grow(b,res,grad,l,depth-1,leaf,q,now)?;let right=grow(b,res,grad,r,depth-1,leaf,q,now)?;
    Ok(Node::Split{feature:f,threshold:b.cuts[f][t],left:Box::new(left),right:Box::new(right)})
}
fn gradients(res:&[V],q:bool)->Vec<V>{
    if q{res.iter().map(|r|std::array::from_fn(|h|if r[h]>0.{0.5}else if r[h]<0.{-0.5}else{0.})).collect()}else{res.to_vec()}
}
fn objective(res:&[V],q:bool)->f64{
    res.iter().flat_map(|r|r.iter()).map(|v|if q{0.5*v.abs()}else{v*v}).sum::<f64>()/(res.len()*Y)as f64
}
struct Model{initial:V,trees:Vec<Node>,mean:f64,sd:f64}
impl Predictor for Model{
    fn predict(&self,x:&OriginInput)->V{
        let mut v=self.initial;for t in &self.trees{for(h,p)in t.predict(&x.x).iter().enumerate(){v[h]+=RATE*p}}
        for p in &mut v{*p=*p*self.sd+self.mean}v
    }
}
fn fit(d:&Data,stages:usize,depth:usize,q:bool,now:Instant)->Result<Model,String>{
    let n=d.x.len();if n<LEAF{return Err("sample too small".into())}
    let b=Bins::new(&d.x);let ids:Vec<_>=(0..n).collect();
    let initial:V=leaf_value(&d.y,&ids,q).try_into().map_err(|_|"initial width")?;
    let mut res:Vec<V>=d.y.iter().map(|y|std::array::from_fn(|h|y[h]-initial[h])).collect();
    println!("boost-initial,{:.10}",objective(&res,q));
    let mut trees=Vec::with_capacity(stages);
    for stage in 0..stages{
        let timer=Instant::now();let grad=gradients(&res,q);
        let t=grow(&b,&res,&grad,ids.clone(),depth,LEAF,q,now)?;
        for(i,r)in res.iter_mut().enumerate(){for(h,p)in t.predict(&d.x[i]).iter().enumerate(){r[h]-=RATE*p}}
        trees.push(t);
        let obj=objective(&res,q);if !obj.is_finite(){return Err("nonfinite training objective".into())}
        println!("boost-stage,{},{:.10}",stage+1,obj);
        if stage==0{let elapsed=timer.elapsed().as_secs_f64();let est=now.elapsed().as_secs_f64()+2.*elapsed*(stages-1)as f64;
            println!("boost-preflight,{n},{stages},{depth},{elapsed:.6},{est:.6}");
            if est>GUARD{return Err("preflight estimated fit exceeds 1200s".into())}}
    }Ok(Model{initial,trees,mean:0.,sd:1.})
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let(stages,depth,q)=match label{
        "hist-32-d3"=>(32,3,false),"hist-64-d3"=>(64,3,false),"hist-32-d5"=>(32,5,false),"hist-64-d5"=>(64,5,false),
        "median-boost-32"=>(32,3,true),"median-boost-64"=>(64,3,true),_=>return Err("unknown boosting configuration".into())};
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();let d=Data::from_rows(&f.train);
    match fit(&d,stages,depth,q,now){
        Ok(mut model)=>{model.mean=f.target_mean;model.sd=f.target_sd;let(nodes,leaves)=model.trees.iter().map(Node::counts).fold((0,0),|(a,b),(c,d)|(a+c,b+d));
        println!("boost-config,{id},{label},{},{},{},{},{},{},{},{},{},{:.6}",d.x.len(),stages,depth,B,LEAF,q,nodes,leaves,(leaves+1)*Y,now.elapsed().as_secs_f64());
        score(&model,label,&f,&mut out)?;},
        Err(e) if e.contains("guard")||e.contains("preflight")=>println!("blocked,{id},{label},{e}"),
        Err(e)=>return Err(e)
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn synthetic()->Data{let mut x=vec![[0.;D];128];let mut y=vec![[0.;Y];128];for i in 0..128{x[i][0]=i as f64;y[i].fill(if i<64{1.}else{9.})}Data{x,y}}
    #[test]fn bin_boundaries(){let d=synthetic();let b=Bins::new(&d.x);assert_eq!(b.cuts[0].len(),31);assert!(b.cuts[1].is_empty());for(i,r)in d.x.iter().enumerate(){assert_eq!(b.rows[i][0]as usize,b.cuts[0].iter().filter(|p|**p<r[0]).count())}}
    #[test]fn shared_split(){let d=synthetic();let b=Bins::new(&d.x);let t=grow(&b,&d.y,&d.y,(0..128).collect(),1,32,false,Instant::now()).unwrap();assert_eq!(t.counts(),(3,2));assert_eq!(t.predict(&d.x[0])[0],1.);assert_eq!(t.predict(&d.x[127])[0],9.);}
    #[test]fn median_not_mean(){let res=vec![[1.;Y],[2.;Y],[100.;Y]];let v=leaf_value(&res,&[0,1,2],true);assert_eq!(v[0],2.);assert_eq!(leaf_value(&res,&[0,1,2],false)[0],103./3.);}
    #[test]fn pinball_subgradient(){let r=vec![std::array::from_fn(|h|match h%3{0=>-1.,1=>0.,_=>2.})];let g=gradients(&r,true);assert_eq!(&g[0][..3],&[-0.5,0.,0.5]);}
    #[test]fn constant_and_minleaf(){let d=synthetic();let b=Bins::new(&d.x);let res=vec![[3.;Y];128];let t=grow(&b,&res,&res,(0..128).collect(),3,32,false,Instant::now()).unwrap();assert_eq!(t.counts(),(1,1));let t=grow(&b,&d.y,&d.y,(0..128).collect(),3,65,false,Instant::now()).unwrap();assert_eq!(t.counts(),(1,1));}
    #[test]fn shrinkage_reduces_objective(){let d=synthetic();for q in [false,true]{let m=fit(&d,3,1,q,Instant::now()).unwrap();let mut p=m.initial;for t in &m.trees{for(h,v)in t.predict(&d.x[0]).iter().enumerate(){p[h]+=RATE*v}}assert!((p[0]-1.).abs()<(m.initial[0]-1.).abs());}}
}
