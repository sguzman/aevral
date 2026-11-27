//! C006 T01: exact numeric CART, shared or horizon-specific outputs.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow,D};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;const SAMPLE:usize=24576;const GUARD:f64=1200.;
pub(super) struct Data{pub x:Vec<[f64;D]>,pub y:Vec<[f64;Y]>}
impl Data{
    pub(super) fn from_rows(rows:&[TrainingRow])->Self{
        let n=rows.len().min(SAMPLE);let mut x=Vec::with_capacity(n);let mut y=Vec::with_capacity(n);
        for i in 0..n{let r=&rows[i*rows.len()/n];x.push(r.input.x);y.push(r.normalized_target)}
        Self{x,y}
    }
}
#[derive(Clone,Copy)]
pub(super) enum Outputs{Shared,Scalar(usize)}
impl Outputs{
    fn n(self)->usize{match self{Self::Shared=>Y,Self::Scalar(_)=>1}}
    fn at(self,j:usize)->usize{match self{Self::Shared=>j,Self::Scalar(h)=>h}}
}
#[derive(Debug)]
pub(super) enum Node{Leaf(Vec<f64>),Split{feature:usize,threshold:f64,left:Box<Node>,right:Box<Node>}}
impl Node{
    pub(super) fn predict(&self,x:&[f64;D])->&[f64]{
        match self{Self::Leaf(v)=>v,Self::Split{feature,threshold,left,right}=>if x[*feature]<=*threshold{left.predict(x)}else{right.predict(x)}}
    }
    pub(super) fn counts(&self)->(usize,usize){match self{Self::Leaf(_)=>(1,1),Self::Split{left,right,..}=>{let(a,b)=left.counts();let(c,d)=right.counts();(a+c+1,b+d)}}}
}
#[derive(Clone,Copy,Debug)]
struct Split{feature:usize,threshold:f64,gain:f64}
fn sums(data:&Data,ids:&[usize],out:Outputs)->(Vec<f64>,f64){
    let mut sum=vec![0.;out.n()];let mut sq=0.;
    for &i in ids{for j in 0..out.n(){let y=data.y[i][out.at(j)];sum[j]+=y;sq+=y*y}}
    (sum,sq)
}
fn norm(v:&[f64])->f64{v.iter().map(|x|x*x).sum()}
fn split(data:&Data,ids:&[usize],out:Outputs,leaf:usize,total:&[f64],now:Instant)->Result<Option<Split>,String>{
    let n=ids.len();let parent=norm(total)/n as f64;
    let mut best:Option<Split>=None;let mut order=ids.to_vec();let mut left=vec![0.;out.n()];
    for f in 0..D{
        if now.elapsed().as_secs_f64()>GUARD{return Err("fit elapsed guard".into())}
        order.sort_unstable_by(|&i,&j|data.x[i][f].total_cmp(&data.x[j][f]).then(i.cmp(&j)));
        left.fill(0.);
        for k in 1..n{
            let i=order[k-1];for j in 0..out.n(){left[j]+=data.y[i][out.at(j)]}
            if k<leaf||n-k<leaf{continue}
            let a=data.x[i][f];let b=data.x[order[k]][f];if a>=b{continue}
            let right=left.iter().zip(total).map(|(l,t)|(t-l).powi(2)).sum::<f64>();
            let gain=norm(&left)/k as f64+right/(n-k)as f64-parent;
            if gain>1e-10&&best.as_ref().is_none_or(|s|gain>s.gain+1e-12){best=Some(Split{feature:f,threshold:a,gain})}
        }
    }Ok(best)
}
pub(super) fn tree(data:&Data,ids:Vec<usize>,out:Outputs,depth:usize,leaf:usize,now:Instant)->Result<Node,String>{
    if ids.is_empty(){return Err("empty node".into())}
    if now.elapsed().as_secs_f64()>GUARD{return Err("fit elapsed guard".into())}
    let(total,sq)=sums(data,&ids,out);let n=ids.len();let mean=||Node::Leaf(total.iter().map(|v|v/n as f64).collect());
    if depth==0||n<2*leaf||sq-norm(&total)/n as f64<=1e-10{return Ok(mean())}
    let Some(s)=split(data,&ids,out,leaf,&total,now)?else{return Ok(mean())};
    let(l,r):(Vec<_>,Vec<_>)=ids.into_iter().partition(|&i|data.x[i][s.feature]<=s.threshold);
    if l.len()<leaf||r.len()<leaf{return Err("split violates leaf size".into())}
    Ok(Node::Split{feature:s.feature,threshold:s.threshold,left:Box::new(tree(data,l,out,depth-1,leaf,now)?),right:Box::new(tree(data,r,out,depth-1,leaf,now)?)})
}
struct Model{trees:Vec<Node>,shared:bool,mean:f64,sd:f64}
impl Predictor for Model{
    fn predict(&self,x:&OriginInput)->[f64;Y]{
        if self.shared{std::array::from_fn(|h|self.trees[0].predict(&x.x)[h]*self.sd+self.mean)}
        else{std::array::from_fn(|h|self.trees[h].predict(&x.x)[0]*self.sd+self.mean)}
    }
}
fn fit(data:&Data,shared:bool,depth:usize,leaf:usize,now:Instant)->Result<Model,String>{
    let n=data.x.len();let ids:Vec<_>=(0..n).collect();if n<leaf{return Err("sample too small".into())}
    let out=if shared{Outputs::Shared}else{Outputs::Scalar(0)};
    let (sum,_)=sums(data,&ids,out);let timer=Instant::now();
    let _=split(data,&ids,out,leaf,&sum,now)?;
    let root=timer.elapsed().as_secs_f64();let count=if shared{1}else{Y};
    // Root split work upper-scale estimate: each depth processes at most n rows.
    // 2x headroom for allocations and irregular node shapes, plus measured root.
    let estimate=root*(1.+2.*depth as f64*count as f64);
    println!("cart-preflight,{n},{depth},{leaf},{count},{root:.6},{estimate:.6}");
    if estimate>GUARD{return Err("preflight estimated fit exceeds 1200s".into())}
    let mut trees=Vec::with_capacity(count);for h in 0..count{
        trees.push(tree(data,ids.clone(),if shared{Outputs::Shared}else{Outputs::Scalar(h)},depth,leaf,now)?);
    }Ok(Model{trees,shared,mean:0.,sd:1.})
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let(shared,depth,leaf)=match label{
        "cart-shared-d4-l32"=>(true,4,32),"cart-shared-d4-l128"=>(true,4,128),
        "cart-shared-d8-l32"=>(true,8,32),"cart-shared-d8-l128"=>(true,8,128),
        "cart-lead-d4-l32"=>(false,4,32),"cart-lead-d8-l32"=>(false,8,32),_=>return Err("unknown CART configuration".into())};
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();let d=Data::from_rows(&f.train);
    match fit(&d,shared,depth,leaf,now){
        Ok(mut model)=>{model.mean=f.target_mean;model.sd=f.target_sd;let(nodes,leaves)=model.trees.iter().map(Node::counts).fold((0,0),|(a,b),(c,d)|(a+c,b+d));
            let scalars=leaves*if shared{Y}else{1};
            println!("cart-config,{id},{label},{},{},{},{},{},{},{},{:.6}",d.x.len(),model.trees.len(),depth,leaf,nodes,leaves,scalars,now.elapsed().as_secs_f64());
            score(&model,label,&f,&mut out)?;
        },Err(e)=>println!("blocked,{id},{label},{e}"),
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn synthetic()->Data{let mut x=vec![[0.;D];8];let mut y=vec![[0.;Y];8];for i in 0..8{x[i][0]=i as f64;y[i].fill(if i<4{1.}else{9.});y[i][1]=2.*y[i][0]}Data{x,y}}
    #[test]fn exact_split(){let d=synthetic();let t=tree(&d,(0..8).collect(),Outputs::Shared,1,2,Instant::now()).unwrap();assert_eq!(t.counts(),(3,2));assert_eq!(t.predict(&d.x[0])[0],1.);assert_eq!(t.predict(&d.x[7])[1],18.);}
    #[test]fn scalar_output(){let d=synthetic();let t=tree(&d,(0..8).collect(),Outputs::Scalar(1),1,2,Instant::now()).unwrap();assert_eq!(t.predict(&d.x[7]),&[18.]);}
    #[test]fn minimum_leaf(){let d=synthetic();let t=tree(&d,(0..8).collect(),Outputs::Shared,8,5,Instant::now()).unwrap();assert_eq!(t.counts(),(1,1));assert_eq!(t.predict(&d.x[0])[0],5.);}
    #[test]fn constant_target(){let mut d=synthetic();d.y.fill([3.;Y]);let t=tree(&d,(0..8).collect(),Outputs::Shared,8,1,Instant::now()).unwrap();assert_eq!(t.counts(),(1,1));}
    #[test]fn identical_features(){let mut d=synthetic();d.x.fill([0.;D]);let t=tree(&d,(0..8).collect(),Outputs::Shared,8,1,Instant::now()).unwrap();assert_eq!(t.counts(),(1,1));}
}
