//! C007 K01-K03: exact neighbor search within the frozen chronological reference subset.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow,D,Y};
use crate::benchmark::Metrics;
use std::{cmp::Ordering,collections::BinaryHeap,io::{BufWriter,Write},path::Path,time::Instant};

const BUDGET:usize=8192;
const GUARD:f64=1400.;
#[derive(Clone,Copy,Debug,PartialEq)]
enum Weight{Uniform,Inverse,Gaussian}
#[derive(Clone,Copy,Debug)]
struct Config{k:usize,weight:Weight,season:Option<u32>}
fn config(label:&str)->Result<Config,String>{
    let (k,weight,season)=match label{
        "knn-k8"=>(8,Weight::Uniform,None),"knn-k16"=>(16,Weight::Uniform,None),
        "knn-k32"=>(32,Weight::Uniform,None),"knn-k64"=>(64,Weight::Uniform,None),
        "analog-k16-inverse"=>(16,Weight::Inverse,None),"analog-k64-inverse"=>(64,Weight::Inverse,None),
        "analog-k16-gaussian"=>(16,Weight::Gaussian,None),"analog-k64-gaussian"=>(64,Weight::Gaussian,None),
        "season-k8-d15"=>(8,Weight::Uniform,Some(15)),"season-k16-d15"=>(16,Weight::Uniform,Some(15)),
        "season-k8-d30"=>(8,Weight::Uniform,Some(30)),"season-k16-d30"=>(16,Weight::Uniform,Some(30)),
        _=>return Err("unknown neighbor configuration".into())
    };Ok(Config{k,weight,season})
}
pub(super) fn subset(train:&[TrainingRow],budget:usize)->Vec<&TrainingRow>{
    let n=train.len().min(budget);(0..n).map(|i|&train[i*train.len()/n]).collect()
}
#[derive(Clone,Copy,Debug)]
pub(super) struct Neighbor{pub(super) d2:f64,pub(super) i:usize}
impl PartialEq for Neighbor{fn eq(&self,other:&Self)->bool{self.cmp(other)==Ordering::Equal}}
impl Eq for Neighbor{}
impl PartialOrd for Neighbor{fn partial_cmp(&self,other:&Self)->Option<Ordering>{Some(self.cmp(other))}}
impl Ord for Neighbor{fn cmp(&self,other:&Self)->Ordering{self.d2.total_cmp(&other.d2).then_with(||self.i.cmp(&other.i))}}
pub(super) fn distance(a:&[f64;D],b:&[f64;D])->f64{
    a.iter().zip(b).map(|(x,y)|(x-y)*(x-y)).sum::<f64>()/D as f64
}
fn phase_match(a:&[f64;D],b:&[f64;D],days:u32)->bool{
    let hour=a[72]*b[72]+a[73]*b[73];let year=a[74]*b[74]+a[75]*b[75];
    hour+1e-12>=(std::f64::consts::TAU*2./24.).cos()&&year+1e-12>=(std::f64::consts::TAU*days as f64/365.).cos()
}
pub(super) fn nearest(rows:&[&TrainingRow],q:&[f64;D],k:usize,season:Option<u32>)->(Vec<Neighbor>,usize,bool){
    let mut heap=BinaryHeap::with_capacity(k);let mut candidates=0;
    for(i,r)in rows.iter().enumerate(){
        if let Some(days)=season{if !phase_match(&r.input.x,q,days){continue}}
        candidates+=1;let n=Neighbor{d2:distance(&r.input.x,q),i};
        if heap.len()<k{heap.push(n)}else if n<*heap.peek().unwrap(){*heap.peek_mut().unwrap()=n;}
    }
    if candidates==0&&season.is_some(){let(v,_,_)=nearest(rows,q,1,None);return(v,0,true)}
    (heap.into_sorted_vec(),candidates,false)
}
fn aggregate(rows:&[&TrainingRow],neighbors:&[Neighbor],weight:Weight)->Result<[f64;Y],String>{
    if neighbors.is_empty(){return Err("empty neighbor set".into())}
    let scale=neighbors.last().unwrap().d2.max(1e-12);
    let exact=weight!=Weight::Uniform&&neighbors[0].d2==0.;
    let mut sum=[0.;Y];let mut norm=0.;
    for n in neighbors{
        let w=if exact{if n.d2==0.{1.}else{0.}}else{match weight{
            Weight::Uniform=>1.,Weight::Inverse=>1./n.d2.sqrt().max(1e-8),
            Weight::Gaussian=>(-n.d2/(2.*scale)).exp()
        }};
        norm+=w;for(h,v)in sum.iter_mut().enumerate(){*v+=w*rows[n.i].normalized_target[h];}
    }
    let pred=sum.map(|v|v/norm);
    if norm<=0.||!norm.is_finite()||pred.iter().any(|v|!v.is_finite()){return Err("invalid analog weights or prediction".into())}
    Ok(pred)
}
pub(super) struct Cached{pub(super) predictions:Vec<(i64,[f64;Y])>}
impl Predictor for Cached{
    fn predict(&self,x:&OriginInput)->[f64;Y]{
        self.predictions.binary_search_by_key(&x.origin,|r|r.0).ok().map(|i|self.predictions[i].1).unwrap_or([f64::NAN;Y])
    }
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let cfg=config(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);
    println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let rows=subset(&f.train,BUDGET);
    if rows.len()<cfg.k||rows.iter().any(|r|r.input.x.iter().chain(&r.normalized_target).any(|v|!v.is_finite())){return Err("invalid reference subset".into())}
    let workspace=rows.len()*std::mem::size_of::<&TrainingRow>()+m*std::mem::size_of::<(i64,[f64;Y])>()+cfg.k*std::mem::size_of::<Neighbor>();
    let now=Instant::now();let sample=m.min(32);
    for i in 0..sample{let q=&f.eval[i*m/sample].input.x;let(v,_,_)=nearest(&rows,q,cfg.k,cfg.season);let _=std::hint::black_box(aggregate(&rows,&v,cfg.weight)?);}
    let secs=now.elapsed().as_secs_f64();let estimate=secs*m as f64/sample as f64;
    println!("neighbor-preflight,{id},{label},{},{sample},{workspace},{secs:.6},{estimate:.6}",rows.len());
    if workspace>2_000_000_000||2.*estimate+180.>GUARD{
        println!("blocked,{id},{label},preflight workspace or runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())
    }
    let mut preds=Vec::with_capacity(m);let(mut min_candidates,mut sum_candidates,mut fallback)=(usize::MAX,0usize,0usize);
    let timer=Instant::now();
    for r in &f.eval{
        if now.elapsed().as_secs_f64()>GUARD{println!("blocked,{id},{label},runtime guard");out.flush().map_err(|e|e.to_string())?;return Ok(())}
        let(v,c,b)=nearest(&rows,&r.input.x,cfg.k,cfg.season);min_candidates=min_candidates.min(c);sum_candidates+=c;fallback+=usize::from(b);
        preds.push((r.input.origin,f.restore(aggregate(&rows,&v,cfg.weight)?)));
    }
    if preds.windows(2).any(|w|w[0].0>=w[1].0){return Err("prediction origins not ordered".into())}
    println!("neighbor-config,{id},{label},{},{},{:?},{},{},{:.8},{fallback},{:.6}",rows.len(),cfg.k,cfg.weight,cfg.season.unwrap_or(0),min_candidates,sum_candidates as f64/m as f64,timer.elapsed().as_secs_f64());
    score(&Cached{predictions:preds},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn row(x:[f64;D],v:f64)->TrainingRow{TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:[v;Y],normalized_target:[v;Y]}}
    fn point(v:f64)->[f64;D]{let mut x=[0.;D];x[0]=v;x}
    fn calendar(hour:f64,day:f64)->[f64;D]{let mut x=[0.;D];let a=std::f64::consts::TAU*hour/24.;let b=std::f64::consts::TAU*day/365.;x[72]=a.sin();x[73]=a.cos();x[74]=b.sin();x[75]=b.cos();x}
    #[test]fn distance_and_ties(){assert!((distance(&point(1.),&point(3.))-0.05).abs()<1e-12);let d=[row(point(-1.),1.),row(point(1.),3.),row(point(2.),9.)];let r:Vec<_>=d.iter().collect();let(v,c,b)=nearest(&r,&point(0.),1,None);assert_eq!((v[0].i,c,b),(0,3,false));let(v,_,_)=nearest(&r,&point(0.),2,None);assert_eq!(aggregate(&r,&v,Weight::Uniform).unwrap(),[2.;Y]);}
    #[test]fn exact_match_precedence(){let d=[row(point(0.),2.),row(point(0.),4.),row(point(1.),100.)];let r:Vec<_>=d.iter().collect();let(v,_,_)=nearest(&r,&point(0.),3,None);for w in [Weight::Inverse,Weight::Gaussian]{assert_eq!(aggregate(&r,&v,w).unwrap(),[3.;Y]);}}
    #[test]fn weighted_analogs(){let d=[row(point(1.),2.),row(point(2.),8.)];let r:Vec<_>=d.iter().collect();let(v,_,_)=nearest(&r,&point(0.),2,None);assert!((aggregate(&r,&v,Weight::Inverse).unwrap()[0]-4.).abs()<1e-12);let a=(-0.125f64).exp();let b=(-0.5f64).exp();assert!((aggregate(&r,&v,Weight::Gaussian).unwrap()[0]-(2.*a+8.*b)/(a+b)).abs()<1e-12);}
    #[test]fn seasonal_wrap_and_fallback(){let q=calendar(23.,364.);assert!(phase_match(&calendar(1.,1.),&q,15));assert!(!phase_match(&calendar(3.,1.),&q,15));assert!(!phase_match(&calendar(1.,20.),&q,15));assert!(phase_match(&calendar(1.,20.),&q,30));let d=[row(calendar(9.,150.),3.),row(calendar(1.,1.),9.)];let r:Vec<_>=d.iter().collect();let(v,c,b)=nearest(&r,&q,8,Some(15));assert_eq!((v.len(),c,b,v[0].i),(1,1,false,1));let(v,c,b)=nearest(&r[..1],&q,8,Some(15));assert_eq!((v.len(),c,b),(1,0,true));}
    #[test]fn deterministic_subset(){let d:Vec<_>=(0..10).map(|i|row(point(i as f64),i as f64)).collect();assert_eq!(subset(&d,4).iter().map(|r|r.normalized_target[0]).collect::<Vec<_>>(),vec![0.,2.,5.,7.]);assert_eq!(subset(&d,20).len(),10);}
}
