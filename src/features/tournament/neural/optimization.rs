//! C012 neural configuration and training, preserving the existing Dense and N-HiTS backbones.
use super::{Adam,D,Y,TrainingRow,Rng,guard,Structured};
use super::super::{optimization::{Candidate,Family,early_split},OriginInput,PreparedFold,representation};
use crate::covariates::Weather;
use std::path::Path;
use crate::dense::Dense;
use std::time::Instant;

#[derive(Clone,Copy,Debug,PartialEq)]
struct Config{sgd:bool,lr:f64,epochs:usize,batch:usize,cosine:bool,decay:f64,dropout:bool,stop:bool}
fn config(c:Candidate)->Result<Config,String>{
 let mut r=Config{sgd:false,lr:0.001,epochs:30,batch:1,cosine:false,decay:0.,dropout:false,stop:false};
 match (c.family,c.configuration){
  (Family::Mlp80|Family::Suffix24,"sgd-lr005-e30-b1")=>{r.sgd=true;r.lr=0.005;},
  (Family::Mlp80|Family::Suffix24|Family::Nhits,"adam-lr001-e12-b1")=>r.epochs=12,
  (Family::Mlp80|Family::Suffix24,"adam-lr0003-e30-b1")=>r.lr=0.0003,
  (Family::Nhits,"adam-lr0003-e12-b1")=>{r.lr=0.0003;r.epochs=12;},
  (Family::Suffix24|Family::Nhits,"adam-lr001-e30-b1")=>{},
  (Family::Mlp80|Family::Nhits,"adam-cosine-lr001-e30-b1")=>r.cosine=true,
  (Family::Mlp80|Family::Nhits,"adam-lr001-e30-b1-wd0001")=>r.decay=0.0001,
  (Family::Mlp80|Family::Nhits,"adam-lr001-e30-b32")=>r.batch=32,
  (Family::Suffix24,"adam-lr001-e30-b32-drop01")=>{r.batch=32;r.dropout=true;},
  (Family::Suffix24,"adam-lr001-e30-b1-stop")=>r.stop=true,
  _=>return Err("unknown frozen C012 neural recipe".into())
 }Ok(r)
}
impl Config{fn rate(self,e:usize)->f64{if self.cosine{self.lr*(0.1+0.9*(1.+(std::f64::consts::PI*e as f64/(self.epochs-1)as f64).cos())/2.)}else{self.lr}}}
#[derive(Clone)]
pub(super) enum Backbone{Mlp(Dense),Hits(Structured)}
impl Backbone{
 fn new(family:Family,seed:u64)->Result<Self,String>{Ok(match family{Family::Mlp80=>Self::Mlp(Dense::new(&[D,64,Y],seed)),Family::Suffix24=>Self::Mlp(Dense::new(&[32,64,Y],seed)),Family::Nhits=>Self::Hits(Structured::new("nhits-w64",seed).ok_or("missing N-HiTS")?),_=>return Err("not a neural family".into())})}
 fn parameters(&self)->usize{match self{Self::Mlp(m)=>m.parameters(),Self::Hits(m)=>m.parameters()}}
 fn finite(&self)->bool{match self{Self::Mlp(m)=>m.finite(),Self::Hits(m)=>m.finite()}}
 pub(super) fn predict(&self,x:&[f64])->Result<[f64;Y],String>{
  let v=match self{Self::Mlp(m)=>m.predict(x),Self::Hits(m)=>m.predict(x)};
  let y:[f64;Y]=v.try_into().map_err(|_|"prediction shape")?;if y.iter().any(|v|!v.is_finite()||v.abs()>1e6){return Err("numerical prediction bound".into())}Ok(y)
 }
}
fn input(family:Family,r:&OriginInput)->&[f64]{if family==Family::Suffix24{&r.x[48..]}else{&r.x}}
#[derive(Debug,PartialEq)]
pub(super) struct Accounting{pub epochs:usize,pub selected_epoch:usize,pub updates:usize,pub examples:usize,pub clips:usize,pub fit:usize,pub val:usize,pub purged:usize}
pub(super) fn train(c:Candidate,rows:&[&TrainingRow],seed:u64,now:Instant,probe:bool,log:Option<&str>)->Result<(Backbone,Accounting),String>{
 let mut cfg=config(c)?;if rows.is_empty(){return Err("empty neural training".into())}
 if probe{cfg.epochs=1;cfg.cosine=false;cfg.stop=false;}
 let(fit_end,val_start)=if cfg.stop{early_split(&rows.iter().map(|r|r.input.origin).collect::<Vec<_>>())?}else{(rows.len(),rows.len())};
 let mut model=Backbone::new(c.family,seed)?;let mut opt=Adam::new(model.parameters(),cfg.lr);
 let mut rng=Rng(seed^0xD1B54A32D192ED03);let mut dropout_rng=Rng(seed^0xA0761D6478BD642F);
 let mut sgd_rng=seed;let mut order:Vec<_>=(0..fit_end).collect();
 let(mut best,mut best_loss,mut stale)=(None,f64::INFINITY,0);
 let mut a=Accounting{epochs:0,selected_epoch:0,updates:0,examples:0,clips:0,fit:fit_end,val:rows.len()-val_start,purged:val_start-fit_end};
 for e in 0..cfg.epochs{
  if cfg.sgd{for i in (1..order.len()).rev(){sgd_rng^=sgd_rng<<13;sgd_rng^=sgd_rng>>7;sgd_rng^=sgd_rng<<17;order.swap(i,(sgd_rng%(i as u64+1))as usize);}}else{rng.shuffle(&mut order);}
  let lr=cfg.rate(e);opt.set_lr(lr);let mut loss=0.;let mut epoch_clips=0;
  for batch in order.chunks(cfg.batch){
   guard(now)?;let examples:Vec<(&[f64],&[f64])>=batch.iter().map(|&i|(input(c.family,&rows[i].input),rows[i].normalized_target.as_slice())).collect();
   let masks=if cfg.dropout{Some((0..batch.len()).map(|_|(0..64).map(|_|if (dropout_rng.next()>>11)as f64/9007199254740992.<0.1{0.}else{1./0.9}).collect::<Vec<_>>()).collect::<Vec<_>>())}else{None};
   let (v,clip)=match &mut model{
    Backbone::Mlp(m) if cfg.sgd=>(m.train_step(examples[0].0,examples[0].1,lr),false),
    Backbone::Mlp(m)=>(m.optimization_batch(&examples,&mut opt,cfg.decay,masks.as_deref()),false),
    Backbone::Hits(m)=>m.optimization_mae_batch(&examples,&mut opt,cfg.decay)
   };
   if !v.is_finite()||!model.finite(){return Err("nonfinite neural training".into())}
   loss+=v*batch.len()as f64;a.updates+=1;a.examples+=batch.len();epoch_clips+=usize::from(clip);
  }
  a.epochs=e+1;a.clips+=epoch_clips;
  if let Some(id)=log{println!("optimization-epoch,{id},{},{seed},{},{lr:.10},{:.10},{epoch_clips}",c.id,e+1,loss/fit_end as f64);}
  if cfg.stop{
   let mut v=0.;for row in &rows[val_start..]{guard(now)?;let p=model.predict(input(c.family,&row.input))?;v+=p.iter().zip(row.normalized_target).map(|(p,y)|(p-y).powi(2)).sum::<f64>()/Y as f64;}v/=(rows.len()-val_start)as f64;
   if !v.is_finite(){return Err("nonfinite early-stop loss".into())}
   // Keep the actual lowest-loss checkpoint, even when a sub-threshold improvement does not reset patience.
   let significant=v<best_loss-1e-6;
   if v<best_loss{best_loss=v;best=Some(model.clone());a.selected_epoch=e+1;}
   if significant{stale=0}else{stale+=1;}
   if let Some(id)=log{println!("optimization-validation,{id},{},{seed},{},{v:.10},{stale},{}",c.id,e+1,a.selected_epoch);}
   if stale>=4{break}
  }else{a.selected_epoch=e+1;}
 }
 if let Some(b)=best{model=b;}Ok((model,a))
}
/// Reusable frozen C015 Adam MLP member M (O01-06).
pub struct AdamMlp {
    model: Dense,
    mean:f64,
    sd:f64,
}
impl AdamMlp {
    pub fn model_bytes(&self)->Result<Vec<u8>,String>{self.model.to_bytes()}
    pub fn from_model_bytes(bytes:&[u8],mean:f64,sd:f64)->Result<Self,String>{
        if !mean.is_finite()||!sd.is_finite()||sd<=0.{return Err("invalid Adam MLP scales".into())}
        Ok(Self{model:Dense::from_bytes(bytes,&[D,64,Y])?,mean,sd})
    }
    pub fn fit(rows:&[TrainingRow],mean:f64,sd:f64)->Result<Self,String>{
        use super::super::optimization::{candidates,subset,Rung};
        if !mean.is_finite()||!sd.is_finite()||sd<=0.{return Err("invalid Adam MLP scales".into())}
        let selected=subset(rows,Rung::Final)?;
        if selected.iter().any(|r|r.input.x.iter().chain(&r.normalized_target).any(|v|!v.is_finite())){return Err("nonfinite Adam MLP training data".into())}
        let c=candidates()?.into_iter().find(|c|c.id=="O01-06").ok_or("missing frozen Adam recipe")?;
        let (model,a)=train(c,&selected,42,Instant::now(),false,None)?;
        if (a.epochs,a.selected_epoch,a.fit,a.val,a.purged,a.updates,a.examples)!=(30,30,32768,0,0,30720,983040){
            return Err("Adam MLP training accounting mismatch".into());
        }
        match model{Backbone::Mlp(model)=>Ok(Self{model,mean,sd}),_=>Err("unexpected backbone".into())}
    }
    pub fn predict(&self,x:&OriginInput)->Result<[f64;Y],String>{
        if x.x.iter().any(|v|!v.is_finite()){return Err("nonfinite Adam MLP input".into())}
        let y:[f64;Y]=self.model.predict(&x.x).try_into().map_err(|_|"prediction shape")?;
        if y.iter().any(|v|!v.is_finite()||v.abs()>1e6){return Err("numerical prediction bound".into())}
        let y=y.map(|v|v*self.sd+self.mean);
        if y.iter().any(|v|!v.is_finite()){return Err("nonfinite Adam MLP output".into())}
        Ok(y)
    }
    pub(crate) fn dense(&self)->&Dense{&self.model}
}
/// Preflight uses only selected training examples. No scored model is fitted until it passes.
pub(in crate::features::tournament) fn evaluate(c:Candidate,f:&PreparedFold,rows:&[&TrainingRow],seed:u64,output:&Path,first:i64,bins:&[Option<Weather>])->Result<(),String>{
 let id=f.fold.id;let label=c.id;let cfg=config(c)?;let n=rows.len();let m=f.eval.len();
 let now=Instant::now();let p=n.min(128);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
 let params=Backbone::new(c.family,seed)?.parameters();
 let workspace=8*(20*params+65536+4*n+f.possible*(Y*2+2)+64*cfg.batch);
 println!("optimization-neural-config,{id},{label},{seed},{n},{},{},{},{},{},{},{},{},{params},{workspace}",cfg.epochs,cfg.lr,cfg.batch,cfg.sgd,cfg.cosine,cfg.decay,cfg.dropout,cfg.stop);
 macro_rules! block{($reason:expr)=>{{println!("blocked,{id},{label},{}",$reason);return Ok(())}}}
 macro_rules! guarded{($e:expr)=>{match $e{Ok(v)=>v,Err(e)if e=="runtime guard"=>{block!(e)},Err(e)=>return Err(e)}}}
 if workspace>2_000_000_000{block!("preflight workspace guard")}
 let timer=Instant::now();let(probe,_)=guarded!(train(c,&small,seed,now,true,None));let fp=timer.elapsed().as_secs_f64();
 let q=n.min(32);let timer=Instant::now();for i in 0..q{std::hint::black_box(guarded!(probe.predict(input(c.family,&rows[i*n/q].input))));}
 let pp=timer.elapsed().as_secs_f64();drop(probe);
 let estimate=fp*n as f64*cfg.epochs as f64/p as f64+pp*(if c.family==Family::Suffix24{m+f.possible}else{m})as f64/q as f64;
 println!("optimization-neural-preflight,{id},{label},{seed},{p},{q},{fp:.6},{pp:.6},{estimate:.6}");
 if 2.*estimate+180.>1400.{block!("preflight runtime guard")}
 let timer=Instant::now();let(model,a)=guarded!(train(c,rows,seed,now,false,Some(id)));let fit=timer.elapsed().as_secs_f64();
 let dir=output.parent().ok_or("output directory")?;
 representation::origins(&dir.join("actual-fit-origins.txt"),rows[..a.fit].iter().map(|r|r.input.origin))?;
 representation::origins(&dir.join("internal-validation-origins.txt"),rows[n-a.val..].iter().map(|r|r.input.origin))?;
 println!("optimization-training,{id},{label},{seed},{},{},{},{},{},{},{},{}",a.epochs,a.selected_epoch,a.updates,a.examples,a.clips,a.fit,a.val,a.purged);
 let timer=Instant::now();let mut matched=Vec::with_capacity(m);
 for r in &f.eval{guarded!(guard(now));matched.push((r.input.origin,r.target,f.restore(guarded!(model.predict(input(c.family,&r.input))))));}
 let natural=if c.family==Family::Suffix24{Some(guarded!(representation::suffix24_natural(f,first,bins,&matched,|x|{guard(now)?;Ok(f.restore(model.predict(x)?))})))}else{None};
 let pred=timer.elapsed().as_secs_f64();
 representation::write_rows(output,id,label,&matched,"")?;
 if let Some((possible,rows))=natural{
  representation::origins(&dir.join("natural-origins.txt"),rows.iter().map(|r|r.0))?;
  representation::write_rows(&dir.join("natural-predictions.csv"),id,label,&rows,"natural-")?;
  println!("optimization-natural,{id},{label},{possible},{},{m}",rows.len());
 }else{println!("optimization-natural,{id},{label},{},{m},{m}",f.possible);}
 println!("optimization-cost,{id},{label},{fit:.6},{pred:.6},{params}");Ok(())
}
#[cfg(test)]
mod tests{
 use super::*;
 use super::super::super::optimization::{candidates,request,Rung};
 fn row(i:usize)->TrainingRow{TrainingRow{input:OriginInput{origin:i as i64*1440,x:[i as f64/100.;D],recent:[0.;Y]},target:[0.2;Y],normalized_target:[0.2;Y]}}
 #[test]fn all_neural_configs(){let rows=candidates().unwrap();assert_eq!(rows.iter().filter(|c|c.stochastic&&config(**c).is_ok()).count(),18);let c=config(request("O01-04",Rung::One,"2019",42).unwrap()).unwrap();assert!((c.rate(0)-0.001).abs()<1e-15);assert!((c.rate(29)-0.0001).abs()<1e-15);}
 #[test]fn batch_remainder_reproducible(){let data:Vec<_>=(0..35).map(row).collect();let rows:Vec<_>=data.iter().collect();let c=request("O02-04",Rung::One,"2019",42).unwrap();let(a,x)=train(c,&rows,42,Instant::now(),true,None).unwrap();let(b,y)=train(c,&rows,42,Instant::now(),true,None).unwrap();assert_eq!(x,y);assert_eq!((x.updates,x.examples),(2,35));assert_eq!(a.predict(&[0.1;32]).unwrap(),b.predict(&[0.1;32]).unwrap());}
 #[test]fn early_stop_accounting(){let data:Vec<_>=(0..40).map(row).collect();let rows:Vec<_>=data.iter().collect();let c=request("O02-06",Rung::One,"2019",42).unwrap();let(_,a)=train(c,&rows,42,Instant::now(),false,None).unwrap();assert_eq!((a.fit,a.val,a.purged),(36,4,0));assert_eq!(a.examples,a.epochs*36);assert_eq!(a.updates,a.examples);assert!((1..=a.epochs).contains(&a.selected_epoch));}
}
