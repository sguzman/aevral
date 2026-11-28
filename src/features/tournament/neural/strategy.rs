//! C010 F01-F14: formulations, corrections and training objectives.
use super::{Adam,D,Y,TrainingRow,Rng,guard,Structured};
use super::super::{OriginInput,load,score,Fold};
use super::super::neighbors::{subset,Cached};
use crate::{dense::Dense,benchmark::Metrics};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
mod objectives;
use objectives::Objective;
enum Mode{Joint(Dense,Adam),Hits(Structured,Adam),Heads(Vec<(Dense,Adam)>,bool),Recursive(Dense,Adam),Correction(Dense,Adam,Dense,Adam)}
#[derive(Clone,Copy)]
enum Transform{Level,Daily,Difference}
struct Forecast{mode:Mode,mean:f64,sd:f64,transform:Transform,objective:Objective,clamp:bool}
fn scalar(dims:&[usize],seed:u64)->(Dense,Adam){let m=Dense::new(dims,seed);let o=Adam::new(m.parameters(),0.001);(m,o)}
impl Forecast{
    fn new(label:&str,seed:u64,mean:f64,sd:f64)->Result<Self,String>{
        let transform=if label.starts_with("daily-residual"){Transform::Daily}else if label.starts_with("difference"){Transform::Difference}else{Transform::Level};
        let objective=if label.starts_with("mae-"){Objective::Mae}else if label.starts_with("huber-"){Objective::Huber(1./sd)}else if label=="pinball-median"{Objective::Median}else if label=="pinball-multi"{Objective::Quantiles}else{match label{"lead-early"=>Objective::Early,"lead-late"=>Objective::Late,"multitask-mean"=>Objective::AuxMean,"multitask-mean-change"=>Objective::AuxChange(0.),"bound-penalty"=>Objective::Bound((-60.-mean)/sd,(60.-mean)/sd),"trajectory-slope"=>Objective::Slope,"trajectory-curvature"=>Objective::Curvature,_=>Objective::Mse}};
        let clamp=label=="bound-clamp";
        let mode=match label{
            "joint-mlp64"|"daily-residual-mlp64"|"difference-mlp64"|"mae-mlp64"|"huber-mlp64"|"pinball-median"|"lead-early"|"lead-late"|"bound-clamp"|"bound-penalty"|"trajectory-slope"|"trajectory-curvature"=>{let(m,o)=scalar(&[D,64,Y],seed);Mode::Joint(m,o)},
            "joint-nhits64"|"daily-residual-nhits64"|"difference-nhits64"|"mae-nhits64"|"huber-nhits64"=>{let m=Structured::new("nhits-w64",seed).ok_or("missing N-HiTS")?;let o=Adam::new(m.parameters(),0.001);Mode::Hits(m,o)},
            "multitask-mean"|"multitask-mean-change"=>{let extra=if label=="multitask-mean"{1}else{2};let(m,o)=scalar(&[D,64,Y+extra],seed);Mode::Joint(m,o)},
            "pinball-multi"=>{let(m,o)=scalar(&[D,64,3*Y],seed);Mode::Joint(m,o)},
            "perlead-h8"|"perlead-h16"|"dirrec-h8"|"dirrec-h16"=>{let dir=label.starts_with("dirrec");let h=if label.ends_with("16"){16}else{8};Mode::Heads((1..=Y).map(|j|scalar(&[D+usize::from(dir),h,1],seed+j as u64)).collect(),dir)},
            "recursive-h16"|"recursive-h32"=>{let h=if label.ends_with("32"){32}else{16};let(m,o)=scalar(&[D+2,h,1],seed);Mode::Recursive(m,o)},
            "correction-affine"|"correction-h16"=>{let(b,bo)=scalar(&[D,64,Y],seed);let dims=if label=="correction-affine"{vec![D+Y,Y]}else{vec![D+Y,16,Y]};let(c,co)=scalar(&dims,seed+1000);Mode::Correction(b,bo,c,co)},
            _=>return Err("unknown C010 recipe".into())
        };Ok(Self{mode,mean,sd,transform,objective,clamp})
    }
    fn parameters(&self)->usize{match &self.mode{Mode::Joint(m,_)|Mode::Recursive(m,_)=>m.parameters(),Mode::Hits(m,_)=>m.parameters(),Mode::Heads(ms,_)=>ms.iter().map(|(m,_)|m.parameters()).sum(),Mode::Correction(b,_,c,_)=>b.parameters()+c.parameters()}}
    fn updates(&self)->usize{match &self.mode{Mode::Joint(..)|Mode::Hits(..)|Mode::Correction(..)=>1,_=>Y}}
    fn z0(&self,r:&OriginInput)->f64{(r.recent[Y-1]-self.mean)/self.sd}
    fn encode(&self,r:&OriginInput,y:[f64;Y])->[f64;Y]{match self.transform{Transform::Level=>y,Transform::Daily=>std::array::from_fn(|j|y[j]-(r.recent[j]-self.mean)/self.sd),Transform::Difference=>std::array::from_fn(|j|y[j]-if j==0{self.z0(r)}else{y[j-1]})}}
    fn decode(&self,r:&OriginInput,mut y:[f64;Y])->[f64;Y]{match self.transform{Transform::Level=>{},Transform::Daily=>for j in 0..Y{y[j]+=(r.recent[j]-self.mean)/self.sd},Transform::Difference=>{let mut prev=self.z0(r);for v in &mut y{*v+=prev;prev=*v;}}}y}
    fn raw_predict(&self,r:&OriginInput)->Result<[f64;Y],String>{
        let mut out=[0.;Y];let mut prev=self.z0(r);
        match &self.mode{
            Mode::Joint(m,_)=>{let p=m.predict(&r.x);if matches!(self.objective,Objective::Quantiles){out.copy_from_slice(&p[Y..2*Y]);}else{out.copy_from_slice(&p[..Y]);}},
            Mode::Hits(m,_)=>out.copy_from_slice(&m.predict(&r.x)),
            Mode::Heads(ms,dir)=>for(j,(m,_))in ms.iter().enumerate(){let mut x=r.x.to_vec();if *dir{x.push(prev)}out[j]=m.predict(&x)[0];prev=out[j];},
            Mode::Recursive(m,_)=>for j in 0..Y{let mut x=r.x.to_vec();x.extend([prev,(j+1)as f64/Y as f64]);out[j]=m.predict(&x)[0];prev=out[j];},
            Mode::Correction(b,_,c,_)=>{let p=b.predict(&r.x);let mut x=r.x.to_vec();x.extend_from_slice(&p);let q=c.predict(&x);for j in 0..Y{out[j]=p[j]+q[j];}}
        }
        let out=self.decode(r,out);
        if out.iter().any(|v|!v.is_finite()||v.abs()>1e6){return Err("numerical prediction bound".into())}Ok(out)
    }
    fn predict(&self,r:&OriginInput)->Result<[f64;Y],String>{
        let mut p=self.raw_predict(r)?;if self.clamp{let lo=(-60.-self.mean)/self.sd;let hi=(60.-self.mean)/self.sd;for v in &mut p{*v=v.clamp(lo,hi);}}Ok(p)
    }
    fn crossings(&self,r:&OriginInput)->Result<usize,String>{
        if !matches!(self.objective,Objective::Quantiles){return Ok(0)}
        let Mode::Joint(m,_)=&self.mode else{return Err("quantile backbone".into())};let p=m.predict(&r.x);
        if p.iter().any(|v|!v.is_finite()||v.abs()>1e6){return Err("quantile numerical bound".into())}
        Ok((0..Y).filter(|&j|p[j]>p[Y+j]||p[Y+j]>p[2*Y+j]).count())
    }
    fn step(&mut self,r:&TrainingRow,now:Instant)->Result<(f64,usize),String>{
        let objective=match self.objective{Objective::AuxChange(_)=>Objective::AuxChange(self.z0(&r.input)),o=>o};let target=self.encode(&r.input,r.normalized_target);let mut prev=self.z0(&r.input);let mut loss=0.;let mut clips=0;
        match &mut self.mode{
            Mode::Joint(m,o)=>{guard(now)?;loss=if matches!(objective,Objective::Mse){m.adam_step(&r.input.x,&target,o).0}else{m.adam_step_with(&r.input.x,&target,o,|p,t|objective.value_grad(p,t)).0};if !m.finite(){return Err("nonfinite dense".into())}},
            Mode::Hits(m,o)=>{guard(now)?;let(l,c)=if matches!(objective,Objective::Mse){m.step(&r.input.x,&target,o)}else{m.step_with(&r.input.x,&target,o,|p,t|objective.value_grad(p,t))};loss=l;clips=usize::from(c);if !m.finite(){return Err("nonfinite hits".into())}},
            Mode::Heads(ms,dir)=>{for(j,(m,o))in ms.iter_mut().enumerate(){guard(now)?;let mut x=r.input.x.to_vec();if *dir{x.push(prev)}let y=r.normalized_target[j];loss+=m.adam_step(&x,&[y],o).0;prev=y;if !m.finite(){return Err("nonfinite head".into())}}loss/=Y as f64;},
            Mode::Recursive(m,o)=>{for j in 0..Y{guard(now)?;let mut x=r.input.x.to_vec();x.extend([prev,(j+1)as f64/Y as f64]);let y=r.normalized_target[j];loss+=m.adam_step(&x,&[y],o).0;prev=y;if !m.finite(){return Err("nonfinite recursive".into())}}loss/=Y as f64;},
            Mode::Correction(..)=>return Err("correction requires staged training".into())
        }
        if !loss.is_finite(){return Err("nonfinite loss".into())}Ok((loss,clips))
    }
}
fn correction_split<'a,'b>(rows:&'a [&'b TrainingRow])->Result<(&'a [&'b TrainingRow],&'a [&'b TrainingRow],usize),String>{
    if rows.len()<4||rows.windows(2).any(|w|w[0].input.origin>=w[1].input.origin){return Err("invalid chronological correction rows".into())}
    let boundary=rows.len()*3/4;let later=&rows[boundary..];let cutoff=later[0].input.origin;
    let end=rows[..boundary].partition_point(|r|r.input.origin+24*60<cutoff);
    if end==0{return Err("empty purged base fit".into())}Ok((&rows[..end],later,boundary-end))
}
fn train_correction(m:&mut Forecast,rows:&[&TrainingRow],epochs:usize,seed:u64,now:Instant,log:Option<(&str,&str)>)->Result<usize,String>{
    let(base_rows,later,dropped)=correction_split(rows)?;
    let Mode::Correction(b,bo,c,co)=&mut m.mode else{return Err("wrong correction mode".into())};
    if let Some((id,label))=log{println!("strategy-split,{id},{label},{seed},{},{},{dropped},{}",base_rows.len(),later.len(),12*(base_rows.len()+later.len()));}
    let mut rng=Rng(seed^0xD1B54A32D192ED03);let mut order:Vec<_>=(0..base_rows.len()).collect();
    for e in 0..epochs{rng.shuffle(&mut order);let mut loss=0.;for &i in &order{guard(now)?;let r=base_rows[i];loss+=b.adam_step(&r.input.x,&r.normalized_target,bo).0;if !b.finite(){return Err("nonfinite base".into())}}if !loss.is_finite(){return Err("nonfinite base loss".into())}if let Some((id,label))=log{println!("strategy-base-epoch,{id},{label},{seed},{},{:.10},0",e+1,loss/base_rows.len()as f64);}}
    let mut rng=Rng((seed+1000)^0xD1B54A32D192ED03);let mut order:Vec<_>=(0..later.len()).collect();
    for e in 0..epochs{rng.shuffle(&mut order);let mut loss=0.;for &i in &order{guard(now)?;let r=later[i];let p=b.predict(&r.input.x);let mut x=r.input.x.to_vec();x.extend_from_slice(&p);let target:[f64;Y]=std::array::from_fn(|j|r.normalized_target[j]-p[j]);loss+=c.adam_step(&x,&target,co).0;if !c.finite(){return Err("nonfinite correction".into())}}if !loss.is_finite(){return Err("nonfinite correction loss".into())}if let Some((id,label))=log{println!("strategy-epoch,{id},{label},{seed},{},{:.10},0",e+1,loss/later.len()as f64);}}
    Ok(0)
}
fn train(m:&mut Forecast,rows:&[&TrainingRow],epochs:usize,seed:u64,now:Instant,log:Option<(&str,&str)>)->Result<usize,String>{
    if matches!(&m.mode,Mode::Correction(..)){return train_correction(m,rows,epochs,seed,now,log)}
    let mut rng=Rng(seed^0xD1B54A32D192ED03);let mut order:Vec<_>=(0..rows.len()).collect();let mut total=0;
    for e in 0..epochs{rng.shuffle(&mut order);let(mut loss,mut clips)=(0.,0);for &i in &order{let(l,c)=m.step(rows[i],now)?;loss+=l;clips+=c;}if !loss.is_finite(){return Err("nonfinite epoch".into())}total+=clips;if let Some((id,label))=log{println!("strategy-epoch,{id},{label},{seed},{},{:.10},{clips}",e+1,loss/rows.len()as f64);}}
    Ok(total)
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str,seed:u64)->Result<(),String>{
    if ![42,123,2026].contains(&seed){return Err("seed outside frozen recipe".into())}
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for row in &f.eval{daily.add(&row.target,&row.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    run_prepared(&f,output,label,seed)
}
pub(in crate::features::tournament) fn run_prepared(f:&super::super::PreparedFold,output:&Path,label:&str,seed:u64)->Result<(),String>{
    let id=f.fold.id;
    if ![42,123,2026].contains(&seed){return Err("seed outside frozen recipe".into())}
    let m=f.eval.len();
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    macro_rules! block{($reason:expr)=>{{println!("blocked,{id},{label},{}",$reason);out.flush().map_err(|e|e.to_string())?;return Ok(())}}}
    macro_rules! checked{($e:expr)=>{match $e{Ok(v)=>v,Err(e)=>{block!(e)}}}}
    let rows=subset(&f.train,8192);let n=rows.len();
    if super::super::Fold::retrospective(id).is_ok(){
        super::super::representation::origins(&output.parent().ok_or("output directory")?.join("backend-fit-origins.txt"),rows.iter().map(|r|r.input.origin))?;
    }
    let mut probe=Forecast::new(label,seed,f.target_mean,f.target_sd)?;let params=probe.parameters();let updates=if matches!(&probe.mode,Mode::Correction(..)){let(a,b,_)=checked!(correction_split(&rows));12*(a.len()+b.len())}else{n*12*probe.updates()};
    let workspace=8*(16*params+65536+4*n+m*(Y+2));
    println!("strategy-config,{id},{label},{seed},{n},12,0.001,{params},{workspace},{}",updates);
    if workspace>2_000_000_000{block!("preflight workspace guard")}
    let now=Instant::now();let p=n.min(128);let small:Vec<_>=(0..p).map(|i|rows[i*n/p]).collect();
    let t=Instant::now();checked!(train(&mut probe,&small,1,seed,now,None));let fp=t.elapsed().as_secs_f64();
    let q=n.min(32);let t=Instant::now();for i in 0..q{checked!(guard(now));std::hint::black_box(checked!(probe.predict(&rows[i*n/q].input)));}let pp=t.elapsed().as_secs_f64();drop(probe);
    let estimate=fp*(n*12)as f64/p as f64+pp*m as f64/q as f64;
    println!("strategy-preflight,{id},{label},{seed},{p},{q},{fp:.6},{pp:.6},{estimate:.6}");
    if 2.*estimate+180.>1400.{block!("preflight runtime guard")}
    let mut model=Forecast::new(label,seed,f.target_mean,f.target_sd)?;let t=Instant::now();let clips=checked!(train(&mut model,&rows,12,seed,now,Some((id,label))));let fit=t.elapsed().as_secs_f64();
    let t=Instant::now();let mut predictions=Vec::with_capacity(m);
    let mut crossings=0;let(mut outside,mut clipped)=(0,0);let(mut slope,mut curvature)=(0.,0.);
    for row in &f.eval{
        checked!(guard(now));let raw=checked!(model.raw_predict(&row.input));let mut p=f.restore(raw);
        if label.starts_with("bound-"){outside+=p.iter().filter(|v|**v< -60.||**v>60.).count();}
        if model.clamp{for v in &mut p{let before=*v;*v=v.clamp(-60.,60.);clipped+=usize::from(before!=*v);}}
        if label.starts_with("trajectory-"){let e:[f64;Y]=std::array::from_fn(|j|p[j]-row.target[j]);for j in 1..Y{slope+=(e[j]-e[j-1]).abs();}for j in 2..Y{curvature+=(e[j]-2.*e[j-1]+e[j-2]).abs();}}
        predictions.push((row.input.origin,p));crossings+=checked!(model.crossings(&row.input));
    }
    if label.starts_with("bound-"){println!("strategy-bounds,{id},{label},{seed},{outside},{clipped},{}",m*Y);}
    if label.starts_with("trajectory-"){println!("strategy-trajectory,{id},{label},{seed},{:.8},{:.8},{},{}",slope/(m*(Y-1))as f64,curvature/(m*(Y-2))as f64,m*(Y-1),m*(Y-2));}
    if matches!(model.objective,Objective::Quantiles){println!("strategy-quantile,{id},{label},{seed},{crossings},{}",m*Y);}
    println!("strategy-fit,{id},{label},{seed},{clips},{fit:.6},{:.6}",t.elapsed().as_secs_f64());
    score(&Cached{predictions},label,&f,&mut out)?;out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn row()->TrainingRow{TrainingRow{input:OriginInput{origin:0,x:[0.1;D],recent:[10.;Y]},target:[12.;Y],normalized_target:[0.4;Y]}}
    #[test]fn all_recipes_seed_shape_and_overfit(){let r=row();for label in ["joint-mlp64","joint-nhits64","perlead-h8","perlead-h16","recursive-h16","recursive-h32","dirrec-h8","dirrec-h16"]{let mut m=Forecast::new(label,42,10.,5.).unwrap();let before=m.predict(&r.input).unwrap();assert_eq!(before,Forecast::new(label,42,10.,5.).unwrap().predict(&r.input).unwrap());let first=m.step(&r,Instant::now()).unwrap().0;let mut last=first;for _ in 0..80{last=m.step(&r,Instant::now()).unwrap().0}assert!(last<first,"{label}");assert_ne!(before,m.predict(&r.input).unwrap());}}
    #[test]fn free_running_matches_manual_unroll(){let r=row();for label in ["recursive-h16","dirrec-h8"]{let m=Forecast::new(label,42,10.,5.).unwrap();let p=m.predict(&r.input).unwrap();let mut prev=0.;for j in 0..Y{let mut x=r.input.x.to_vec();x.push(prev);let v=match &m.mode{Mode::Recursive(d,_)=>{x.push((j+1)as f64/Y as f64);d.predict(&x)[0]},Mode::Heads(h,_)=>h[j].0.predict(&x)[0],_=>unreachable!()};assert_eq!(v,p[j]);prev=v;}}}
    #[test]fn joint_controls_match_original_backbones(){let r=row();let mut m=Forecast::new("joint-mlp64",42,10.,5.).unwrap();let mut d=Dense::new(&[D,64,Y],42);let mut a=Adam::new(d.parameters(),0.001);assert_eq!(m.predict(&r.input).unwrap().to_vec(),d.predict(&r.input.x));m.step(&r,Instant::now()).unwrap();d.adam_step(&r.input.x,&r.normalized_target,&mut a);assert_eq!(m.predict(&r.input).unwrap().to_vec(),d.predict(&r.input.x));}
    #[test]fn transformed_targets_roundtrip(){let mut r=row();for j in 0..Y{r.input.recent[j]=j as f64;r.normalized_target[j]=j as f64/10.;}for label in ["daily-residual-mlp64","daily-residual-nhits64","difference-mlp64","difference-nhits64"]{let mut m=Forecast::new(label,42,10.,5.).unwrap();let encoded=m.encode(&r.input,r.normalized_target);let restored=m.decode(&r.input,encoded);for j in 0..Y{assert!((restored[j]-r.normalized_target[j]).abs()<1e-12)}let first=m.step(&r,Instant::now()).unwrap().0;let mut last=first;for _ in 0..80{last=m.step(&r,Instant::now()).unwrap().0}assert!(last<first,"{label}");assert!(m.predict(&r.input).is_ok());}}
    #[test]fn correction_purge_and_frozen_base(){let mut rs:Vec<_>=(0..128).map(|i|{let mut r=row();r.input.origin=i*60;r}).collect();let refs:Vec<_>=rs.iter().collect();let(a,b,d)=correction_split(&refs).unwrap();assert_eq!((a.len(),b.len(),d),(72,32,24));assert!(a.last().unwrap().input.origin+24*60<b[0].input.origin);for label in ["correction-affine","correction-h16"]{let mut m=Forecast::new(label,42,10.,5.).unwrap();let mut n=Forecast::new(label,42,10.,5.).unwrap();train(&mut m,&refs,2,42,Instant::now(),None).unwrap();train(&mut n,&refs,2,42,Instant::now(),None).unwrap();assert_eq!(m.predict(&refs[0].input).unwrap(),n.predict(&refs[0].input).unwrap());let Mode::Correction(base,_,_,_)=&m.mode else{unreachable!()};let mut expected=Dense::new(&[D,64,Y],42);let mut opt=Adam::new(expected.parameters(),0.001);let mut rng=Rng(42^0xD1B54A32D192ED03);let mut order:Vec<_>=(0..a.len()).collect();for _ in 0..2{rng.shuffle(&mut order);for &i in &order{expected.adam_step(&a[i].input.x,&a[i].normalized_target,&mut opt);}}assert_eq!(base.predict(&refs[0].input.x),expected.predict(&refs[0].input.x));}drop(refs);rs.reverse();assert!(correction_split(&rs.iter().collect::<Vec<_>>()).is_err());}
    #[test]fn objectives_train_and_median_readout(){let r=row();for label in ["mae-mlp64","mae-nhits64","huber-mlp64","huber-nhits64","pinball-median","pinball-multi"]{let mut m=Forecast::new(label,42,10.,5.).unwrap();let first=m.step(&r,Instant::now()).unwrap().0;let mut last=first;for _ in 0..80{last=m.step(&r,Instant::now()).unwrap().0;}assert!(last<first,"{label}");assert!(m.predict(&r.input).is_ok());if label=="pinball-multi"{let Mode::Joint(d,_)=&m.mode else{unreachable!()};let p=d.predict(&r.input.x);assert_eq!(m.predict(&r.input).unwrap().to_vec(),p[Y..2*Y]);let expected=(0..Y).filter(|&j|p[j]>p[Y+j]||p[Y+j]>p[2*Y+j]).count();assert_eq!(m.crossings(&r.input).unwrap(),expected);}}}
    #[test]fn final_objectives_readout_and_fit(){
        let r=row();
        for label in ["lead-early","lead-late","multitask-mean","multitask-mean-change","bound-clamp","bound-penalty","trajectory-slope","trajectory-curvature"]{
            let mut m=Forecast::new(label,42,10.,5.).unwrap();let first=m.step(&r,Instant::now()).unwrap().0;let mut last=first;for _ in 0..80{last=m.step(&r,Instant::now()).unwrap().0;}assert!(last<first,"{label}");assert!(m.predict(&r.input).is_ok());
            if label.starts_with("multitask-"){let Mode::Joint(d,_)=&m.mode else{unreachable!()};assert_eq!(m.predict(&r.input).unwrap().to_vec(),d.predict(&r.input.x)[..Y]);}
        }
        let a=Forecast::new("bound-clamp",42,100.,1.).unwrap();let b=Forecast::new("joint-mlp64",42,100.,1.).unwrap();assert_eq!(a.raw_predict(&r.input).unwrap(),b.predict(&r.input).unwrap());assert!(a.predict(&r.input).unwrap().iter().all(|v|*v>= -160.&&*v<= -40.));
    }
}
