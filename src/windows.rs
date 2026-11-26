//! AEV-002: matched complete 72-hour history, varying the visible suffix.
use crate::{Mlp,benchmark::{Metrics,Climatology},weather::{Report,Example,Split,split_for,HORIZON}};
const MAX:usize=72;
pub const WINDOWS:[usize;5]=[6,12,24,48,72];
#[derive(Debug)]
struct Case{origin:i64,split:Split,history:[f64;MAX],target:[f64;HORIZON]}
fn cases(bins:&[Option<Report>],first:i64)->Vec<Case>{
    let mut out=Vec::new();
    for i in MAX-1..bins.len().saturating_sub(HORIZON){
        let origin=first+i as i64*60;
        let Some(split)=split_for(origin)else{continue};
        if split==Split::Test{continue}
        let v:Option<Vec<_>>=bins[i+1-MAX..=i+HORIZON].iter().map(|r|r.and_then(|v|v.celsius)).collect();
        if let Some(v)=v{out.push(Case{origin,split,history:v[..MAX].try_into().unwrap(),target:v[MAX..].try_into().unwrap()})}
    }out
}
fn suffix(c:&Case,w:usize)->&[f64]{&c.history[MAX-w..]}
fn scale(train:&[&Case],w:usize)->Result<(f64,f64),String>{
    if train.is_empty()||train.iter().any(|r|r.split!=Split::Train){return Err("scale requires training cases".into())}
    let n=(train.len()*w)as f64;
    let mean=train.iter().flat_map(|r|suffix(r,w)).sum::<f64>()/n;
    let sd=(train.iter().flat_map(|r|suffix(r,w)).map(|v|(v-mean).powi(2)).sum::<f64>()/n).sqrt();
    if !mean.is_finite()||!sd.is_finite()||sd<=0.0{return Err("invalid scaling".into())}Ok((mean,sd))
}
pub fn run(bins:&[Option<Report>],first:i64)->Result<(),String>{
    let rows=cases(bins,first);
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let val:Vec<_>=rows.iter().filter(|r|r.split==Split::Validation).collect();
    if train.is_empty()||val.is_empty(){return Err("empty training or validation set".into())}
    println!("AEV-002 matched train={} validation={}",train.len(),val.len());
    let clim_rows:Vec<_>=train.iter().map(|r|Example{origin:r.origin,split:r.split,x:suffix(r,24).try_into().unwrap(),y:r.target}).collect();
    let clim=Climatology::fit(&clim_rows.iter().collect::<Vec<_>>())?;
    let mut baselines:[Metrics;3]=std::array::from_fn(|_|Metrics::default());
    for r in &val{
        let e=Example{origin:r.origin,split:r.split,x:suffix(r,24).try_into().unwrap(),y:r.target};
        for (m,p) in baselines.iter_mut().zip([crate::benchmark::persistence(&e),crate::benchmark::daily(&e),clim.predict(&e)]){m.add(&e.y,&p)}
    }
    let daily=baselines[1].mae();
    for (name,m) in ["persistence","daily","climatology"].iter().zip(&baselines){report(name,m,daily)?}
    for w in WINDOWS{
        let(mean,sd)=scale(&train,w)?;
        let mut model=Mlp::new(w,32,HORIZON,42);
        let mut order:Vec<_>=(0..train.len()).collect();let mut rng=42u64;
        for epoch in 0..10{
            for i in (1..order.len()).rev(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;order.swap(i,(rng%(i as u64+1))as usize);}
            let mut loss=0.0;
            for &i in &order{
                let r=train[i];let x:Vec<_>=suffix(r,w).iter().map(|v|(v-mean)/sd).collect();
                loss+=model.train_step(&x,&r.target.map(|v|(v-mean)/sd),0.005);
            }
            println!("window={w} epoch={} normalized_mse={:.8}",epoch+1,loss/train.len()as f64);
        }
        let mut m=Metrics::default();
        for r in &val{
            let x:Vec<_>=suffix(r,w).iter().map(|v|(v-mean)/sd).collect();
            let p:Vec<_>=model.predict(&x).iter().map(|v|v*sd+mean).collect();m.add(&r.target,&p);
        }
        println!("window={w} scale_mean={mean:.8} scale_sd={sd:.8} parameters={}",32*w+32+32*HORIZON+HORIZON);
        report(&format!("mlp-{w}h"),&m,daily)?;
    }Ok(())
}
fn report(name:&str,m:&Metrics,daily:f64)->Result<(),String>{
    if !m.mae().is_finite()||!m.rmse().is_finite(){return Err("nonfinite metrics".into())}
    println!("aggregate,{name},{},{:.8},{:.8},{:.8}",m.count,m.mae(),m.rmse(),1.0-m.mae()/daily);
    for i in 0..HORIZON{let(a,b)=m.lead(i);println!("lead,{name},{},{:.8},{:.8}",i+1,a,b)}
    Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn exact_alignment_and_common_missingness(){
        let mut b:Vec<_>=(0..98).map(|i|Some(Report{minute:i*60,celsius:Some(i as f64)})).collect();
        let r=cases(&b,0);assert_eq!(r.len(),3);assert_eq!(r[0].origin,71*60);
        assert_eq!(suffix(&r[0],6),&[66.,67.,68.,69.,70.,71.]);assert_eq!(r[0].target[0],72.);assert_eq!(r[0].target[23],95.);
        b[2]=None;assert!(cases(&b,0).is_empty());
    }
    #[test]fn no_test_cases(){
        let b:Vec<_>=(0..98).map(|i|Some(Report{minute:i*60,celsius:Some(i as f64)})).collect();
        assert!(cases(&b,crate::weather::minute("2023-01-01 00:00").unwrap()).is_empty());
    }
}
