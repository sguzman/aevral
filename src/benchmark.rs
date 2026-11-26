//! Matched-origin evaluation for AEV-001, validation only.
use crate::{Mlp,weather::{Example,Split,WINDOW,HORIZON,minute}};

#[derive(Debug,Clone,Copy)]
pub struct Scale {pub mean:f64,pub sd:f64}
impl Scale {
    pub fn fit(rows:&[&Example])->Result<Self,String>{
        if rows.is_empty()||rows.iter().any(|r|r.split!=Split::Train){return Err("scale requires nonempty training-only examples".into())}
        let n=(rows.len()*WINDOW) as f64;
        let mean=rows.iter().flat_map(|r|r.x).sum::<f64>()/n;
        let sd=(rows.iter().flat_map(|r|r.x).map(|v|(v-mean).powi(2)).sum::<f64>()/n).sqrt();
        if !mean.is_finite()||!sd.is_finite()||sd<=0.0{return Err("invalid scale".into())}
        Ok(Self{mean,sd})
    }
    pub fn normalize(&self,v:f64)->f64{(v-self.mean)/self.sd}
    pub fn restore(&self,v:f64)->f64{v*self.sd+self.mean}
}
/// Hour-of-day by UTC calendar month, fitted solely to training labels.
/// Use only lead-one labels, so each valid hour occurs at most once.
pub struct Climatology{means:[f64;288]}
fn slot(t:i64)->usize{
    let hour=(t/60).rem_euclid(24) as usize;
    let mut year=2000;
    while year<2026&&t>=minute(&format!("{}-01-01 00:00",year+1)).unwrap(){year+=1}
    let mut month=1;
    while month<12&&t>=minute(&format!("{year}-{:02}-01 00:00",month+1)).unwrap(){month+=1}
    (month-1)*24+hour
}
impl Climatology{
    pub fn fit(rows:&[&Example])->Result<Self,String>{
        if rows.is_empty()||rows.iter().any(|r|r.split!=Split::Train){return Err("climatology requires training only".into())}
        let(mut sum,mut n)=([0.0;288],[0usize;288]);
        for r in rows{let s=slot(r.origin+60);sum[s]+=r.y[0];n[s]+=1}
        let fallback=sum.iter().sum::<f64>()/rows.len() as f64;
        Ok(Self{means:std::array::from_fn(|i|if n[i]==0{fallback}else{sum[i]/n[i] as f64})})
    }
    pub fn predict(&self,r:&Example)->[f64;HORIZON]{std::array::from_fn(|i|self.means[slot(r.origin+60*(i+1) as i64)])}
}
#[derive(Debug,Clone)]
pub struct Metrics{pub count:usize,abs:[f64;HORIZON],sq:[f64;HORIZON]}
impl Default for Metrics{fn default()->Self{Self{count:0,abs:[0.0;HORIZON],sq:[0.0;HORIZON]}}}
impl Metrics{
    pub fn add(&mut self,y:&[f64;HORIZON],p:&[f64]){
        assert_eq!(p.len(),HORIZON);
        self.count+=1;
        for i in 0..HORIZON{let e=p[i]-y[i];self.abs[i]+=e.abs();self.sq[i]+=e*e}
    }
    pub fn mae(&self)->f64{self.abs.iter().sum::<f64>()/(self.count*HORIZON) as f64}
    pub fn rmse(&self)->f64{(self.sq.iter().sum::<f64>()/(self.count*HORIZON) as f64).sqrt()}
    pub fn lead(&self,i:usize)->(f64,f64){(self.abs[i]/self.count as f64,(self.sq[i]/self.count as f64).sqrt())}
}
pub fn persistence(r:&Example)->[f64;HORIZON]{[r.x[WINDOW-1];HORIZON]}
pub fn daily(r:&Example)->[f64;HORIZON]{r.x}
pub fn run(rows:&[Example])->Result<(),String>{
    if rows.iter().any(|r|r.split==Split::Test){return Err("test cases forbidden in validation benchmark".into())}
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let val:Vec<_>=rows.iter().filter(|r|r.split==Split::Validation).collect();
    if val.is_empty(){return Err("no validation examples".into())}
    let scale=Scale::fit(&train)?;
    let clim=Climatology::fit(&train)?;
    let mut model=Mlp::new(WINDOW,32,HORIZON,42);
    let mut order:Vec<_>=(0..train.len()).collect();
    let mut rng=42u64;
    // Fixed initial recipe; shuffle training examples reproducibly each epoch.
    for epoch in 0..10{
        for i in (1..order.len()).rev(){
            rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
            order.swap(i,(rng%(i as u64+1))as usize);
        }
        let mut loss=0.0;
        for &i in &order{
            let r=train[i];
            loss+=model.train_step(&r.x.map(|x|scale.normalize(x)),&r.y.map(|y|scale.normalize(y)),0.005);
        }
        println!("epoch={} training_normalized_mse={:.8}",epoch+1,loss/train.len() as f64);
    }
    let mut m:[Metrics;4]=std::array::from_fn(|_|Metrics::default());
    let mut seasons:[[Metrics;4];4]=std::array::from_fn(|_|std::array::from_fn(|_|Metrics::default()));
    for r in &val{
        let pred=model.predict(&r.x.map(|x|scale.normalize(x))).into_iter().map(|v|scale.restore(v)).collect::<Vec<_>>();
        let p=[persistence(r),daily(r),clim.predict(r),pred.try_into().map_err(|_|"invalid model output")?];
        let month=slot(r.origin+60)/24+1;
        let season=(month%12)/3;
        for i in 0..4{m[i].add(&r.y,&p[i]);seasons[season][i].add(&r.y,&p[i]);}
    }
    let names=["persistence","daily","climatology","mlp"];
    println!("train={} validation={} scale_mean={:.8} scale_sd={:.8}",train.len(),val.len(),scale.mean,scale.sd);
    println!("model,count,mae_c,rmse_c,skill_vs_daily");
    for i in 0..4{println!("{},{},{:.8},{:.8},{:.8}",names[i],m[i].count,m[i].mae(),m[i].rmse(),1.0-m[i].mae()/m[1].mae())}
    println!("model,lead_hours,mae_c,rmse_c");
    for i in 0..4{for h in 0..HORIZON{let(a,b)=m[i].lead(h);println!("{},{},{:.8},{:.8}",names[i],h+1,a,b)}}
    println!("model,season_first_target,count,mae_c,rmse_c");
    for (s,name) in ["DJF","MAM","JJA","SON"].iter().enumerate(){for i in 0..4{let v=&seasons[s][i];if v.count>0{println!("{},{},{},{:.8},{:.8}",names[i],name,v.count,v.mae(),v.rmse())}}}
    if m.iter().any(|v|!v.mae().is_finite()||!v.rmse().is_finite()){return Err("nonfinite scores".into())}
    Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn row(split:Split)->Example{Example{origin:minute("2018-01-01 00:00").unwrap(),split,x:std::array::from_fn(|i|i as f64),y:[24.0;24]}}
    #[test]fn scale_rejects_leakage(){let r=row(Split::Validation);assert!(Scale::fit(&[&r]).is_err());assert!(Climatology::fit(&[&r]).is_err());}
    #[test]fn baseline_alignment_and_metrics(){let r=row(Split::Train);assert_eq!(daily(&r),r.x);assert_eq!(persistence(&r),[23.0;24]);let mut m=Metrics::default();m.add(&r.y,&persistence(&r));assert_eq!(m.mae(),1.0);assert_eq!(m.rmse(),1.0);assert_eq!(m.lead(0),(1.0,1.0));let s=Scale::fit(&[&r]).unwrap();assert_eq!(s.mean,11.5);assert!((s.restore(s.normalize(4.0))-4.0).abs()<1e-12);}
    #[test]fn calendar_slots(){assert_eq!(slot(minute("2018-12-31 23:00").unwrap()),287);assert_eq!(slot(minute("2019-01-01 00:00").unwrap()),0);}
    #[test]fn rejects_test(){assert!(run(&[row(Split::Test)]).is_err());}
}
