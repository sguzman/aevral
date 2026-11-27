//! C008 D08: observable quantile-state Markov chain.
use super::{TrainingRow,OriginInput,Y};
use std::time::Instant;
pub(super) struct Model{pub thresholds:Vec<f64>,pub means:Vec<f64>,pub trans:Vec<Vec<f64>>,pub pairs:usize,pub gaps:usize,pub empty:usize,pub row_error:f64,b:usize,hourly:bool}
fn bucket(t:i64)->usize{(t.div_euclid(60).rem_euclid(24)/6)as usize}
fn bin(th:&[f64],x:f64)->usize{th.partition_point(|v|*v<x)}
fn normalize(row:&mut[f64]){let s=row.iter().sum::<f64>();for v in row{*v/=s;}}
impl Model{
    pub(super) fn fit(rows:&[&TrainingRow],b:usize,hourly:bool,now:Instant)->Result<Self,String>{
        if rows.len()<b||b<2||rows.iter().any(|r|!r.input.x[71].is_finite()){return Err("invalid Markov training".into())}
        let mut sorted:Vec<_>=rows.iter().map(|r|r.input.x[71]).collect();sorted.sort_by(f64::total_cmp);
        let thresholds:Vec<_>=(1..b).map(|j|sorted[j*sorted.len()/b]).collect();
        let states:Vec<_>=rows.iter().map(|r|bin(&thresholds,r.input.x[71])).collect();
        let mut means=vec![0.;b];let mut n=vec![0usize;b];for(r,&s)in rows.iter().zip(&states){means[s]+=r.input.x[71];n[s]+=1;}
        if n.contains(&0){return Err("empty quantile bin".into())}
        for i in 0..b{means[i]/=n[i]as f64;}
        let mut pooled=vec![0.001;b*b];let mut counts=vec![vec![0.;b*b];4];let(mut pairs,mut gaps)=(0,0);
        for i in 1..rows.len(){if i%1024==0&&now.elapsed().as_secs_f64()>1400.{return Err("runtime guard".into())}
            if rows[i].input.origin-rows[i-1].input.origin!=60{gaps+=1;continue}
            let j=states[i-1]*b+states[i];pooled[j]+=1.;counts[bucket(rows[i-1].input.origin)][j]+=1.;pairs+=1;
        }
        if pairs==0{return Err("no adjacent training transitions".into())}
        for row in pooled.chunks_exact_mut(b){normalize(row)}
        let mut empty=0;let trans=if hourly{for a in &mut counts{for i in 0..b{let row=&mut a[i*b..(i+1)*b];if row.iter().all(|v|*v==0.){empty+=1;}for j in 0..b{row[j]+=pooled[i*b+j];}normalize(row);}}counts}else{vec![pooled]};
        let row_error=trans.iter().flat_map(|t|t.chunks_exact(b)).map(|r|(r.iter().sum::<f64>()-1.).abs()).fold(0.,f64::max);
        if row_error>1e-12{return Err("nonstochastic Markov rows".into())}
        Ok(Self{thresholds,means,trans,pairs,gaps,empty,row_error,b,hourly})
    }
    pub(super) fn predict(&self,x:&OriginInput)->Result<[f64;Y],String>{
        let mut p=vec![0.;self.b];p[bin(&self.thresholds,x.x[71])]=1.;
        let mut pred=[0.;Y];for(h,y)in pred.iter_mut().enumerate(){let t=&self.trans[if self.hourly{bucket(x.origin+60*h as i64)}else{0}];let mut next=vec![0.;self.b];
            for i in 0..self.b{for j in 0..self.b{next[j]+=p[i]*t[i*self.b+j];}}
            if next.iter().any(|v|!v.is_finite()||*v<0.)||(next.iter().sum::<f64>()-1.).abs()>1e-10{return Err("invalid Markov probability".into())}
            *y=next.iter().zip(&self.means).map(|(a,b)|a*b).sum();p=next;
        }Ok(pred)
    }
}
#[cfg(test)]
mod tests{
use super::*;use super::super::D;
fn row(t:i64,v:f64)->TrainingRow{let mut x=[0.;D];x[71]=v;TrainingRow{input:OriginInput{origin:t,x,recent:[0.;Y]},target:[0.;Y],normalized_target:[0.;Y]}}
#[test]fn quantile_tie(){assert_eq!(bin(&[1.,2.],1.),0);assert_eq!(bin(&[1.,2.],1.1),1);assert_eq!(bin(&[1.,2.],2.),1)}
#[test]fn utc_wrap(){assert_eq!(bucket(23*60),3);assert_eq!(bucket(24*60),0);assert_eq!(bucket(6*60),1);}
#[test]fn gap_and_stochastic(){let r:Vec<_>=(0..16).map(|i|row((i+if i>=8{1}else{0})*60,i as f64)).collect();let refs:Vec<_>=r.iter().collect();for hourly in [false,true]{let m=Model::fit(&refs,4,hourly,Instant::now()).unwrap();assert_eq!(m.pairs,14);assert_eq!(m.gaps,1);assert!(m.row_error<1e-12);assert!(m.predict(&r[0].input).unwrap().iter().all(|v|*v>=0.&&*v<=15.));}}
#[test]fn probability_propagation(){let m=Model{thresholds:vec![0.],means:vec![-2.,3.],trans:vec![vec![0.,1.,1.,0.]],pairs:1,gaps:0,empty:0,row_error:0.,b:2,hourly:false};let p=m.predict(&row(0,-1.).input).unwrap();assert_eq!(p[0],3.);assert_eq!(p[1],-2.);assert_eq!(p[23],-2.);}
}
