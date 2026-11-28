//! C014 empirical, parametric and conformal residual models.
use super::{core::*,ensemble::{Features,Row}};
pub fn center(x:&Features,b:bool)->f64{if b{(x.base[0]+x.base[1]+x.base[2]+x.base[3])/4.}else{x.base[0]}}
pub fn spread(x:&Features)->f64{let v=&x.base[..4];v.iter().copied().fold(f64::NEG_INFINITY,f64::max)-v.iter().copied().fold(f64::INFINITY,f64::min)}
#[derive(Clone,Debug)]
pub struct Group {pub support:Vec<f64>,pub count:usize,pub mean:f64,pub sd:f64,pub fallback:bool}
fn empirical(rows:&[&Row],b:bool)->Group{
 let v=sorted(rows.iter().map(|r|r.target-center(&r.x,b)).collect());
 let mean=v.iter().sum::<f64>()/v.len()as f64;
 let sd=(v.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/v.len()as f64).sqrt();
 Group{support:sample(&v),count:v.len(),mean,sd,fallback:false}
}
fn parametric(rows:&[&Row],b:bool,z:&[f64])->Group{
 let mut g=empirical(rows,b);g.sd=g.sd.max(0.05);
 g.support=z.iter().map(|x|g.mean+g.sd*x).collect();g
}
#[derive(Clone,Debug)]
pub struct Residual {pub b:bool,pub groups:Vec<Group>,pub thresholds:Vec<f64>,pub conditional:bool,pub global:bool}
impl Residual{
 pub fn support(&self,x:&Features)->Vec<f64>{
  let i=if self.global{0}else if self.conditional{(x.lead-1)*2+usize::from(spread(x)>self.thresholds[x.lead-1])}else{x.lead-1};
  self.groups[i].support.iter().map(|r|center(x,self.b)+r).collect()
 }
}
pub fn fit_residual(id:u8,rows:&[&Row])->Residual{
 assert!((1..=9).contains(&id)&&!rows.is_empty());
 let b=matches!(id,2|4|6|7|8|9);let global=id<=2;let conditional=matches!(id,7|9);
 let z=if id>=5{standardized_support(id>=8)}else{Vec::new()};
 let fit=|r:&[&Row]|if id>=5{parametric(r,b,&z)}else{empirical(r,b)};
 let mut groups=Vec::new();let mut thresholds=Vec::new();
 if global{groups.push(fit(rows));}else{for h in 1..=24{
  let r:Vec<_>=rows.iter().copied().filter(|r|r.x.lead==h).collect();
  let base=fit(&r);
  if conditional{
   let v=sorted(r.iter().map(|r|spread(&r.x)).collect());let t=quantile(&v,0.5);thresholds.push(t);
   for high in [false,true]{
    let s:Vec<_>=r.iter().copied().filter(|r|(spread(&r.x)>t)==high).collect();
    if s.len()<128{let mut g=base.clone();g.count=s.len();g.fallback=true;groups.push(g);}else{groups.push(fit(&s));}
   }
  }else{groups.push(base);}
 }}
 Residual{b,groups,thresholds,conditional,global}
}
#[derive(Clone,Debug)]
pub struct Conformal{pub b:bool,pub normalized:bool,pub bands:Vec<[(f64,f64);4]>,pub counts:Vec<usize>}
impl Conformal{
 pub fn issue(&self,x:&Features)->Issued{
  let c=center(x,self.b);let s=if self.normalized{spread(x).max(0.25)}else{1.};
  let bands=self.bands[x.lead-1].map(|(l,u)|(c-s*l,c+s*u));
  assert!(bands.iter().all(|(l,u)|l<=u));
  Issued{median:c,quantiles:None,bands,crps:None}
 }
}
pub fn fit_conformal(id:u8,rows:&[&Row])->Conformal{
 assert!((21..=24).contains(&id));let b=id!=21;let normalized=id==24;
 let mut bands=Vec::new();let mut counts=Vec::new();
 for h in 1..=24{
  let r:Vec<_>=rows.iter().copied().filter(|r|r.x.lead==h).collect();
  let n=r.len();assert!(n>0);counts.push(n);
  let scores=sorted(r.iter().map(|r|{let e=r.target-center(&r.x,b);if id==23{e}else{e.abs()/if normalized{spread(&r.x).max(0.25)}else{1.}}}).collect());
  let lower=if id==23{sorted(scores.iter().map(|s|-s).collect())}else{scores.clone()};
  let limits=LEVELS.map(|c|{let p=if id==23{(1.+c)/2.}else{c};let k=conformal_rank(p,n);(lower[k],scores[k])});
  assert!(limits.iter().all(|(l,u)|l+u>=0.));bands.push(limits);
 }
 Conformal{b,normalized,bands,counts}
}
#[cfg(test)]
mod tests{
 use super::*;
 fn row(h:usize,b:[f64;8],y:f64)->Row{Row{x:Features{origin:0,lead:h,month:1,base:b},target:y,daily:0.}}
 fn fixture()->Vec<Row>{(1..=24).flat_map(|h|[row(h,[2.;8],1.),row(h,[2.;8],4.)]).collect()}
 #[test]fn empirical(){let r=fixture();let a:Vec<_>=r.iter().collect();let g=fit_residual(3,&a);let v=g.support(&r[0].x);assert_eq!(v.len(),99);assert_eq!(v[0],1.);assert_eq!(v[98],4.);}
 #[test]fn parametric_mean_sd(){let r=fixture();let a:Vec<_>=r.iter().collect();let g=fit_residual(6,&a);assert!((g.groups[0].mean-0.5).abs()<1e-12);assert!((g.groups[0].sd-1.5).abs()<1e-12);assert!((g.support(&r[0].x)[49]-2.5).abs()<1e-8);}
 #[test]fn conditional_fallback(){let r=fixture();let a:Vec<_>=r.iter().collect();let g=fit_residual(7,&a);assert_eq!(g.groups.len(),48);assert!(g.groups.iter().all(|g|g.fallback));assert_eq!(g.groups[1].count,0);}
 #[test]fn symmetric_conformal(){let r=fixture();let a:Vec<_>=r.iter().collect();let p=fit_conformal(22,&a).issue(&r[0].x);assert_eq!(p.bands[3],(0.,4.));assert!(p.crps.is_none());}
 #[test]fn asymmetric_conformal(){let r=fixture();let a:Vec<_>=r.iter().collect();let p=fit_conformal(23,&a).issue(&r[0].x);assert_eq!(p.bands[3],(1.,4.));}
 #[test]fn normalized_conformal(){let r=fixture();let a:Vec<_>=r.iter().collect();let p=fit_conformal(24,&a).issue(&r[0].x);assert_eq!(p.bands[3],(0.,4.));}
}
