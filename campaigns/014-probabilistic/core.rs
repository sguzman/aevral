//! C014 probability and scoring primitives. All distributions are issued finite equal-mass supports.
pub const LEVELS:[f64;4]=[0.50,0.80,0.90,0.95];
pub const QUANTILES:[f64;9]=[0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975];
pub const M:usize=99;
pub fn sorted(mut v:Vec<f64>)->Vec<f64>{assert!(!v.is_empty()&&v.iter().all(|x|x.is_finite()));v.sort_by(f64::total_cmp);v}
pub fn rank(p:f64,n:usize)->usize{
 assert!(n>0&&(0.0..=1.0).contains(&p));
 // Avoid spurious rank increments when a decimal probability times n is integral.
 let x=p*n as f64;let nearest=x.round();let x=if (x-nearest).abs()<1e-10{nearest}else{x};
 (x.ceil() as usize).clamp(1,n)-1
}
pub fn quantile(v:&[f64],p:f64)->f64{v[rank(p,v.len())]}
pub fn sample(v:&[f64])->Vec<f64>{(0..M).map(|j|quantile(v,(j as f64+0.5)/M as f64)).collect()}
pub fn conformal_rank(c:f64,n:usize)->usize{rank(c,n+1).min(n-1)}
pub fn pinball(y:f64,q:f64,p:f64)->f64{let e=y-q;if e>=0.{p*e}else{(p-1.)*e}}
pub fn interval(y:f64,lo:f64,hi:f64,c:f64)->(f64,f64,f64){
 assert!(y.is_finite()&&lo.is_finite()&&hi.is_finite()&&lo<=hi&&c>0.&&c<1.);
 (if lo<=y&&y<=hi{1.}else{0.},hi-lo,hi-lo+2./(1.-c)*((lo-y).max(0.)+(y-hi).max(0.)))
}
#[derive(Clone,Debug)]
pub struct Issued{
 pub median:f64,pub quantiles:Option<[f64;9]>,pub bands:[(f64,f64);4],pub crps:Option<f64>,
}
pub fn distribution(v:&[f64],y:f64)->Issued{
 assert!(!v.is_empty()&&v.iter().all(|x|x.is_finite())&&v.windows(2).all(|w|w[0]<=w[1])&&y.is_finite());
 let n=v.len()as f64;
 // Center differences on y for numerical stability; sum of coefficients is zero.
 let a=v.iter().map(|x|(x-y).abs()).sum::<f64>()/n;
 let b=v.iter().enumerate().map(|(i,x)|(2.*(i+1)as f64-n-1.)*(x-y)).sum::<f64>()/(n*n);
 let crps=a-b;assert!(crps>=-1e-10&&crps.is_finite());
 Issued{median:quantile(v,0.5),quantiles:Some(QUANTILES.map(|p|quantile(v,p))),bands:LEVELS.map(|c|(quantile(v,(1.-c)/2.),quantile(v,(1.+c)/2.))),crps:Some(crps.max(0.))}
}
#[derive(Clone,Debug,Default)]
pub struct Metrics{
 pub n:usize,pub distribution_n:usize,pub ae:f64,pub se:f64,pub crps:f64,pub pinball:[f64;9],pub coverage:[f64;4],pub width:[f64;4],pub interval_score:[f64;4]
}
impl Metrics{
 pub fn add(&mut self,y:f64,p:&Issued){
  assert!(y.is_finite()&&p.median.is_finite());
  assert!(p.bands.windows(2).all(|w|w[1].0<=w[0].0+1e-10&&w[1].1>=w[0].1-1e-10));
  self.n+=1;let e=p.median-y;self.ae+=e.abs();self.se+=e*e;
  match(p.crps,p.quantiles){
   (Some(c),Some(q))=>{assert!(c.is_finite()&&c>=0.&&q.windows(2).all(|w|w[0]<=w[1]));self.distribution_n+=1;self.crps+=c;for i in 0..9{self.pinball[i]+=pinball(y,q[i],QUANTILES[i]);}},
   (None,None)=>{},_=>panic!("incomplete distribution"),
  }
  for (i,&(lo,hi)) in p.bands.iter().enumerate(){let(c,w,s)=interval(y,lo,hi,LEVELS[i]);self.coverage[i]+=c;self.width[i]+=w;self.interval_score[i]+=s;}
 }
}
/// Standard normal CDF using a bounded series for erf on |z|<=4 and tail saturation outside.
/// Only inverse probabilities in [0.5/99,98.5/99] are required.
pub fn normal_cdf(z:f64)->f64{
 if z<=-8.{return 0.;}if z>=8.{return 1.;}
 // Integral of standard normal density using composite Simpson, error <1e-10 on [-8,8].
 let x=z.abs();let n=512;let h=x/n as f64;
 let density=|v:f64|(-0.5*v*v).exp()/(2.*std::f64::consts::PI).sqrt();
 let mut s=density(0.)+density(x);
 for i in 1..n{s+=if i%2==0{2.}else{4.}*density(i as f64*h);}
 0.5+z.signum()*h*s/3.
}
pub fn t3_cdf(t:f64)->f64{let z=t/3f64.sqrt();0.5+(z.atan()+z/(1.+z*z))/std::f64::consts::PI}
pub fn inverse(p:f64,cdf:fn(f64)->f64)->f64{
 assert!(p>0.&&p<1.);let(mut lo,mut hi)=(-64.,64.);assert!(cdf(lo)<p&&cdf(hi)>p);
 for _ in 0..48{let x=(lo+hi)/2.;if cdf(x)<p{lo=x;}else{hi=x;}}(lo+hi)/2.
}
pub fn standardized_support(student:bool)->Vec<f64>{
 let cdf=if student{t3_cdf}else{normal_cdf};
 (0..M).map(|j|inverse((j as f64+0.5)/M as f64,cdf)/if student{3f64.sqrt()}else{1.}).collect()
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn inverse_cdf_rank(){let v=[1.,2.,3.,4.];assert_eq!(quantile(&v,0.5),2.);assert_eq!(quantile(&v,0.95),4.);assert_eq!(rank(0.95,100),94);assert_eq!(rank(0.0,4),0);}
 #[test]fn conformal(){assert_eq!(conformal_rank(0.9,9),8);assert_eq!(conformal_rank(0.8,99),79);assert_eq!(conformal_rank(0.95,2),1);}
 #[test]fn crps_point(){let p=distribution(&[3.],1.);assert_eq!(p.crps,Some(2.));assert_eq!(p.median,3.);}
 #[test]fn crps_pair(){let p=distribution(&[0.,2.],1.);assert_eq!(p.crps,Some(0.5));assert_eq!(p.median,0.);}
 #[test]fn crps_pairwise_parity(){let v=[-2.,0.,0.,1.,5.];let y=0.4;let a=v.iter().map(|x|(x-y).abs()).sum::<f64>()/5.;let b=v.iter().flat_map(|x|v.iter().map(move|z|(x-z).abs())).sum::<f64>()/50.;assert!((distribution(&v,y).crps.unwrap()-a+b).abs()<1e-12);}
 #[test]fn loss(){assert!((pinball(2.,1.,0.1)-0.1).abs()<1e-12);assert!((pinball(0.,1.,0.1)-0.9).abs()<1e-12);assert_eq!(interval(1.,1.,3.,0.8),(1.,2.,2.));assert!((interval(0.,1.,3.,0.8).2-12.).abs()<1e-12);}
 #[test]fn normal_reference(){assert!((inverse(0.975,normal_cdf)-1.95996398454).abs()<1e-8);assert!((inverse(0.005,normal_cdf)+2.57582930355).abs()<1e-8);}
 #[test]fn student_reference(){assert!((inverse(0.975,t3_cdf)-3.18244630528).abs()<1e-8);assert!((t3_cdf(0.)-0.5).abs()<1e-12);}
 #[test]fn support_order(){for student in [false,true]{let v=standardized_support(student);assert_eq!(v.len(),99);assert!(v.windows(2).all(|w|w[0]<w[1]));assert!(v[49].abs()<1e-8);assert!((v[0]+v[98]).abs()<1e-8);}}
}
