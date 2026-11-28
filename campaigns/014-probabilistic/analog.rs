//! P14–P20 scalar analog and finite member distributions.
use super::{core::{self,M},ensemble::{Features,Row},residual::{center,spread}};
#[derive(Clone,Copy,Debug)]
pub struct AnalogRow{pub key:f64,pub origin:i64,pub residual:f64}
#[derive(Debug)]
pub struct Analog{pub b:bool,pub by_spread:bool,pub leads:Vec<Vec<AnalogRow>>}
fn order(a:&AnalogRow,b:&AnalogRow)->std::cmp::Ordering{a.key.total_cmp(&b.key).then(a.origin.cmp(&b.origin))}
/// Search sorted scalar keys, grouping equal keys. A group yields earliest origins first.
/// At equal distance between opposite keys, choose earliest origin across the two groups.
pub fn nearest(v:&[AnalogRow],x:f64,k:usize)->Vec<f64>{
 assert!(x.is_finite()&&v.len()>=k&&k>0);
 let split=v.partition_point(|r|r.key<x);
 let(mut left_end,mut right_start)=(split,split);
 let(mut li,mut le)=(0,0);let(mut ri,mut re)=(0,0);
 let mut out=Vec::with_capacity(k);
 while out.len()<k{
  if li==le&&left_end>0{
   let key=v[left_end-1].key;li=v[..left_end].partition_point(|r|r.key<key);le=left_end;left_end=li;
  }
  if ri==re&&right_start<v.len(){
   let key=v[right_start].key;ri=right_start;re=right_start+v[right_start..].partition_point(|r|r.key<=key);right_start=re;
  }
  let take_left=match(li<le,ri<re){
   (true,false)=>true,(false,true)=>false,(true,true)=>{
    let a=(x-v[li].key).abs();let b=(x-v[ri].key).abs();
    a<b||(a==b&&v[li].origin<v[ri].origin)
   },_=>panic!("insufficient analogs")
  };
  if take_left{out.push(v[li].residual);li+=1;}else{out.push(v[ri].residual);ri+=1;}
 }
 out
}
impl Analog{
 pub fn fit(rows:&[&Row],b:bool,by_spread:bool)->Self{
  let leads=(1..=24).map(|h|{
   let mut v:Vec<_>=rows.iter().filter(|r|r.x.lead==h).map(|r|{
    let c=center(&r.x,b);AnalogRow{key:if by_spread{spread(&r.x)}else{c},origin:r.x.origin,residual:r.target-c}
   }).collect();assert!(v.len()>=128);v.sort_by(order);v
  }).collect();Self{b,by_spread,leads}
 }
 pub fn support(&self,x:&Features)->Vec<f64>{
  let c=center(x,self.b);let key=if self.by_spread{spread(x)}else{c};
  core::sample(&core::sorted(nearest(&self.leads[x.lead-1],key,128))).iter().map(|r|c+r).collect()
 }
}
#[derive(Debug)]
pub struct Mixture{pub id:u8,pub residuals:Vec<Vec<Vec<f64>>>}
impl Mixture{
 pub fn fit(id:u8,rows:&[&Row])->Self{
  assert!((17..=20).contains(&id));
  let residuals=if id<=18{Vec::new()}else{(1..=24).map(|h|{
   let r:Vec<_>=rows.iter().copied().filter(|r|r.x.lead==h).collect();assert!(!r.is_empty());
   if id==19{
    let v=core::sorted(r.iter().flat_map(|r|(0..4).map(move|j|r.target-r.x.base[j])).collect());
    vec![core::sample(&v)]
   }else{(0..4).map(|j|core::sample(&core::sorted(r.iter().map(|r|r.target-r.x.base[j]).collect()))).collect()}
  }).collect()};
  Self{id,residuals}
 }
 pub fn support(&self,x:&Features)->Vec<f64>{
  if self.id<=18{return core::sorted(x.base[..if self.id==17{4}else{8}].to_vec());}
  let mut v=Vec::with_capacity(4*M);for j in 0..4{
   for &r in &self.residuals[x.lead-1][if self.id==19{0}else{j}]{v.push(x.base[j]+r);}
  }
  core::sorted(v)
 }
}
#[cfg(test)]
mod tests{
 use super::*;
 fn a(key:f64,origin:i64)->AnalogRow{AnalogRow{key,origin,residual:origin as f64}}
 #[test]fn tie_across_groups(){let mut v=vec![a(1.,3),a(1.,1),a(3.,2),a(3.,4)];v.sort_by(order);assert_eq!(nearest(&v,2.,3),vec![1.,2.,3.]);}
 #[test]fn identical_keys(){let mut v=vec![a(2.,8),a(2.,2),a(2.,4),a(1.,1)];v.sort_by(order);assert_eq!(nearest(&v,2.,2),vec![2.,4.]);}
 #[test]fn nearest_bruteforce(){let mut v:Vec<_>=(0..301).map(|i|a((i%13)as f64,i)).collect();v.sort_by(order);for x in [-3.,0.,3.5,6.,20.]{let mut b=v.clone();b.sort_by(|a,b|(a.key-x).abs().total_cmp(&(b.key-x).abs()).then(a.origin.cmp(&b.origin)));assert_eq!(nearest(&v,x,128),b[..128].iter().map(|r|r.residual).collect::<Vec<_>>());}}
 fn rows()->Vec<Row>{(1..=24).map(|h|Row{x:Features{origin:1,lead:h,month:1,base:[1.,2.,3.,4.,5.,6.,7.,8.]},target:10.,daily:0.}).collect()}
 #[test]fn raw_members(){let r=rows();let a:Vec<_>=r.iter().collect();assert_eq!(Mixture::fit(17,&a).support(&r[0].x),vec![1.,2.,3.,4.]);assert_eq!(Mixture::fit(18,&a).support(&r[0].x).len(),8);}
 #[test]fn member_specific(){let r=rows();let a:Vec<_>=r.iter().collect();assert_eq!(Mixture::fit(20,&a).support(&r[0].x),vec![10.;396]);}
 #[test]fn pooled_member(){let r=rows();let a:Vec<_>=r.iter().collect();let v=Mixture::fit(19,&a).support(&r[0].x);assert_eq!(v.len(),396);assert_eq!(v[0],7.);assert_eq!(v[395],13.);}
}
