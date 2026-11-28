//! Audit-only reconstruction from persisted parameters and frozen calibration rows.
//! Does not import probabilistic fitting or scoring code.
use std::{collections::BTreeMap,fs,path::Path};
#[derive(Clone,Copy,Debug)]
pub struct Row{pub origin:i64,pub lead:usize,pub target:f64,pub base:[f64;8]}
impl Row{
 pub fn center(&self,b:bool)->f64{if b{(self.base[0]+self.base[1]+self.base[2]+self.base[3])/4.}else{self.base[0]}}
 pub fn spread(&self)->f64{let(mut lo,mut hi)=(self.base[0],self.base[0]);for x in self.base[..4].iter().copied(){lo=lo.min(x);hi=hi.max(x);}hi-lo}
}
pub fn number(s:&str)->f64{let x:f64=s.parse().unwrap();assert!(x.is_finite());x}
pub fn read_rows(path:&Path,year:u16)->Vec<Row>{
 let text=fs::read_to_string(path).unwrap();let mut lines=text.lines();
 assert_eq!(lines.next().unwrap(),"fold,origin,lead,target,daily,I,L,R,M,S,A,H,X");
 let mut v=Vec::new();
 for l in lines{let p:Vec<_>=l.split(',').collect();assert_eq!(p.len(),13);assert_eq!(p[0].parse::<u16>().unwrap(),year);let origin=p[1].parse().unwrap();let lead=p[2].parse().unwrap();let target=number(p[3]);let base=std::array::from_fn(|i|number(p[5+i]));
 assert_eq!(lead,v.len()%24+1);if let Some(prev)=v.last(){let prev:&Row=prev;assert!(if lead==1{origin>prev.origin}else{origin==prev.origin});}
 v.push(Row{origin,lead,target,base});}
 assert_eq!(v.len(),24*match year{2020=>7543,2021=>7501,2022=>7639,_=>panic!("year")});v
}
fn sample(v:&[f64],p:f64)->f64{let x=p*v.len() as f64;let k=if (x-x.round()).abs()<1e-10{x.round()}else{x.ceil()};v[(k as usize).clamp(1,v.len())-1]}
fn empirical99(mut v:Vec<f64>)->Vec<f64>{v.sort_by(f64::total_cmp);(0..99).map(|j|sample(&v,(j as f64+0.5)/99.)).collect()}
fn support(p:&[&str])->Vec<f64>{assert_eq!(p.len(),99);let v:Vec<_>=p.iter().map(|s|number(s)).collect();assert!(v.windows(2).all(|w|w[0]<=w[1]));v}
fn flag(s:&str)->bool{s.parse().unwrap()}
fn count(s:&str)->usize{s.parse().unwrap()}
#[derive(Debug)]
enum Node{Leaf(Vec<f64>),Split(usize,f64)}
type Nodes=BTreeMap<(usize,String),Node>;
#[derive(Clone,Copy,Debug)]
struct Analog{key:f64,origin:i64,residual:f64}
#[derive(Debug)]
struct Index{groups:Vec<(f64,Vec<Analog>)>}
impl Index{
 fn new(mut v:Vec<Analog>)->Self{
  v.sort_by(|a,b|a.key.total_cmp(&b.key).then(a.origin.cmp(&b.origin)));
  let mut groups:Vec<(f64,Vec<Analog>)>=Vec::new();
  for a in v{if groups.last().is_some_and(|g|g.0==a.key){groups.last_mut().unwrap().1.push(a);}else{groups.push((a.key,vec![a]));}}
  Self{groups}
 }
 fn nearest(&self,x:f64)->Vec<f64>{
  let g=&self.groups;let k=g.partition_point(|a|a.0<x);let(mut left,mut right)=(k.checked_sub(1),k);let(mut li,mut ri)=(0,0);let mut out=Vec::with_capacity(128);
  while out.len()<128{
   let l=left.map(|i|&g[i].1[li]);let r=g.get(right).map(|a|&a.1[ri]);
   let choose=match(l,r){(Some(a),Some(b))=>((a.key-x).abs(),a.origin)<((b.key-x).abs(),b.origin),(Some(_),None)=>true,(None,Some(_))=>false,_=>panic!("analog exhaustion")};
   if choose{out.push(l.unwrap().residual);li+=1;let i=left.unwrap();if li==g[i].1.len(){left=i.checked_sub(1);li=0;}}
   else{out.push(r.unwrap().residual);ri+=1;if ri==g[right].1.len(){right+=1;ri=0;}}
  }out
 }
}
#[derive(Debug)]
enum Kind{
 Residual{b:bool,global:bool,conditional:bool,thresholds:Vec<f64>,groups:Vec<Vec<f64>>},
 Quantile{b:bool,curves:Vec<[(f64,f64);19]>},
 Tree{b:bool,nodes:Nodes},
 Analog{b:bool,by_spread:bool,index:Vec<Index>},
 Member{supports:Vec<Vec<Vec<f64>>>},
 Conformal{b:bool,normalized:bool,bands:Vec<[(f64,f64);4]>},
}
pub struct Reconstruct{id:u8,kind:Kind}
pub enum Forecast{Distribution(Vec<f64>,bool),Intervals(f64,[(f64,f64);4])}
impl Reconstruct{
 pub fn load(path:&Path,year:u16,id:u8,cal:&[Row])->Self{
  let raw=fs::read_to_string(path).unwrap();let mut lines=raw.lines();
  assert_eq!(lines.next().unwrap(),format!("recipe,{year},P{id:02},calibration_rows,{}",cal.len()));
  let h:Vec<_>=lines.next().unwrap().split(',').collect();let mut rows=lines.map(|l|l.split(',').collect::<Vec<_>>());
  let per=cal.len()/24;assert_eq!(per,if year==2021{7543}else{15044});
  let kind=match id{
   1..=9=>{
    assert_eq!(h.len(),4);assert_eq!(h[0],"residual");let b=flag(h[1]);let global=flag(h[2]);let conditional=flag(h[3]);
    assert_eq!(b,matches!(id,2|4|6|7|8|9));assert_eq!(global,id<=2);assert_eq!(conditional,matches!(id,7|9));
    let thresholds=if conditional{(1..=24).map(|lead|{let p=rows.next().unwrap();assert_eq!(p.len(),3);assert_eq!(p[0],"threshold");assert_eq!(count(p[1]),lead);number(p[2])}).collect()}else{Vec::new()};
    let n=if global{1}else if conditional{48}else{24};let mut groups=Vec::new();let mut counts=Vec::new();
    for i in 0..n{let p=rows.next().unwrap();assert_eq!(p.len(),105);assert_eq!(p[0],"group");assert_eq!(count(p[1]),i);let c=count(p[2]);let fallback=flag(p[3]);number(p[4]);let sd=number(p[5]);assert!(sd>=if id>=5{0.05}else{0.});assert_eq!(fallback,conditional&&c<128);counts.push(c);groups.push(support(&p[6..]));}
    if global{assert_eq!(counts,[cal.len()]);}else if conditional{for l in 0..24{assert_eq!(counts[2*l]+counts[2*l+1],per);let c=cal.iter().filter(|r|r.lead==l+1&&r.spread()<=thresholds[l]).count();assert_eq!(counts[2*l],c);}}else{assert!(counts.iter().all(|&c|c==per));}
    Kind::Residual{b,global,conditional,thresholds,groups}
   }
   10..=11=>{
    assert_eq!(h,["quantile",if id==11{"true"}else{"false"}]);let b=id==11;
    let mut curves=Vec::new();for lead in 1..=24{let mut c=[(0.,0.);19];for j in 1..=19{let p=rows.next().unwrap();assert_eq!(p.len(),6);assert_eq!(p[0],"curve");assert_eq!(count(p[1]),lead);assert_eq!(count(p[2]),j);let a=number(p[3]);let s=number(p[4]);assert!([0.,-0.25,0.25,-0.5,0.5,-1.,1.,-2.,2.].contains(&s));assert!(number(p[5])>=0.);c[j-1]=(a,s);}curves.push(c);}
    Kind::Quantile{b,curves}
   }
   12..=13=>{
    assert_eq!(h,["tree",if id==13{"true"}else{"false"}]);let b=id==13;
    let mut nodes=Nodes::new();let mut counts=BTreeMap::new();
    for p in rows.by_ref(){
     let lead=count(p[1]);assert!((1..=24).contains(&lead));let path=p[2];assert!(path.starts_with('R')&&path.len()<=3&&path[1..].chars().all(|c|matches!(c,'L'|'R')));let n=count(p[3]);assert!(n>=128);let imp=number(p[4]);assert!(imp>=-1e-12);let key=(lead,path.to_owned());
     let node=match p[0]{"leaf"=>{assert_eq!(p.len(),104);Node::Leaf(support(&p[5..]))},"split"=>{assert_eq!(p.len(),8);assert!(path.len()<3&&n>=256);assert!(number(p[5])<imp-1e-12);let f=count(p[6]);assert!(f<=1);Node::Split(f,number(p[7]))},_=>panic!("node")};
     assert!(counts.insert(key.clone(),n).is_none());assert!(nodes.insert(key,node).is_none());
    }
    for lead in 1..=24{assert_eq!(counts[&(lead,"R".into())],per);}
    for ((l,p),n) in &nodes{if p.len()>1{assert!(matches!(nodes[&(*l,p[..p.len()-1].to_owned())],Node::Split(..)));}if matches!(n,Node::Split(..)){assert_eq!(counts[&(*l,p.clone())],counts[&(*l,format!("{p}L"))]+counts[&(*l,format!("{p}R"))]);}}
    Kind::Tree{b,nodes}
   }
   14..=16=>{
    let b=id!=14;let by_spread=id==16;assert_eq!(h,["analog",if b{"true"}else{"false"},if by_spread{"true"}else{"false"},"128",if year==2021{"2020"}else{"2020+2021"}]);
    let mut index=Vec::new();for lead in 1..=24{let p=rows.next().unwrap();assert_eq!(p.len(),3);assert_eq!(p[0],"analog-count");assert_eq!(count(p[1]),lead);assert_eq!(count(p[2]),per);
     let v=cal.iter().filter(|r|r.lead==lead).map(|r|Analog{key:if by_spread{r.spread()}else{r.center(b)},origin:r.origin,residual:r.target-r.center(b)}).collect();index.push(Index::new(v));}
    Kind::Analog{b,by_spread,index}
   }
   17..=20=>{
    assert_eq!(h.len(),2);assert_eq!(h[0],"mixture");assert_eq!(count(h[1]),id as usize);let mut supports=Vec::new();
    if id>=19{for lead in 1..=24{let mut group=Vec::new();for j in 0..if id==19{1}else{4}{let p=rows.next().unwrap();assert_eq!(p.len(),102);assert_eq!(p[0],"member-support");assert_eq!(count(p[1]),lead);assert_eq!(count(p[2]),j);group.push(support(&p[3..]));}supports.push(group);}}
    Kind::Member{supports}
   }
   21..=24=>{
    let b=id!=21;let normalized=id==24;assert_eq!(h,["conformal",if b{"true"}else{"false"},if normalized{"true"}else{"false"}]);let mut bands=Vec::new();
    for lead in 1..=24{let p=rows.next().unwrap();assert_eq!(p.len(),11);assert_eq!(p[0],"bands");assert_eq!(count(p[1]),lead);assert_eq!(count(p[2]),per);
     let v:[(f64,f64);4]=std::array::from_fn(|i|(number(p[3+2*i]),number(p[4+2*i])));assert!(v.iter().all(|(l,u)|l+u>=0.));assert!(v.windows(2).all(|w|w[1].0>=w[0].0&&w[1].1>=w[0].1));bands.push(v);}
    Kind::Conformal{b,normalized,bands}
   }
   _=>panic!("recipe")
  };
  assert!(rows.next().is_none());Self{id,kind}
 }
 pub fn forecast(&self,r:&Row)->Forecast{
  let h=r.lead-1;let d=r.spread();let mut cross=false;
  let mut v=match &self.kind{
   Kind::Residual{b,global,conditional,thresholds,groups}=>{let i=if *global{0}else if *conditional{2*h+usize::from(d>thresholds[h])}else{h};groups[i].iter().map(|x|r.center(*b)+x).collect()},
   Kind::Quantile{b,curves}=>{let mut q:Vec<_>=curves[h].iter().map(|(a,s)|a+s*d).collect();cross=q.windows(2).any(|w|w[0]>w[1]);q.sort_by(f64::total_cmp);(0..99).map(|j|{let p=((j as f64+0.5)/99.*20.-1.).clamp(0.,18.);let k=p.floor()as usize;let u=q[(k+1).min(18)];r.center(*b)+(q[k]+(u-q[k])*(p-k as f64)).clamp(q[k],u)}).collect()},
   Kind::Tree{b,nodes}=>{let mut path="R".to_owned();let c=r.center(*b);loop{match &nodes[&(r.lead,path.clone())]{Node::Leaf(v)=>break v.iter().map(|x|c+x).collect(),Node::Split(f,t)=>path.push(if (if *f==0{c}else{d})<=*t{'L'}else{'R'})}}},
   Kind::Analog{b,by_spread,index}=>{let c=r.center(*b);empirical99(index[h].nearest(if *by_spread{d}else{c})).iter().map(|x|c+x).collect()},
   Kind::Member{supports}=>{if self.id<=18{r.base[..if self.id==17{4}else{8}].to_vec()}else{let mut v=Vec::new();for j in 0..4{for x in &supports[h][if self.id==19{0}else{j}]{v.push(r.base[j]+x);}}v}},
   Kind::Conformal{b,normalized,bands}=>{let c=r.center(*b);let s=if *normalized{d.max(0.25)}else{1.};return Forecast::Intervals(c,bands[h].map(|(l,u)|(c-s*l,c+s*u)))}
  };
  v.sort_by(f64::total_cmp);Forecast::Distribution(v,cross)
 }
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn analog_index_brute(){let v:Vec<_>=(0..301).map(|i|Analog{key:(i%13)as f64,origin:i,residual:i as f64}).collect();let idx=Index::new(v.clone());for x in [-3.,0.,3.5,6.,20.]{let mut b=v.clone();b.sort_by(|a,b|(a.key-x).abs().total_cmp(&(b.key-x).abs()).then(a.origin.cmp(&b.origin)));assert_eq!(idx.nearest(x),b[..128].iter().map(|r|r.residual).collect::<Vec<_>>());}}
 #[test]fn sample_ties(){assert_eq!(sample(&[1.,2.,3.,4.],0.5),2.);assert_eq!(empirical99(vec![3.,3.]),vec![3.;99]);}
}
