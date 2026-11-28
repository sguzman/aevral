//! Frozen spread-conditioned quantile regression and depth-2 distributional trees.
use super::{core::{self,M},ensemble::{Features,Row},residual::{center,spread}};
const SLOPES:[f64;9]=[0.,-0.25,0.25,-0.5,0.5,-1.,1.,-2.,2.];
#[derive(Clone,Copy,Debug)]
pub struct Curve{pub intercept:f64,pub slope:f64,pub loss:f64}
#[derive(Debug)]
pub struct Quantile{pub b:bool,pub curves:Vec<[Curve;19]>}
fn curves(rows:&[&Row],b:bool)->[Curve;19]{
 let mut best=[Curve{intercept:0.,slope:0.,loss:f64::INFINITY};19];
 for slope in SLOPES{
  let v=core::sorted(rows.iter().map(|r|r.target-center(&r.x,b)-slope*spread(&r.x)).collect());
  let n=v.len();let mut prefix=vec![0.;n+1];for i in 0..n{prefix[i+1]=prefix[i]+v[i];}
  for j in 0..19{
   let p=(j+1)as f64/20.;let k=core::rank(p,n);let a=v[k];let mid=k+1;
   let loss=((1.-p)*(mid as f64*a-prefix[mid])+p*(prefix[n]-prefix[mid]-(n-mid)as f64*a)).max(0.);
   if loss<best[j].loss{best[j]=Curve{intercept:a,slope,loss};}
  }
 }best
}
impl Quantile{
 pub fn fit(rows:&[&Row],b:bool)->Self{Self{b,curves:(1..=24).map(|h|curves(&rows.iter().copied().filter(|r|r.x.lead==h).collect::<Vec<_>>(),b)).collect()}}
 pub fn support(&self,x:&Features)->(Vec<f64>,bool){
  let c=center(x,self.b);let d=spread(x);
  let mut v:Vec<_>=self.curves[x.lead-1].iter().map(|q|q.intercept+q.slope*d).collect();
  let crossed=v.windows(2).any(|w|w[0]>w[1]);if crossed{v.sort_by(f64::total_cmp);}
  let support=(0..M).map(|j|{
   let p=((j as f64+0.5)/M as f64*20.-1.).clamp(0.,18.);
   let k=p.floor()as usize;let t=p-k as f64;
   c+v[k]*(1.-t)+v[(k+1).min(18)]*t
  }).collect();(support,crossed)
 }
}
#[derive(Clone,Copy,Debug)]
pub struct Cal{pub center:f64,pub spread:f64,pub residual:f64}
fn feature(r:&Cal,f:usize)->f64{if f==0{r.center}else{r.spread}}
fn impurity(v:&[Cal])->f64{
 let s=core::sorted(v.iter().map(|r|r.residual).collect());
 let n=s.len()as f64;
 s.iter().enumerate().map(|(i,x)|(2.*i as f64+1.-n)*x).sum::<f64>()/(n*n)
}
#[derive(Debug)]
pub enum Node{
 Leaf{count:usize,impurity:f64,support:Vec<f64>},
 Split{count:usize,impurity:f64,objective:f64,feature:usize,threshold:f64,left:Box<Node>,right:Box<Node>}
}
impl Node{
 fn fit(v:&[Cal],depth:usize)->Self{
  let n=v.len();let imp=impurity(v);
  let mut best=None;let mut objective=imp;
  if depth<2&&n>=256{
   for f in 0..2{
    let values=core::sorted(v.iter().map(|r|feature(r,f)).collect());
    for p in [0.25,0.5,0.75]{
     let t=core::quantile(&values,p);let(l,r):(Vec<_>,Vec<_>)=v.iter().copied().partition(|x|feature(x,f)<=t);
     if l.len()<128||r.len()<128{continue;}
     let score=(l.len()as f64*impurity(&l)+r.len()as f64*impurity(&r))/n as f64;
     if score<imp-1e-12 && score<objective{objective=score;best=Some((f,t,l,r));}
    }
   }
  }
  if let Some((f,t,l,r))=best{Self::Split{count:n,impurity:imp,objective,feature:f,threshold:t,left:Box::new(Self::fit(&l,depth+1)),right:Box::new(Self::fit(&r,depth+1))}}
  else{Self::Leaf{count:n,impurity:imp,support:core::sample(&core::sorted(v.iter().map(|r|r.residual).collect()))}}
 }
 pub fn leaf(&self,c:f64,d:f64)->&[f64]{match self{
  Self::Leaf{support,..}=>support,
  Self::Split{feature,threshold,left,right,..}=>if (if *feature==0{c}else{d})<=*threshold{left.leaf(c,d)}else{right.leaf(c,d)}
 }}
}
#[derive(Debug)]
pub struct Tree{pub b:bool,pub roots:Vec<Node>}
impl Tree{
 pub fn fit(rows:&[&Row],b:bool)->Self{Self{b,roots:(1..=24).map(|h|{
  let v:Vec<_>=rows.iter().filter(|r|r.x.lead==h).map(|r|{let c=center(&r.x,b);Cal{center:c,spread:spread(&r.x),residual:r.target-c}}).collect();
  Node::fit(&v,0)
 }).collect()}}
 pub fn support(&self,x:&Features)->Vec<f64>{let c=center(x,self.b);self.roots[x.lead-1].leaf(c,spread(x)).iter().map(|r|c+r).collect()}
}
#[cfg(test)]
mod tests{
 use super::*;
 fn row(h:usize,d:f64,r:f64)->Row{Row{x:Features{origin:0,lead:h,month:1,base:[0.,d,0.,0.,0.,0.,0.,0.]},target:r,daily:0.}}
 #[test]fn regression_exact_slope(){let v:Vec<_>=(0..20).map(|i|row(1,i as f64,2.+i as f64)).collect();let refs:Vec<_>=v.iter().collect();let c=curves(&refs,false);assert!(c.iter().all(|x|x.slope==1.&&x.intercept==2.&&x.loss<1e-9));}
 #[test]fn regression_tie(){let v=[row(1,0.,1.),row(1,0.,2.)];let c=curves(&v.iter().collect::<Vec<_>>(),false);assert!(c.iter().all(|x|x.slope==0.));}
 #[test]fn rearranges_crossings(){let mut c=[Curve{intercept:0.,slope:0.,loss:0.};19];for(i,x)in c.iter_mut().enumerate(){x.intercept=-(i as f64);}let q=Quantile{b:false,curves:vec![c;24]};let(v,cross)=q.support(&row(1,0.,0.).x);assert!(cross);assert!(v.windows(2).all(|w|w[0]<=w[1]));assert_eq!(v[0],-18.);assert_eq!(v[98],0.);}
 #[test]fn tree_split_and_tie(){let v:Vec<_>=(0..512).map(|i|{let x=if i<256{0.}else{1.};Cal{center:x,spread:x,residual:10.*x}}).collect();let t=Node::fit(&v,0);if let Node::Split{feature,threshold,..}=&t{assert_eq!(*feature,0);assert_eq!(*threshold,0.);}else{panic!("no split");}assert_eq!(t.leaf(0.,0.)[0],0.);assert_eq!(t.leaf(1.,1.)[0],10.);}
 #[test]fn tree_min_child(){let v=vec![Cal{center:0.,spread:0.,residual:2.};255];assert!(matches!(Node::fit(&v,0),Node::Leaf{count:255,..}));}
 #[test]fn impurity_pair(){let v=[Cal{center:0.,spread:0.,residual:0.},Cal{center:0.,spread:0.,residual:2.}];assert_eq!(impurity(&v),0.5);}
}
