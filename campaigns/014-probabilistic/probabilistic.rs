//! C014 distributional forecasting implementation; synthetic tests only until assessment gate.
#[path="../013-ensembles/ensemble.rs"]
pub mod ensemble;
pub mod core;
pub mod residual;
pub mod regression;
pub mod analog;
use ensemble::{Features,Row};
use core::Issued;
#[derive(Debug)]
pub enum Model{
 Residual(residual::Residual),Quantile(regression::Quantile),Tree(regression::Tree),
 Analog(analog::Analog),Mixture(analog::Mixture),Conformal(residual::Conformal),
}
pub fn fit(id:u8,rows:&[&Row])->Model{match id{
 1..=9=>Model::Residual(residual::fit_residual(id,rows)),
 10..=11=>Model::Quantile(regression::Quantile::fit(rows,id==11)),
 12..=13=>Model::Tree(regression::Tree::fit(rows,id==13)),
 14..=16=>Model::Analog(analog::Analog::fit(rows,id!=14,id==16)),
 17..=20=>Model::Mixture(analog::Mixture::fit(id,rows)),
 21..=24=>Model::Conformal(residual::fit_conformal(id,rows)),
 _=>panic!("unknown recipe")
}}
impl Model{
 /// Return exact issued support or the four interval-only bands, plus crossing incidence.
 pub fn support(&self,x:&Features)->(Option<Vec<f64>>,bool){match self{
  Self::Residual(m)=>(Some(m.support(x)),false),
  Self::Quantile(m)=>{let(v,c)=m.support(x);(Some(v),c)},
  Self::Tree(m)=>(Some(m.support(x)),false),
  Self::Analog(m)=>(Some(m.support(x)),false),
  Self::Mixture(m)=>(Some(m.support(x)),false),
  Self::Conformal(_)=>(None,false),
 }}
 pub fn issue(&self,x:&Features,y:f64)->(Issued,bool){
  if let Self::Conformal(m)=self{return(m.issue(x),false);}
  let(v,c)=self.support(x);(core::distribution(&v.unwrap(),y),c)
 }
}
