//! C008 D09: finite symbolic grammar and residualized forward selection.
use super::{TrainingRow,D,Y};
use super::decomposition::Readout;
use super::kernel_ridge::target_mean;
use std::{collections::HashSet,time::Instant};
const LAMBDA:f64=0.001;
const GUARD:f64=1400.;
#[derive(Clone,Debug)]
enum Expr{Terminal(usize),Unary(u8,Box<Expr>),Binary(u8,Box<Expr>,Box<Expr>)}
const NAMES:[&str;20]=["lag0","lag1","lag3","lag6","lag12","lag24","lag48","lag71","mean6","mean24","mean72","diff24","x72","x73","x74","x75","x76","x77","x78","x79"];
fn terminal(x:&[f64;D])->[f64;20]{
    let mut t=[0.;20];for(i,lag)in [0,1,3,6,12,24,48,71].into_iter().enumerate(){t[i]=x[71-lag];}
    for(i,n)in [6,24,72].into_iter().enumerate(){t[8+i]=x[72-n..72].iter().sum::<f64>()/n as f64;}
    t[11]=x[71]-x[47];t[12..].copy_from_slice(&x[72..80]);t
}
impl Expr{
    fn eval(&self,t:&[f64;20])->f64{match self{Self::Terminal(i)=>t[*i],Self::Unary(op,a)=>{let x=a.eval(t);match op{0=>x*x,1=>x.tanh(),_=>x.abs()}},Self::Binary(op,a,b)=>{let(x,y)=(a.eval(t),b.eval(t));match op{0=>x+y,1=>x-y,_=>x*y}}}}
    fn name(&self)->String{match self{Self::Terminal(i)=>NAMES[*i].into(),Self::Unary(op,a)=>format!("{}({})",["sq","tanh","abs"][*op as usize],a.name()),Self::Binary(op,a,b)=>{let(mut x,mut y)=(a.name(),b.name());if *op!=1&&x>y{std::mem::swap(&mut x,&mut y);}format!("({x}{}{y})",["+","-","*"][*op as usize])}}}
    #[cfg(test)]fn depth(&self)->usize{match self{Self::Terminal(_)=>0,Self::Unary(_,a)=>1+a.depth(),Self::Binary(_,a,b)=>1+a.depth().max(b.depth())}}
}
fn push(v:&mut Vec<Expr>,seen:&mut HashSet<String>,e:Expr){if seen.insert(e.name()){v.push(e)}}
fn enumerate(depth:usize)->Vec<Expr>{
    let mut v=Vec::new();let mut seen=HashSet::new();
    for i in 0..20{push(&mut v,&mut seen,Expr::Terminal(i));}
    for op in 0..3{for i in 0..20{push(&mut v,&mut seen,Expr::Unary(op,Box::new(Expr::Terminal(i))));}}
    for op in 0..3{for i in 0..20{for j in 0..20{if op!=1&&i>j{continue}push(&mut v,&mut seen,Expr::Binary(op,Box::new(Expr::Terminal(i)),Box::new(Expr::Terminal(j))));}}}
    if depth==2{let d1=v[20..].to_vec();let cap=v.len()+1024;
        'outer:for op in 0..3{for a in &d1{push(&mut v,&mut seen,Expr::Unary(op,Box::new(a.clone())));if v.len()==cap{break 'outer}}}
        'binary:for op in 0..3{for a in d1.iter().take(32){for i in 0..8{if v.len()==cap{break 'binary}push(&mut v,&mut seen,Expr::Binary(op,Box::new(a.clone()),Box::new(Expr::Terminal(i))));}}}
    }v
}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>GUARD{Err("runtime guard".into())}else{Ok(())}}
struct Column{expr:Expr,mean:f64,sd:f64,z:Vec<f64>}
struct Term{expr:Expr,mean:f64,sd:f64}
pub(super) struct Model{terms:Vec<Term>,readout:Option<Readout>,mean:[f64;Y],pub candidates:usize,pub valid:usize,pub steps:Vec<(String,f64,f64,f64)>}
fn residualize(v:&mut[f64],q:&[Vec<f64>]){for _ in 0..2{for u in q{let a=dot(v,u);for(x,y)in v.iter_mut().zip(u){*x-=a*y;}}}}
fn best(cols:&[Column],selected:&[usize],q:&[Vec<f64>],res:&[Vec<f64>],now:Instant)->Result<Option<(usize,f64,Vec<f64>)>,String>{
    let n=res[0].len();let mut winner=None;let mut score=1e-10;
    for(i,c)in cols.iter().enumerate(){guard(now)?;if selected.contains(&i){continue}
        let mut z=c.z.clone();residualize(&mut z,q);let ss=dot(&z,&z);if ss<=1e-12*n as f64{continue}
        let gain=res.iter().map(|r|dot(&z,r).powi(2)).sum::<f64>()/(ss*n as f64*Y as f64);
        if !gain.is_finite(){return Err("nonfinite symbolic gain".into())}
        if gain>score{score=gain;winner=Some((i,gain,z));}
    }Ok(winner)
}
impl Model{
    pub(super) fn fit(rows:&[&TrainingRow],depth:usize,max_terms:usize,now:Instant)->Result<Self,String>{
        if rows.is_empty(){return Err("empty symbolic training".into())}
        let n=rows.len();let inputs:Vec<_>=rows.iter().map(|r|terminal(&r.input.x)).collect();let grammar=enumerate(depth);let candidates=grammar.len();let mut cols=Vec::new();
        for expr in grammar{guard(now)?;let mut z:Vec<_>=inputs.iter().map(|t|expr.eval(t)).collect();let mean=z.iter().sum::<f64>()/n as f64;for v in &mut z{*v-=mean;}let sd=(dot(&z,&z)/n as f64).sqrt();if !sd.is_finite()||sd==0.||z.iter().any(|v|!v.is_finite()){continue}for v in &mut z{*v/=sd;}cols.push(Column{expr,mean,sd,z});}
        let valid=cols.len();let mean=target_mean(rows);let mut res:Vec<Vec<f64>>=(0..Y).map(|h|rows.iter().map(|r|r.normalized_target[h]-mean[h]).collect()).collect();let mut selected=Vec::new();let mut q=Vec::new();let mut steps=Vec::new();let mut readout=None;
        for _ in 0..max_terms{let Some((idx,gain,mut z))=best(&cols,&selected,&q,&res,now)? else{break};let norm=dot(&z,&z).sqrt();for v in &mut z{*v/=norm;}q.push(z);selected.push(idx);
            let features:Vec<Vec<f64>>=(0..n).map(|i|selected.iter().map(|&j|cols[j].z[i]).collect()).collect();
            let(rd,err)=Readout::fit(rows,&features,LAMBDA,now)?;let mut mse=0.;for(i,v)in features.iter().enumerate(){let pred=rd.predict(v)?;for h in 0..Y{res[h][i]=rows[i].normalized_target[h]-pred[h];mse+=res[h][i]*res[h][i];}}
            steps.push((cols[idx].expr.name(),gain,mse/(n*Y)as f64,err));readout=Some(rd);
        }
        let terms=selected.iter().map(|&i|Term{expr:cols[i].expr.clone(),mean:cols[i].mean,sd:cols[i].sd}).collect();
        Ok(Self{terms,readout,mean,candidates,valid,steps})
    }
    pub(super) fn predict(&self,x:&[f64;D])->Result<[f64;Y],String>{
        let t=terminal(x);match &self.readout{None=>Ok(self.mean),Some(rd)=>{let z:Vec<_>=self.terms.iter().map(|v|(v.expr.eval(&t)-v.mean)/v.sd).collect();rd.predict(&z)}}
    }
    pub(super) fn diagnostics(&self,id:&str,label:&str)->Result<(),String>{
        let d=self.terms.len();println!("symbolic-grammar,{id},{label},{},{},{d}",self.candidates,self.valid);
        let intercept=match &self.readout{Some(r)=>r.predict(&vec![0.;d])?,None=>self.mean};println!("symbolic-intercept,{id},{label},{}",intercept.iter().map(|v|format!("{v:.17e}")).collect::<Vec<_>>().join(","));
        for(i,term)in self.terms.iter().enumerate(){let (name,gain,mse,err)=&self.steps[i];let mut z=vec![0.;d];z[i]=1.;let p=self.readout.as_ref().unwrap().predict(&z)?;
            let coef=p.iter().zip(intercept).map(|(a,b)|format!("{:.17e}",a-b)).collect::<Vec<_>>().join(",");
            println!("symbolic-term,{id},{label},{},{name},{:.17e},{:.17e},{gain:.17e},{mse:.17e},{err:.17e},{coef}",i+1,term.mean,term.sd);
        }Ok(())
    }
}
#[cfg(test)]
mod tests{
use super::*;use super::super::OriginInput;
fn row(v:f64)->TrainingRow{let mut x=[0.;D];x[71]=v;let y=v*v;TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:[y;Y],normalized_target:[y;Y]}}
#[test]fn grammar_order_cap(){let a=enumerate(1);let b=enumerate(2);assert_eq!(a.len(),900);assert_eq!(b.len(),1924);assert_eq!(a[0].name(),"lag0");assert_eq!(a[20].name(),"sq(lag0)");assert!(a.iter().all(|x|x.depth()<=1));assert!(b.iter().all(|x|x.depth()<=2));assert_eq!(b.iter().map(Expr::name).collect::<HashSet<_>>().len(),b.len());assert_eq!(b.iter().map(Expr::name).collect::<Vec<_>>(),enumerate(2).iter().map(Expr::name).collect::<Vec<_>>());}
#[test]fn known_expression(){let rows:Vec<_>=(-10..=10).map(|i|row(i as f64/10.)).collect();let refs:Vec<_>=rows.iter().collect();let m=Model::fit(&refs,1,1,Instant::now()).unwrap();assert_eq!(m.steps.len(),1);assert!((m.predict(&row(0.7).input.x).unwrap()[0]-0.49).abs()<0.001);}
#[test]fn no_mutation_or_target_at_prediction(){let rows:Vec<_>=(-5..=5).map(|i|row(i as f64)).collect();let refs:Vec<_>=rows.iter().collect();let m=Model::fit(&refs,1,2,Instant::now()).unwrap();let x=row(0.3).input.x;let a=m.predict(&x).unwrap();m.predict(&row(999.).input.x).ok();assert_eq!(a,m.predict(&x).unwrap());}
#[test]fn exact_tie_and_collinear(){let z=vec![-1.,0.,1.];let cols=vec![Column{expr:Expr::Terminal(0),mean:0.,sd:1.,z:z.clone()},Column{expr:Expr::Terminal(1),mean:0.,sd:1.,z:z.clone()}];let res=vec![z.clone();Y];let(i,_,q)=best(&cols,&[],&[],&res,Instant::now()).unwrap().unwrap();assert_eq!(i,0);let norm=dot(&q,&q).sqrt();let q:Vec<_>=q.iter().map(|v|v/norm).collect();assert!(best(&cols,&[0],&[q],&res,Instant::now()).unwrap().is_none());}
}
