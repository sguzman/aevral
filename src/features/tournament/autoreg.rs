//! C005 S01–S04: one-step least squares with causal recursive forecasting.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow};
use crate::{benchmark::Metrics,windows::Calendar};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;
#[derive(Clone,Copy,Debug)]
enum Kind{Ar(usize,bool),Var(bool)}
impl Kind{
    fn parse(s:&str)->Result<Self,String>{
        match s{"var-1"=>Ok(Self::Var(false)),"varx-1"=>Ok(Self::Var(true)),_=>{
            let (prefix,p)=s.split_once('-').ok_or("unknown configuration")?;
            let p=p.parse::<usize>().map_err(|_|"invalid order")?;
            if ![6,24,48,72].contains(&p)||!["ar","arx"].contains(&prefix){return Err("unknown AR configuration".into())}
            Ok(Self::Ar(p,prefix=="arx"))
        }}
    }
    fn dim(self)->usize{match self{Self::Ar(p,x)=>1+p+if x{8}else{0},Self::Var(x)=>6+if x{4}else{0}}}
    fn outputs(self)->usize{match self{Self::Ar(..)=>1,Self::Var(..)=>5}}
}
fn state(x:&OriginInput)->[f64;5]{[x.x[71],x.x[76],x.x[77],x.x[78],x.x[79]]}
fn design(kind:Kind,history:&[f64],weather:&[f64;4],time:i64)->Vec<f64>{
    let mut z=vec![1.];
    match kind{
        Kind::Ar(p,x)=>{z.extend(history.iter().rev().take(p).copied());if x{z.extend(Calendar::Both.features(time));z.extend(weather)}},
        Kind::Var(x)=>{z.push(*history.last().unwrap());z.extend(weather);if x{z.extend(Calendar::Both.features(time))}}
    }z
}
pub(super) struct Qr{r:Vec<Vec<f64>>,z:Vec<Vec<f64>>,n:usize,m:usize}
impl Qr{
    pub(super) fn new(p:usize,m:usize)->Self{Self{r:vec![vec![0.;p];p],z:vec![vec![0.;m];p],n:0,m}}
    pub(super) fn add(&mut self,mut a:Vec<f64>,mut y:Vec<f64>){
        self.n+=1;
        for j in 0..a.len(){
            let d=self.r[j][j].hypot(a[j]);if d==0.{continue}
            let c=self.r[j][j]/d;let s=a[j]/d;self.r[j][j]=d;
            for k in j+1..a.len(){let old=self.r[j][k];self.r[j][k]=c*old+s*a[k];a[k]= -s*old+c*a[k]}
            for k in 0..self.m{let old=self.z[j][k];self.z[j][k]=c*old+s*y[k];y[k]= -s*old+c*y[k]}
        }
    }
    pub(super) fn solve(&self)->Result<(Vec<Vec<f64>>,f64),String>{
        let p=self.r.len();let max=(0..p).map(|j|self.r[j][j]).fold(0.,f64::max);
        let min=(0..p).map(|j|self.r[j][j]).fold(f64::INFINITY,f64::min);
        if self.n<p||max==0.||min/max<1e-10{return Err("rank-deficient one-step QR".into())}
        let mut b=self.z.clone();
        for j in (0..p).rev(){for k in j+1..p{for h in 0..self.m{b[j][h]-=self.r[j][k]*b[k][h]}}for x in &mut b[j]{*x/=self.r[j][j]}}
        if b.iter().flatten().any(|v|!v.is_finite()){return Err("nonfinite coefficients".into())}
        Ok((b,min/max))
    }
}
fn predict_step(b:&[Vec<f64>],z:&[f64])->Vec<f64>{
    let mut y=vec![0.;b[0].len()];for (row,x) in b.iter().zip(z){for h in 0..y.len(){y[h]+=x*row[h]}}y
}
// Finite-horizon homogeneous companion impulse diagnostic, not exact spectral radius.
fn impulse(kind:Kind,b:&[Vec<f64>])->f64{
    let n=match kind{Kind::Ar(p,_)=>p,Kind::Var(_)=>5};
    let mut max=1f64;
    for j in 0..n{
        let mut v=vec![0.;n];v[j]=1.;
        for _ in 0..256{
            match kind{
                Kind::Ar(p,_)=>{let y=(0..p).map(|k|b[k+1][0]*v[k]).sum::<f64>();v.rotate_right(1);v[0]=y},
                Kind::Var(_)=>{let next=(0..5).map(|h|(0..5).map(|k|b[k+1][h]*v[k]).sum::<f64>()).collect();v=next}
            }
            for &x in &v{if !x.is_finite(){return f64::INFINITY}max=max.max(x.abs())}
        }
    }max
}
struct Model{kind:Kind,b:Vec<Vec<f64>>,mean:f64,sd:f64}
impl Model{
    fn normalized(&self,x:&OriginInput)->[f64;Y]{
        let mut hist=x.x[..72].to_vec();let mut weather:[f64;4]=x.x[76..80].try_into().unwrap();
        let mut out=[0.;Y];for (h,o) in out.iter_mut().enumerate(){
            let z=design(self.kind,&hist,&weather,x.origin+60*h as i64);
            let y=predict_step(&self.b,&z);*o=y[0];hist.push(y[0]);
            if matches!(self.kind,Kind::Var(_)){weather.copy_from_slice(&y[1..5])}
        }out
    }
}
impl Predictor for Model{fn predict(&self,x:&OriginInput)->[f64;Y]{self.normalized(x).map(|y|y*self.sd+self.mean)}}
fn fit(rows:&[TrainingRow],kind:Kind,mean:f64,sd:f64)->Result<(Model,usize,f64,f64,f64),String>{
    let refs:Vec<_>=rows.iter().collect();
    fit_refs(&refs,kind,mean,sd)
}
fn fit_refs(rows:&[&TrainingRow],kind:Kind,mean:f64,sd:f64)->Result<(Model,usize,f64,f64,f64),String>{
    let mut qr=Qr::new(kind.dim(),kind.outputs());
    let mut mse=0.;
    let mut samples=Vec::new(); // small bounded diagnostic sample, never evaluation
    for (i,r) in rows.iter().enumerate(){
        let weather=r.input.x[76..80].try_into().unwrap();
        let y=match kind{
            Kind::Ar(..)=>vec![r.normalized_target[0]],
            Kind::Var(..)=>{let Some(next)=rows.get(i+1)else{continue};if next.input.origin-r.input.origin!=60{continue}
                if (next.input.x[71]-r.normalized_target[0]).abs()>1e-8{return Err("VAR transition mismatch".into())}
                state(&next.input).to_vec()}
        };
        let z=design(kind,&r.input.x[..72],&weather,r.input.origin);
        if i%127==0{samples.push((z.clone(),y.clone()))}
        qr.add(z,y);
    }
    let(b,rank)=qr.solve()?;
    if samples.is_empty(){return Err("empty diagnostic sample".into())}
    for(z,y)in &samples{mse+=predict_step(&b,z).iter().zip(y).map(|(p,t)|(p-t).powi(2)).sum::<f64>()}
    mse/= (samples.len()*kind.outputs())as f64;
    let growth=impulse(kind,&b);
    if !growth.is_finite()||growth>1e6{return Err("divergent homogeneous impulse response".into())}
    Ok((Model{kind,b,mean,sd},qr.n,rank,mse,growth))
}
pub fn run(data:&Path,id:&str,output:&Path,selection:&str)->Result<(),String>{
    let labels:Vec<String>=match selection{
        "ar-all"=>[6,24,48,72].iter().map(|p|format!("ar-{p}")).collect(),
        "arx-all"=>[6,24,48,72].iter().map(|p|format!("arx-{p}")).collect(),
        "var-all"=>vec!["var-1".into(),"varx-1".into()],
        _=>vec![selection.into()]
    };
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);
    println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    for label in labels{
        let kind=Kind::parse(&label)?;let now=Instant::now();
        match fit(&f.train,kind,f.target_mean,f.target_sd){
            Ok((model,n,rank,mse,growth))=>{
                println!("ar-config,{id},{label},{n},{},{},{rank:.12e},{mse:.12e},{growth:.12e},{:.6}",kind.dim(),kind.dim()*kind.outputs(),now.elapsed().as_secs_f64());
                for(i,b)in model.b.iter().enumerate(){println!("ar-coef,{id},{label},{i},{b:?}")}
                score(&model,&label,&f,&mut out)?;
            },Err(e)=>println!("blocked,{id},{label},{e}"),
        }
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
/// C012 ARX: independent one-step examples, not synthetic adjacent transitions.
pub(super) fn optimize(f:&super::PreparedFold,rows:&[&TrainingRow],label:&str,order:usize,out:&mut impl Write)->Result<(),String>{
    if ![6,12,24,36,48,72].contains(&order){return Err("invalid C012 ARX order".into())}
    let id=f.fold.id;let kind=Kind::Ar(order,true);let now=Instant::now();
    let (model,n,rank,mse,growth)=match fit_refs(rows,kind,f.target_mean,f.target_sd){
        Ok(v)=>v,Err(e)=>{println!("blocked,{id},{label},{e}");return Ok(())}
    };
    if n!=rows.len(){return Err("C012 ARX row budget mismatch".into())}
    let fit=now.elapsed().as_secs_f64();
    println!("ar-config,{id},{label},{n},{},{},{rank:.12e},{mse:.12e},{growth:.12e},{fit:.6}",kind.dim(),kind.dim()*kind.outputs());
    let timer=Instant::now();score(&model,label,f,out)?;
    println!("optimization-cost,{id},{label},{fit:.6},{:.6},{}",timer.elapsed().as_secs_f64(),kind.dim()*kind.outputs());Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn known_qr(){let mut q=Qr::new(3,2);for i in 0..100{let a=i as f64/100.;let b=(i as f64).sin();q.add(vec![1.,a,b],vec![1.+2.*a-3.*b,-2.+4.*b])}let(w,_)=q.solve().unwrap();for (a,b) in w.iter().flatten().zip([1.,-2.,2.,0.,-3.,4.]){assert!((a-b).abs()<1e-9)}}
    #[test]fn recursion(){let mut x=[0.;80];x[71]=2.;let input=OriginInput{origin:0,x,recent:[0.;24]};let mut b=vec![vec![0.];7];b[1][0]=0.5;let m=Model{kind:Kind::Ar(6,false),b,mean:0.,sd:1.};for(h,p)in m.predict(&input).iter().enumerate(){assert!((p-2.*0.5_f64.powi(h as i32+1)).abs()<1e-10)}}
    #[test]fn joint_recursion(){let mut x=[0.;80];x[71]=2.;x[76]=3.;let input=OriginInput{origin:0,x,recent:[0.;24]};let mut b=vec![vec![0.;5];6];b[2][0]=1.;b[2][1]=0.5;let m=Model{kind:Kind::Var(false),b,mean:0.,sd:1.};let y=m.predict(&input);assert_eq!(y[0],3.);assert_eq!(y[1],1.5);}
    #[test]fn calendar_advances(){let h=[0.;72];let w=[1.,2.,3.,4.];let a=design(Kind::Ar(6,true),&h,&w,0);let b=design(Kind::Ar(6,true),&h,&w,60);assert_ne!(a[7..11],b[7..11]);assert_eq!(a[11..15],b[11..15]);}
}
