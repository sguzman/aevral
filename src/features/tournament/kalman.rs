//! C005 S10–S11: causal Gaussian structural and dynamic linear models.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const H:usize=72;const Y:usize=24;const M:usize=4;const SAMPLE:usize=1024;
type Vec4=[f64;M];type Mat4=[[f64;M];M];
#[derive(Clone,Copy,Debug,PartialEq)]
enum Kind{Level,Trend,Season,Dlm,DlmTrend}
impl Kind{
    fn parse(s:&str)->Result<Self,String>{match s{
        "struct-level"=>Ok(Self::Level),"struct-trend"=>Ok(Self::Trend),"struct-season"=>Ok(Self::Season),
        "dlm-harmonic"=>Ok(Self::Dlm),"dlm-trend-harmonic"=>Ok(Self::DlmTrend),_=>Err("unknown Kalman configuration".into())}}
    fn dim(self)->usize{match self{Self::Level=>1,Self::Trend=>2,Self::Season|Self::Dlm=>3,Self::DlmTrend=>4}}
    fn trend(self)->bool{matches!(self,Self::Trend|Self::DlmTrend)}
    fn seasonal(self)->bool{matches!(self,Self::Season|Self::Dlm|Self::DlmTrend)}
    fn offset(self)->usize{if self.trend(){2}else{1}}
    fn transition(self)->Mat4{
        let mut f=[[0.;M];M];for i in 0..self.dim(){f[i][i]=1.}
        if self.trend(){f[0][1]=1.}
        if self==Self::Season{let (s,c)=(std::f64::consts::TAU/24.).sin_cos();f[1][1]=c;f[1][2]=s;f[2][1]=-s;f[2][2]=c;}
        f
    }
    fn observation(self,t:i64)->Vec4{
        let mut h=[0.;M];h[0]=1.;
        if self==Self::Season{h[1]=1.}
        if matches!(self,Self::Dlm|Self::DlmTrend){let (s,c)=(std::f64::consts::TAU*t.rem_euclid(1440)as f64/1440.).sin_cos();h[self.offset()]=s;h[self.offset()+1]=c;}
        h
    }
}
#[derive(Clone,Copy,Debug)]
struct Noise{r:f64,level:f64,trend:f64,season:f64}
impl Noise{fn q(self,k:Kind)->Vec4{
    let mut q=[0.;M];q[0]=self.level*self.r;if k.trend(){q[1]=self.trend*self.r}
    if k.seasonal(){q[k.offset()]=self.season*self.r;q[k.offset()+1]=self.season*self.r}q
}}
#[derive(Clone)]
struct State{x:Vec4,p:Mat4,n:usize}
fn dot(a:&Vec4,b:&Vec4,n:usize)->f64{(0..n).map(|i|a[i]*b[i]).sum()}
impl State{
    fn new(k:Kind)->Self{let n=k.dim();let mut p=[[0.;M];M];for i in 0..n{p[i][i]=1.}Self{x:[0.;M],p,n}}
    fn predict(&mut self,f:&Mat4,q:&Vec4){
        let n=self.n;let mut x=[0.;M];let mut fp=[[0.;M];M];let mut p=[[0.;M];M];
        for i in 0..n{x[i]=dot(&f[i],&self.x,n);for j in 0..n{fp[i][j]=(0..n).map(|k|f[i][k]*self.p[k][j]).sum()}}
        for i in 0..n{for j in 0..n{p[i][j]=dot(&fp[i],&f[j],n)}p[i][i]+=q[i]}
        self.x=x;self.p=p;
    }
    fn observe(&self,h:&Vec4,r:f64)->Option<(f64,f64,Vec4)>{
        let mut v=[0.;M];for i in 0..self.n{v[i]=dot(&self.p[i],h,self.n)}
        let mean=dot(h,&self.x,self.n);let var=dot(h,&v,self.n)+r;
        if !mean.is_finite()||!var.is_finite()||var<=0.{None}else{Some((mean,var,v))}
    }
    fn update(&mut self,h:&Vec4,r:f64,y:f64)->Option<()>{
        let(m,s,v)=self.observe(h,r)?;let e=y-m;
        for i in 0..self.n{self.x[i]+=v[i]*e/s}
        for i in 0..self.n{for j in 0..=i{let p=0.5*(self.p[i][j]+self.p[j][i])-v[i]*v[j]/s;self.p[i][j]=p;self.p[j][i]=p}}
        if self.x.iter().any(|v|!v.is_finite())||(0..self.n).any(|i|!self.p[i][i].is_finite()||self.p[i][i]< -1e-10){return None}Some(())
    }
    fn psd(&self)->bool{
        let mut l=[[0.;M];M];
        for i in 0..self.n{for j in 0..=i{let v=self.p[i][j]-(0..j).map(|k|l[i][k]*l[j][k]).sum::<f64>();
            if i==j{let d=v+1e-10;if !d.is_finite()||d<=0.{return false}l[i][i]=d.sqrt()}
            else{l[i][j]=v/l[j][j];if !l[i][j].is_finite(){return false}}
        }}true
    }
}
fn filter(k:Kind,p:Noise,y:&[f64],origin:i64)->Option<State>{
    let mut s=State::new(k);let f=k.transition();let q=p.q(k);
    for (i,&v) in y.iter().enumerate(){let t=origin+(i as i64+1-y.len()as i64)*60;if i!=0{s.predict(&f,&q)}s.update(&k.observation(t),p.r,v)?;}
    if !s.psd(){return None}Some(s)
}
fn final_distribution(k:Kind,p:Noise,y:&[f64],origin:i64)->Option<(f64,f64)>{
    let mut s=filter(k,p,y,origin)?;s.predict(&k.transition(),&p.q(k));
    let(m,v,_)=s.observe(&k.observation(origin+60),p.r)?;Some((m,v))
}
fn fit(rows:&[TrainingRow],k:Kind)->Result<(Noise,f64,usize,usize),String>{
    let n=rows.len().min(SAMPLE);if n==0{return Err("empty Kalman sample".into())}
    let mut best:Option<(Noise,f64)>=None;let mut trials=0;
    for r in [0.01,0.05,0.2]{for level in [0.01,0.1,1.,10.]{
        for &trend in if k.trend(){&[0.001,0.01,0.1][..]}else{&[0.][..]}{
            for &season in if k.seasonal(){&[0.001,0.01,0.1][..]}else{&[0.][..]}{
                let p=Noise{r,level,trend,season};let mut loss=0.;trials+=1;
                for i in 0..n{let row=&rows[i*rows.len()/n];
                    let Some((m,v))=final_distribution(k,p,&row.input.x[..H],row.input.origin)else{loss=f64::INFINITY;break};
                    let e=row.normalized_target[0]-m;loss+=0.5*(v.ln()+e*e/v+std::f64::consts::TAU.ln());
                }
                loss/=n as f64;if loss.is_finite()&&best.as_ref().is_none_or(|(_,x)|loss<*x){best=Some((p,loss))}
            }
        }
    }}
    let(p,loss)=best.ok_or("no finite PSD Kalman fit")?;Ok((p,loss,n,trials))
}
struct Model{k:Kind,p:Noise,mean:f64,sd:f64}
impl Predictor for Model{
    fn predict(&self,x:&OriginInput)->[f64;Y]{
        let Some(mut s)=filter(self.k,self.p,&x.x[..H],x.origin)else{return [f64::NAN;Y]};
        let f=self.k.transition();let q=self.p.q(self.k);
        std::array::from_fn(|h|{s.predict(&f,&q);
            if !s.psd(){return f64::NAN}
            s.observe(&self.k.observation(x.origin+60*(h as i64+1)),self.p.r).map_or(f64::NAN,|(m,_,_)|m*self.sd+self.mean)
        })
    }
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let k=Kind::parse(label)?;let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);
    println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();match fit(&f.train,k){
        Ok((p,nll,n,trials))=>{
            let count=2+usize::from(k.trend())+usize::from(k.seasonal());
            println!("kalman-config,{id},{label},{n},{trials},{count},{},{},{},{},{},{nll:.12e},{:.6}",k.dim(),p.r,p.level,p.trend,p.season,now.elapsed().as_secs_f64());
            score(&Model{k,p,mean:f.target_mean,sd:f.target_sd},label,&f,&mut out)?;
        },Err(e)=>println!("blocked,{id},{label},{e}"),
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    fn noise()->Noise{Noise{r:0.1,level:0.1,trend:0.01,season:0.01}}
    #[test]fn scalar_update(){let mut s=State::new(Kind::Level);let h=[1.,0.,0.,0.];s.update(&h,1.,2.).unwrap();assert!((s.x[0]-1.).abs()<1e-12);assert!((s.p[0][0]-0.5).abs()<1e-12)}
    #[test]fn covariance_psd(){for k in [Kind::Level,Kind::Trend,Kind::Season,Kind::Dlm,Kind::DlmTrend]{let y:Vec<_>=(0..72).map(|i|(i as f64/3.).sin()).collect();let mut s=filter(k,noise(),&y,0).unwrap();for _ in 0..24{s.predict(&k.transition(),&noise().q(k));assert!(s.psd())}}}
    #[test]fn seasonal_rotation(){let k=Kind::Season;let mut s=State::new(k);s.x[1]=2.;for _ in 0..24{s.predict(&k.transition(),&[0.;M])}assert!((s.x[1]-2.).abs()<1e-12);assert!(s.x[2].abs()<1e-12)}
    #[test]fn trend_transition(){let k=Kind::Trend;let mut s=State::new(k);s.x[0]=1.;s.x[1]=2.;s.predict(&k.transition(),&[0.;M]);assert_eq!(s.x[0],3.);}
    #[test]fn fixed_window_causal(){let mut x=[0.;80];x[..72].fill(2.);let a=OriginInput{origin:0,x,recent:[0.;24]};let mut b=a.clone();b.x[76]=10000.;let m=Model{k:Kind::Level,p:noise(),mean:0.,sd:1.};assert_eq!(m.predict(&a),m.predict(&b));assert!(m.predict(&a).iter().all(|v|(*v-2.).abs()<0.01))}
}
