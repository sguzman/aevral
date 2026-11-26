//! C005 S08–S09: additive innovations ETS and harmonic regression with AR errors.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow,autoreg::Qr};
use crate::{benchmark::Metrics,windows::Calendar};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const H:usize=72;const Y:usize=24;const SAMPLE:usize=2048;
#[derive(Clone,Copy,Debug,PartialEq)]
enum Ets{Ann,Aan,Aadn,Aaa}
impl Ets{fn trend(self)->bool{self!=Self::Ann}fn season(self)->bool{self==Self::Aaa}}
#[derive(Clone,Copy,Debug)]
struct Smooth{a:f64,b:f64,g:f64,phi:f64}
struct EState{level:f64,trend:f64,season:[f64;24],i:usize}
impl EState{
    fn init(y:&[f64],kind:Ets)->Self{
        let level=y[..24].iter().sum::<f64>()/24.;
        let mut season=[0.;24];if kind.season(){for j in 0..24{season[j]=y[j]-level}}
        let trend=if kind.trend()&&!kind.season(){(y[12..24].iter().sum::<f64>()-y[..12].iter().sum::<f64>())/144.}else{0.};
        Self{level,trend,season,i:24}
    }
    fn forecast(&self,p:Smooth,kind:Ets)->f64{self.level+p.phi*self.trend+if kind.season(){self.season[self.i%24]}else{0.}}
    fn advance(&mut self,e:f64,p:Smooth,kind:Ets){
        self.level+=p.phi*self.trend+p.a*e;
        if kind.trend(){self.trend=p.phi*self.trend+p.b*e}
        if kind.season(){self.season[self.i%24]+=p.g*e}
        self.i+=1;
    }
}
fn filter(y:&[f64],p:Smooth,kind:Ets)->EState{
    let mut s=EState::init(y,kind);for &v in &y[24..]{let e=v-s.forecast(p,kind);s.advance(e,p,kind)}s
}
fn ets_fit(rows:&[TrainingRow],kind:Ets)->Result<(Smooth,f64,usize,usize),String>{
    let n=rows.len().min(SAMPLE);if n==0{return Err("empty ETS training".into())}
    let mut best=None;let mut trials=0;
    let grid=[0.05,0.2,0.5,0.8];
    for a in grid{for b in if kind.trend(){&[0.0,0.02,0.1,0.2][..]}else{&[0.0][..]}{
        for g in if kind.season(){&grid[..]}else{&[0.0][..]}{
            for phi in if kind==Ets::Aadn{&[0.8,0.9,0.95,0.98][..]}else{&[1.0][..]}{
                if *b>a||*g>1.-a{continue}
                let p=Smooth{a,b:*b,g:*g,phi:*phi};let mut mse=0.;
                for i in 0..n{let r=&rows[i*rows.len()/n];let s=filter(&r.input.x[..H],p,kind);let e=r.normalized_target[0]-s.forecast(p,kind);mse+=e*e}
                mse/=n as f64;trials+=1;
                if mse.is_finite()&&best.as_ref().is_none_or(|(_,v)|mse<*v){best=Some((p,mse))}
            }
        }
    }}
    let (p,mse)=best.ok_or("no finite ETS fit")?;Ok((p,mse,n,trials))
}
struct EModel{kind:Ets,p:Smooth,mean:f64,sd:f64}
impl Predictor for EModel{
    fn predict(&self,x:&OriginInput)->[f64;Y]{
        let mut s=filter(&x.x[..H],self.p,self.kind);
        std::array::from_fn(|_|{let v=s.forecast(self.p,self.kind);s.advance(0.,self.p,self.kind);v*self.sd+self.mean})
    }
}
fn harmonic(t:i64,k:usize)->Vec<f64>{
    let f=Calendar::Both.features(t);let mut z=vec![1.];
    for j in 0..2{let a=f[2*j];let b=f[2*j+1];z.extend([a,b]);if k==2{z.extend([2.*a*b,b*b-a*a])}}z
}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn residual_design(res:&[f64],p:usize)->Vec<f64>{let mut z=vec![1.];z.extend(res.iter().rev().take(p));z}
struct HModel{k:usize,p:usize,b:Vec<f64>,ar:Vec<f64>,mean:f64,sd:f64}
impl Predictor for HModel{
    fn predict(&self,x:&OriginInput)->[f64;Y]{
        let mut e:Vec<_>=(H-self.p..H).map(|i|x.x[i]-dot(&self.b,&harmonic(x.origin+(i as i64+1-H as i64)*60,self.k))).collect();
        std::array::from_fn(|h|{let v=dot(&self.ar,&residual_design(&e,self.p));e.push(v);(v+dot(&self.b,&harmonic(x.origin+(h as i64+1)*60,self.k)))*self.sd+self.mean})
    }
}
fn harmonic_fit(rows:&[TrainingRow],k:usize,p:usize,mean:f64,sd:f64)->Result<(HModel,f64,f64),String>{
    let mut q=Qr::new(1+4*k,1);
    for r in rows{q.add(harmonic(r.input.origin,k),vec![r.input.x[71]])}
    let(b,r1)=q.solve()?;let b:Vec<_>=b.into_iter().map(|v|v[0]).collect();
    let mut q=Qr::new(p+1,1);
    for r in rows{
        let e:Vec<_>=(H-p..H).map(|i|r.input.x[i]-dot(&b,&harmonic(r.input.origin+(i as i64+1-H as i64)*60,k))).collect();
        let y=r.normalized_target[0]-dot(&b,&harmonic(r.input.origin+60,k));q.add(residual_design(&e,p),vec![y]);
    }
    let(ar,r2)=q.solve()?;let ar:Vec<_>=ar.into_iter().map(|v|v[0]).collect();
    Ok((HModel{k,p,b,ar,mean,sd},r1,r2))
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);
    println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();
    let kind=match label{"ets-ann"=>Some(Ets::Ann),"ets-aan"=>Some(Ets::Aan),"ets-aadn"=>Some(Ets::Aadn),"ets-aaa"=>Some(Ets::Aaa),_=>None};
    if let Some(kind)=kind{
        match ets_fit(&f.train,kind){
            Ok((p,mse,n,trials))=>{let count=match kind{Ets::Ann=>1,Ets::Aan=>2,Ets::Aadn|Ets::Aaa=>3};
                println!("ets-config,{id},{label},{n},{trials},{count},{},{},{},{},{mse:.12e},{:.6}",p.a,p.b,p.g,p.phi,now.elapsed().as_secs_f64());
                score(&EModel{kind,p,mean:f.target_mean,sd:f.target_sd},label,&f,&mut out)?;
            },Err(e)=>println!("blocked,{id},{label},{e}"),
        }
    }else{
        let(k,p)=match label{"harmonic-1-ar1"=>(1,1),"harmonic-1-ar6"=>(1,6),"harmonic-2-ar1"=>(2,1),"harmonic-2-ar6"=>(2,6),_=>return Err("unknown smoothing configuration".into())};
        match harmonic_fit(&f.train,k,p,f.target_mean,f.target_sd){
            Ok((model,r1,r2))=>{
                println!("harmonic-config,{id},{label},{n},{},{r1:.12e},{r2:.12e},{:.6}",model.b.len()+model.ar.len(),now.elapsed().as_secs_f64());
                println!("harmonic-coef,{id},{label},{:?},{:?}",model.b,model.ar);
                score(&model,label,&f,&mut out)?;
            },Err(e)=>println!("blocked,{id},{label},{e}"),
        }
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn constant_level(){let y=[2.;72];let p=Smooth{a:0.2,b:0.,g:0.,phi:1.};let s=filter(&y,p,Ets::Ann);assert_eq!(s.forecast(p,Ets::Ann),2.);}
    #[test]fn seasonal_repetition(){let y:Vec<_>=(0..72).map(|i|(i%24)as f64).collect();let p=Smooth{a:0.2,b:0.02,g:0.2,phi:1.};let mut s=filter(&y,p,Ets::Aaa);for i in 0..24{assert!((s.forecast(p,Ets::Aaa)-i as f64).abs()<1e-10);s.advance(0.,p,Ets::Aaa)}}
    #[test]fn damped_trend(){let p=Smooth{a:0.2,b:0.1,g:0.,phi:0.8};let mut s=EState{level:0.,trend:1.,season:[0.;24],i:24};assert_eq!(s.forecast(p,Ets::Aadn),0.8);s.advance(0.,p,Ets::Aadn);assert!((s.forecast(p,Ets::Aadn)-1.44).abs()<1e-12);}
    #[test]fn harmonic_period(){let a=harmonic(0,2);let b=harmonic(1440,2);for i in 0..5{assert!((a[i]-b[i]).abs()<1e-12)}assert_eq!(a.len(),9);}
    #[test]fn ar_residual_recursive(){let mut x=[0.;80];x[71]=2.;let input=OriginInput{origin:0,x,recent:[0.;24]};let m=HModel{k:1,p:1,b:vec![0.;5],ar:vec![0.,0.5],mean:0.,sd:1.};let y=m.predict(&input);assert_eq!(y[0],1.);assert_eq!(y[1],0.5);}
}
