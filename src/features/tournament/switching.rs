//! C005 S12–S13: Markov-switching autoregression and conditional GARCH variance.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow,autoreg::Qr};
use crate::benchmark::Metrics;
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const H:usize=72;const Y:usize=24;const SAMPLE:usize=512;
fn design(y:&[f64],i:usize,p:usize)->Vec<f64>{let mut z=vec![1.];z.extend((1..=p).map(|j|y[i-j]));z}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn logpdf(e:f64,v:f64)->f64{-0.5*(std::f64::consts::TAU.ln()+v.ln()+e*e/v)}
#[derive(Clone)]
struct Switch{p:usize,b:[Vec<f64>;2],v:[f64;2],t:[[f64;2];2]}
impl Switch{
    fn initial(&self)->[f64;2]{let a=self.t[0][1];let b=self.t[1][0];[b/(a+b),a/(a+b)]}
    fn filter(&self,y:&[f64])->Option<(Vec<[f64;2]>,Vec<[f64;2]>,Vec<f64>,f64)>{
        let mut f=Vec::new();let mut emission=Vec::new();let mut scales=Vec::new();let mut ll=0.;let mut prior=self.initial();
        for i in self.p..y.len(){
            let z=design(y,i,self.p);let l:[f64;2]=std::array::from_fn(|j|logpdf(y[i]-dot(&self.b[j],&z),self.v[j]));
            let shift=l[0].max(l[1]);let e=l.map(|x|(x-shift).exp());
            let a=[prior[0]*e[0],prior[1]*e[1]];let s=a[0]+a[1];if !s.is_finite()||s<=0.{return None}
            let a=a.map(|v|v/s);ll+=s.ln()+shift;f.push(a);emission.push(e);scales.push(s);
            prior=std::array::from_fn(|j|a[0]*self.t[0][j]+a[1]*self.t[1][j]);
        }Some((f,emission,scales,ll))
    }
    fn forecast(&self,y:&[f64])->[f64;Y]{
        let Some((f,_,_,_))=self.filter(y)else{return [f64::NAN;Y]};
        let mut prob=*f.last().unwrap();
        // Joint regime-weighted lag moments. Exact first moment, no sampling or plug-in regime.
        let mut moments:[Vec<f64>;2]=std::array::from_fn(|j|(0..self.p).map(|l|prob[j]*y[y.len()-1-l]).collect());
        std::array::from_fn(|_|{
            let next:[f64;2]=std::array::from_fn(|j|prob[0]*self.t[0][j]+prob[1]*self.t[1][j]);
            let nm:[Vec<f64>;2]=std::array::from_fn(|j|{
                let mix:Vec<f64>=(0..self.p).map(|l|moments[0][l]*self.t[0][j]+moments[1][l]*self.t[1][j]).collect();
                let mut z=vec![self.b[j][0]*next[j]+dot(&self.b[j][1..],&mix)];z.extend(&mix[..self.p-1]);z
            });prob=next;moments=nm;moments[0][0]+moments[1][0]
        })
    }
}
fn fit_ar(rows:&[TrainingRow],p:usize)->Result<Vec<f64>,String>{
    let mut q=Qr::new(p+1,1);for r in rows{q.add(design(&r.input.x[..H],H,p),vec![r.normalized_target[0]])}
    Ok(q.solve()?.0.into_iter().map(|v|v[0]).collect())
}
fn em(rows:&[TrainingRow],p:usize)->Result<(Switch,usize,f64,bool),String>{
    let n=rows.len().min(SAMPLE);if n==0{return Err("empty switching training".into())}
    let windows:Vec<Vec<f64>>=(0..n).map(|i|{let r=&rows[i*rows.len()/n];let mut y=r.input.x[..H].to_vec();y.push(r.normalized_target[0]);y}).collect();
    let ar=fit_ar(rows,p)?;let mut b=[ar.clone(),ar];b[0][0]-=0.05;b[1][0]+=0.05;
    let mut m=Switch{p,b,v:[0.01,0.05],t:[[0.95,0.05],[0.05,0.95]]};
    let mut old=f64::NEG_INFINITY;let mut last=f64::NEG_INFINITY;
    for it in 0..=40{
        let mut q=[Qr::new(p+1,1),Qr::new(p+1,1)];let mut trans=[[0.;2];2];let mut weights=[0.;2];let mut yy=[0.;2];let mut xy=[vec![0.;p+1],vec![0.;p+1]];let mut ll=0.;
        for y in &windows{
            let(f,e,s,l)=m.filter(y).ok_or("nonfinite switching filter")?;ll+=l;
            let mut back=[1.;2];
            for t in (0..f.len()).rev(){
                let mut g=[f[t][0]*back[0],f[t][1]*back[1]];let norm=g[0]+g[1];if !norm.is_finite()||norm<=0.{return Err("invalid smoothed training weights".into())}g=g.map(|v|v/norm);
                let z=design(y,t+p,p);let target=y[t+p];
                for j in 0..2{let w=g[j];weights[j]+=w;yy[j]+=w*target*target;for k in 0..z.len(){xy[j][k]+=w*z[k]*target}q[j].add(z.iter().map(|v|v*w.sqrt()).collect(),vec![target*w.sqrt()]);}
                if t>0{
                    let mut xi=[[0.;2];2];let mut sum=0.;for i in 0..2{for j in 0..2{xi[i][j]=f[t-1][i]*m.t[i][j]*e[t][j]*back[j];sum+=xi[i][j]}}
                    if !sum.is_finite()||sum<=0.{return Err("invalid transition weights".into())}
                    for i in 0..2{for j in 0..2{trans[i][j]+=xi[i][j]/sum}}
                    back=std::array::from_fn(|i|(m.t[i][0]*e[t][0]*back[0]+m.t[i][1]*e[t][1]*back[1])/s[t]);
                }
            }
        }
        last=ll/(n*(H+1-p))as f64;
        // The fixed stationary initial prior depends on T; the transition M-step
        // ignores that boundary term. Track the actual objective, not exact EM monotonicity.
        if it>0&&(last-old).abs()<1e-6{return Ok((m,it,last,true))}
        if it==40{break}old=last;
        for j in 0..2{
            if weights[j]<10.{return Err("collapsed switching regime".into())}
            let w:Vec<_>=q[j].solve()?.0.into_iter().map(|v|v[0]).collect();
            // Weighted least squares normal equation gives RSS=yy-b'X'y.
            let rss=yy[j]-dot(&w,&xy[j]);if !rss.is_finite()||rss< -1e-6{return Err("invalid switching RSS".into())}
            m.v[j]=(rss/weights[j]).max(1e-5);m.b[j]=w;
            let sum=trans[j][0]+trans[j][1];if sum<=0.{return Err("empty transition".into())}
            let a=(trans[j][0]/sum).clamp(0.01,0.99);m.t[j]=[a,1.-a];
        }
    }Ok((m,40,last,false))
}
struct SModel{model:Switch,mean:f64,sd:f64}
impl Predictor for SModel{fn predict(&self,x:&OriginInput)->[f64;Y]{self.model.forecast(&x.x[..H]).map(|v|v*self.sd+self.mean)}}
#[derive(Clone,Copy,Debug)]
struct Garch{omega:f64,a:f64,b:f64,uncond:f64}
impl Garch{fn next(self,v:f64,e:f64)->f64{self.omega+self.a*e*e+self.b*v}}
fn garch_fit(rows:&[TrainingRow],ar:&[f64])->Result<(Garch,f64,usize,usize),String>{
    let p=ar.len()-1;let e:Vec<_>=rows.iter().map(|r|(r.input.origin,r.normalized_target[0]-dot(ar,&design(&r.input.x[..H],H,p)))).collect();
    if e.is_empty(){return Err("empty GARCH training".into())}
    let var=e.iter().map(|(_,x)|x*x).sum::<f64>()/e.len()as f64;
    if !var.is_finite()||var<=0.{return Err("invalid residual variance".into())}
    let mut best:Option<(Garch,f64)>=None;let mut trials=0;let mut segments=0;
    for i in 0..e.len(){if i==0||e[i].0-e[i-1].0!=60{segments+=1}}
    for scale in [0.5,1.,2.]{for a in [0.02,0.05,0.1,0.2]{for b in [0.5,0.7,0.85,0.9,0.95]{
        if a+b>=0.99{continue}let g=Garch{omega:var*scale*(1.-a-b),a,b,uncond:var*scale};let mut v=g.uncond;let mut nll=0.;
        for i in 0..e.len(){if i==0||e[i].0-e[i-1].0!=60{v=g.uncond}nll-=logpdf(e[i].1,v);v=g.next(v,e[i].1)}
        nll/=e.len()as f64;trials+=1;if nll.is_finite()&&best.as_ref().is_none_or(|(_,x)|nll<*x){best=Some((g,nll))}
    }}}
    let(g,nll)=best.ok_or("no finite GARCH fit")?;Ok((g,nll,trials,segments))
}
struct GModel{ar:Vec<f64>,g:Garch,mean:f64,sd:f64}
impl GModel{
    fn distribution(&self,x:&OriginInput)->([f64;Y],[f64;Y]){
        let p=self.ar.len()-1;let mut y=x.x[..H].to_vec();let mut v=self.g.uncond;
        for i in p..H{let e=y[i]-dot(&self.ar,&design(&y,i,p));v=self.g.next(v,e)}
        let mut psi=[0.;Y];psi[0]=1.;for h in 1..Y{psi[h]=(1..=p.min(h)).map(|j|self.ar[j]*psi[h-j]).sum()}
        let mut innovations=[0.;Y];let mut mean=[0.;Y];let mut var=[0.;Y];
        for h in 0..Y{
            innovations[h]=v;v=self.g.omega+(self.g.a+self.g.b)*v;
            let next=dot(&self.ar,&design(&y,H+h,p));y.push(next);mean[h]=next*self.sd+self.mean;
            var[h]=(0..=h).map(|j|psi[h-j].powi(2)*innovations[j]).sum::<f64>()*self.sd*self.sd;
        }(mean,var)
    }
}
impl Predictor for GModel{fn predict(&self,x:&OriginInput)->[f64;Y]{self.distribution(x).0}}
fn uncertainty(model:&GModel,f:&super::PreparedFold,label:&str,output:&Path)->Result<(),String>{
    let mut out=BufWriter::new(std::fs::File::create(output.with_extension("variance.csv")).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,mean_c,variance_c2").map_err(|e|e.to_string())?;
    let mut nll=[0.;Y];let mut cover=[0.;Y];let mut sq=[0.;Y];let mut vv=[0.;Y];
    for r in &f.eval{let(m,v)=model.distribution(&r.input);
        for h in 0..Y{if !v[h].is_finite()||v[h]<=0.{return Err("invalid predictive variance".into())}let e=r.target[h]-m[h];nll[h]-=logpdf(e,v[h]);cover[h]+=if e.abs()<=1.6448536269514722*v[h].sqrt(){1.}else{0.};sq[h]+=e*e;vv[h]+=v[h];
            writeln!(out,"{},{label},{},{},{},{},{}",f.fold.id,r.input.origin,h+1,r.target[h],m[h],v[h]).map_err(|e|e.to_string())?;
        }
    }
    let n=f.eval.len()as f64;for h in 0..Y{println!("variance-lead,{},{label},{},{:.8},{:.8},{:.8}",f.fold.id,h+1,nll[h]/n,cover[h]/n,sq[h]/vv[h]);}
    println!("variance-score,{},{label},{},{:.8},{:.8},{:.8}",f.fold.id,f.eval.len(),nll.iter().sum::<f64>()/(n*Y as f64),cover.iter().sum::<f64>()/(n*Y as f64),sq.iter().sum::<f64>()/vv.iter().sum::<f64>());
    out.flush().map_err(|e|e.to_string())
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str)->Result<(),String>{
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();
    match label{
        "switch-ar1"|"switch-ar6"=>{let p=if label=="switch-ar1"{1}else{6};match em(&f.train,p){
            Ok((model,it,ll,conv))=>{println!("switch-config,{id},{label},{},{},{it},{ll:.12e},{conv},{:.6}",SAMPLE,2*(p+2)+2,now.elapsed().as_secs_f64());println!("switch-coef,{id},{label},{:?},{:?},{:?}",model.b,model.v,model.t);score(&SModel{model,mean:f.target_mean,sd:f.target_sd},label,&f,&mut out)?;},
            Err(e)=>println!("blocked,{id},{label},{e}"),
        }},
        "garch-ar6"|"garch-ar24"=>{let p=if label=="garch-ar6"{6}else{24};match fit_ar(&f.train,p).and_then(|ar|garch_fit(&f.train,&ar).map(|(g,ll,trials,segments)|(ar,g,ll,trials,segments))){
            Ok((ar,g,ll,trials,segments))=>{println!("garch-config,{id},{label},{n},{segments},{trials},{},{},{},{},{ll:.12e},{:.6}",p+4,g.omega,g.a,g.b,now.elapsed().as_secs_f64());println!("garch-ar-coef,{id},{label},{ar:?}");let model=GModel{ar,g,mean:f.target_mean,sd:f.target_sd};score(&model,label,&f,&mut out)?;uncertainty(&model,&f,label,output)?;},
            Err(e)=>println!("blocked,{id},{label},{e}"),
        }},
        _=>return Err("unknown switching/GARCH config".into()),
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn switch_identical_regimes(){let m=Switch{p:1,b:[vec![0.,0.5],vec![0.,0.5]],v:[1.,1.],t:[[0.9,0.1],[0.2,0.8]]};let mut y=[0.;72];y[71]=2.;let f=m.forecast(&y);assert!((f[0]-1.).abs()<1e-12);assert!((f[1]-0.5).abs()<1e-12);}
    #[test]fn regime_moments(){let m=Switch{p:1,b:[vec![0.,0.],vec![2.,0.]],v:[1.,1.],t:[[0.5,0.5],[0.5,0.5]]};assert!(m.forecast(&[0.;72]).iter().all(|v|(*v-1.).abs()<1e-12));}
    #[test]fn garch_stationary(){let g=Garch{omega:0.1,a:0.1,b:0.8,uncond:1.};assert!((g.next(1.,1.)-1.).abs()<1e-12)}
    #[test]fn ar_variance_propagation(){let g=Garch{omega:1.,a:0.,b:0.,uncond:1.};let m=GModel{ar:vec![0.,0.5],g,mean:0.,sd:1.};let x=OriginInput{origin:0,x:[0.;80],recent:[0.;24]};let(_,v)=m.distribution(&x);assert!((v[0]-1.).abs()<1e-12);assert!((v[1]-1.25).abs()<1e-12)}
}
