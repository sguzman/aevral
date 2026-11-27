//! C004 L07–L09: PCR, PLS2 (regression deflation), reduced-rank regression.
use super::{load,score,Fold,OriginInput,Predictor,D,sparse::Gram};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;
const CONFIGS:[(&str,&str,usize);12]=[
("pcr-8","pcr",8),("pcr-16","pcr",16),("pcr-32","pcr",32),("pcr-64","pcr",64),
("pls-4","pls",4),("pls-8","pls",8),("pls-16","pls",16),("pls-32","pls",32),
("rrr-2","rrr",2),("rrr-4","rrr",4),("rrr-8","rrr",8),("rrr-16","rrr",16)];
// Symmetric cyclic Jacobi, columns of V accumulated as eigenvectors, sorted descending.
pub(super) fn eig<const N:usize>(a:&[[f64;N]])->Result<Vec<(f64,[f64;N])>,String>{
    if a.len()!=N||a.iter().flatten().any(|v|!v.is_finite()){return Err("bad eigensystem input".into())}
    let mut m=a.to_vec();let mut v=vec![[0.;N];N];for i in 0..N{v[i][i]=1.}
    let scale=a.iter().flatten().map(|x|x.abs()).fold(0.,f64::max);
    let tol=1e-12*scale.max(1e-30);
    let mut converged=false;
    for _ in 0..100{
        let mut off=0.0f64;
        for p in 0..N{for q in p+1..N{
            let b=m[p][q];off=off.max(b.abs());if b.abs()<=tol{continue}
            let tau=(m[q][q]-m[p][p])/(2.*b);
            let t=if tau>=0.{1./(tau+tau.hypot(1.))}else{-1./(-tau+tau.hypot(1.))};
            let c=1./(1.+t*t).sqrt();let s=t*c;
            m[p][p]-=t*b;m[q][q]+=t*b;m[p][q]=0.;m[q][p]=0.;
            for i in 0..N{if i!=p&&i!=q{
                let (x,y)=(m[i][p],m[i][q]);m[i][p]=c*x-s*y;m[p][i]=m[i][p];m[i][q]=s*x+c*y;m[q][i]=m[i][q];
            }let(x,y)=(v[i][p],v[i][q]);v[i][p]=c*x-s*y;v[i][q]=s*x+c*y;}
        }}
        if off<=tol{converged=true;break}
    }
    if !converged{return Err("Jacobi eigensolver sweep cap".into())}
    let mut out:Vec<_>=(0..N).map(|j|(m[j][j],std::array::from_fn(|i|v[i][j]))).collect();
    out.sort_by(|a,b|b.0.total_cmp(&a.0));
    for (j,(ev,w)) in out.iter().enumerate(){
        let err=(0..N).map(|i|(dot(&a[i],w)-ev*w[i]).abs()).fold(0.,f64::max);
        if err>1e-8*scale.max(1e-30){return Err("eigen residual exceeds tolerance".into())}
        for (k,(_,u)) in out.iter().enumerate(){if (dot(w,u)-if j==k{1.}else{0.}).abs()>1e-8{return Err("eigenvector orthogonality failure".into())}}
    }Ok(out)
}
fn dot<const N:usize>(a:&[f64;N],b:&[f64;N])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
type Beta=Vec<[f64;Y]>;
fn pcr(q:&Gram,e:&[(f64,[f64;D])],rank:usize)->Result<Beta,String>{
    let mut b=vec![[0.;Y];D];
    for (value,w) in e.iter().take(rank){
        if *value<=e[0].0*1e-10{return Err("PCR retained eigenvalue below rank tolerance".into())}
        for k in 0..Y{
            let a=(0..D).map(|i|w[i]*q.xy[i][k]).sum::<f64>()/value;
            for i in 0..D{b[i][k]+=w[i]*a}
        }
    }Ok(b)
}
fn rrr(q:&Gram,ols:&Beta,rank:usize)->Result<Beta,String>{
    // Eigenvectors of covariance of fitted multivariate OLS responses, not raw Y.
    let mut c=vec![[0.;Y];Y];for i in 0..Y{for j in 0..=i{
        let a=(0..D).map(|k|q.xy[k][i]*ols[k][j]).sum::<f64>();
        let b=(0..D).map(|k|q.xy[k][j]*ols[k][i]).sum::<f64>();
        c[i][j]=(a+b)/2.;c[j][i]=c[i][j];
    }}
    let e=eig(&c)?;if e[rank-1].0<=e[0].0*1e-10{return Err("RRR retained response rank deficient".into())}
    let mut b=vec![[0.;Y];D];for (_,v) in e.iter().take(rank){for i in 0..D{
        let a=dot(&ols[i],v);for k in 0..Y{b[i][k]+=a*v[k]}
    }}Ok(b)
}
fn pls(q:&Gram,rank:usize)->Result<Beta,String>{
    // PLS2/NIPALS regression deflation represented in covariance space.
    let mut g=q.g.clone();let mut s=q.xy.clone();let mut a=vec![[0.;D];D];
    for i in 0..D{a[i][i]=1.}
    let mut beta=vec![[0.;Y];D];let original_scale=g.iter().enumerate().map(|(i,r)|r[i]).fold(0.,f64::max);
    let cross_scale=s.iter().flatten().map(|v|v.abs()).fold(0.,f64::max);
    for _ in 0..rank{
        let mut c=vec![[0.;Y];Y];for i in 0..Y{for j in 0..=i{
            c[i][j]=(0..D).map(|k|s[k][i]*s[k][j]).sum();c[j][i]=c[i][j];
        }}
        let e=eig(&c)?;let v=&e[0].1;
        let mut w:[f64;D]=std::array::from_fn(|i|dot(&s[i],v));let norm=dot(&w,&w).sqrt();
        if norm<=1e-10*cross_scale{return Err("PLS residual cross covariance exhausted".into())}
        for x in &mut w{*x/=norm}
        let gw:[f64;D]=std::array::from_fn(|i|dot(&g[i],&w));let var=dot(&w,&gw);
        if var<=1e-10*original_scale||!var.is_finite(){return Err("PLS score variance exhausted".into())}
        let p=gw.map(|x|x/var);
        let qy:[f64;Y]=std::array::from_fn(|k|(0..D).map(|i|w[i]*s[i][k]).sum::<f64>()/var);
        let aw:[f64;D]=std::array::from_fn(|i|dot(&a[i],&w));
        for i in 0..D{for k in 0..Y{beta[i][k]+=aw[i]*qy[k];s[i][k]-=gw[i]*qy[k];}
            for j in 0..D{a[i][j]-=aw[i]*p[j];g[i][j]-=gw[i]*p[j];}
        }
    }Ok(beta)
}
struct Model{b:Beta,mx:[f64;D],my:[f64;Y],mean:f64,sd:f64}
impl Predictor for Model{fn predict(&self,x:&OriginInput)->[f64;Y]{
    let mut y=self.my;for i in 0..D{let z=x.x[i]-self.mx[i];for k in 0..Y{y[k]+=z*self.b[i][k]}}
    y.map(|v|v*self.sd+self.mean)
}}
pub fn run(data:&Path,id:&str,output:&Path,selection:&str)->Result<(),String>{
    if selection!="all"&&!CONFIGS.iter().any(|c|c.0==selection){return Err("unknown latent config".into())}
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=crate::benchmark::Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);
    println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let start=Instant::now();let q=Gram::fit(&f.train)?;println!("gram-seconds,{id},{:.6}",start.elapsed().as_secs_f64());
    let start=Instant::now();let e=eig(&q.g);println!("eigen-seconds,{id},{:.6}",start.elapsed().as_secs_f64());
    for (label,method,r) in CONFIGS{
        if selection!="all"&&selection!=label{continue}
        let now=Instant::now();let b=match method{
            "pcr"=>e.as_ref().map_err(|x|x.clone()).and_then(|e|pcr(&q,e,r)),
            "pls"=>pls(&q,r),
            _=>e.as_ref().map_err(|x|x.clone()).and_then(|e|pcr(&q,e,D)).and_then(|b|rrr(&q,&b,r))
        };
        match b{
            Ok(b) if b.iter().flatten().all(|v|v.is_finite())=>{
                println!("latent-config,{id},{label},{r},{},{:.6}",(D+1)*Y,now.elapsed().as_secs_f64());
                let model=Model{b,mx:q.mx,my:q.my,mean:f.target_mean,sd:f.target_sd};score(&model,label,&f,&mut out)?;
            },
            Ok(_)=>println!("blocked,{id},{label},nonfinite coefficients"),
            Err(err)=>println!("blocked,{id},{label},{err}"),
        }
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    use super::super::TrainingRow;
    fn fixture()->Gram{
        let mut rows=Vec::new();
        for i in 0..D{for sign in [-1.,1.]{
            let mut x=[0.;D];x[i]=sign*(D as f64).sqrt()*if i==0{2.}else{1.};
            let mut y=[3.;Y];if i==1{y[0]+=x[i]*4.}if i==2{y[1]+=x[i]*2.}
            rows.push(TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:y,normalized_target:y});
        }}Gram::fit(&rows).unwrap()
    }
    #[test]fn eig_rotated(){let e=eig(&[[2.,1.],[1.,2.]]).unwrap();assert!((e[0].0-3.).abs()<1e-10);assert!((e[1].0-1.).abs()<1e-10);}
    #[test]fn pcr_vs_pls(){
        let q=fixture();let e=eig(&q.g).unwrap();let b=pcr(&q,&e,1).unwrap();assert!(b.iter().flatten().all(|v|v.abs()<1e-10));
        let b=pls(&q,1).unwrap();assert!((b[1][0]-4.).abs()<1e-10);assert!(b[2][1].abs()<1e-10);
        let b=pls(&q,2).unwrap();assert!((b[2][1]-2.).abs()<1e-10);assert!(pls(&q,3).is_err());
    }
    #[test]fn rrr_fitted_response_rank(){
        let q=fixture();let e=eig(&q.g).unwrap();let full=pcr(&q,&e,D).unwrap();
        assert!((full[1][0]-4.).abs()<1e-10);assert!((full[2][1]-2.).abs()<1e-10);
        let b=rrr(&q,&full,1).unwrap();assert!((b[1][0]-4.).abs()<1e-10);assert!(b[2][1].abs()<1e-10);
        let b=rrr(&q,&full,2).unwrap();assert!((b[2][1]-2.).abs()<1e-10);
        let m=Model{b,mx:q.mx,my:q.my,mean:10.,sd:2.};assert_eq!(m.predict(&OriginInput{origin:0,x:[0.;D],recent:[0.;Y]}),[16.;Y]);
    }
}
