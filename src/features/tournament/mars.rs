//! C004 L12: bounded-grid multivariate adaptive regression splines.
use super::{load,score,Fold,OriginInput,Predictor,TrainingRow,D};
use std::{io::{BufWriter,Write},path::Path,time::Instant};
const Y:usize=24;
const SAMPLE:usize=1024;
const SHORTLIST:usize=8;
#[derive(Clone,Copy,Debug,PartialEq)]
struct Hinge{feature:usize,knot:f64,sign:f64}
impl Hinge{fn eval(self,x:&[f64;D])->f64{(self.sign*(x[self.feature]-self.knot)).max(0.)}}
type Term=Vec<Hinge>;
fn value(t:&Term,x:&[f64;D])->f64{t.iter().map(|h|h.eval(x)).product()}
fn dot(x:&[f64],y:&[f64])->f64{x.iter().zip(y).map(|(a,b)|a*b).sum()}
fn orth(mut v:Vec<f64>,q:&[Vec<f64>])->Option<Vec<f64>>{
    let raw=dot(&v,&v);
    for _ in 0..2{for u in q{let a=dot(&v,u);for(i,x)in v.iter_mut().enumerate(){*x-=a*u[i]}}}
    let norm=dot(&v,&v).sqrt();
    if !norm.is_finite()||norm<=1e-9*raw.sqrt().max(1.){None}else{for x in &mut v{*x/=norm}Some(v)}
}
fn cross(v:&[f64],r:&[[f64;Y]])->[f64;Y]{let mut c=[0.;Y];for (a,row)in v.iter().zip(r){for k in 0..Y{c[k]+=a*row[k]}}c}
fn remove(v:&[f64],r:&mut[[f64;Y]],c:&[f64;Y]){for(a,row)in v.iter().zip(r){for k in 0..Y{row[k]-=a*c[k]}}}
fn gain(c:&[f64;Y])->f64{c.iter().map(|x|x*x).sum()}
fn forward(rows:&[TrainingRow],max_terms:usize,degree:usize)->Result<Vec<Term>,String>{
    forward_refs(&rows.iter().collect::<Vec<_>>(),max_terms,degree)
}
fn forward_refs(rows:&[&TrainingRow],max_terms:usize,degree:usize)->Result<Vec<Term>,String>{
    if rows.len()<4{return Err("too few training rows".into())}
    let n=rows.len().min(SAMPLE);
    let s:Vec<_>=(0..n).map(|i|&rows[i*rows.len()/n]).collect();
    let mut knots=vec![Vec::new();D];
    for j in 0..D{
        let mut x:Vec<_>=s.iter().map(|r|r.input.x[j]).collect();x.sort_by(f64::total_cmp);
        for k in [1,2,3]{let v=x[(n-1)*k/4];if v>x[0]&&v<x[n-1]&&!knots[j].contains(&v){knots[j].push(v)}}
    }
    let mut terms:Vec<Term>=vec![vec![]];
    let mut q=vec![vec![1./(n as f64).sqrt();n]];
    let mut residual:Vec<_>=s.iter().map(|r|r.normalized_target).collect();
    let c=cross(&q[0],&residual);remove(&q[0],&mut residual,&c);
    let clock=Instant::now();
    while terms.len()+2<=max_terms+1{
        if clock.elapsed().as_secs_f64()>900.{return Err("forward search exceeded 900s guard".into())}
        // Root plus at most four most recent degree-one parents, fixed before fit.
        let mut parents=vec![0];if degree==2{parents.extend((1..terms.len()).rev().filter(|&i|terms[i].len()==1).take(4))}
        let mut short:Vec<(f64,Term,Term,Vec<f64>,Vec<f64>)>=Vec::new();
        for parent in parents{
            let pv:Vec<_>=s.iter().map(|r|value(&terms[parent],&r.input.x)).collect();
            for (j,ks)in knots.iter().enumerate(){
                if terms[parent].iter().any(|h|h.feature==j){continue}
                for &knot in ks{
                    let mut a=terms[parent].clone();a.push(Hinge{feature:j,knot,sign:1.});
                    let mut b=terms[parent].clone();b.push(Hinge{feature:j,knot,sign:-1.});
                    if terms.contains(&a)||terms.contains(&b){continue}
                    let va:Vec<_>=s.iter().zip(&pv).map(|(r,p)|p*(r.input.x[j]-knot).max(0.)).collect();
                    let vb:Vec<_>=s.iter().zip(&pv).map(|(r,p)|p*(knot-r.input.x[j]).max(0.)).collect();
                    let na=dot(&va,&va);let nb=dot(&vb,&vb);
                    if na<=1e-12||nb<=1e-12{continue}
                    let rough=gain(&cross(&va,&residual))/na+gain(&cross(&vb,&residual))/nb;
                    if short.len()<SHORTLIST||rough>short.last().unwrap().0{
                        short.push((rough,a,b,va,vb));short.sort_by(|a,b|b.0.total_cmp(&a.0));short.truncate(SHORTLIST);
                    }
                }
            }
        }
        let mut best:Option<(f64,Term,Term,Vec<f64>,Vec<f64>)>=None;
        for(_,a,b,va,vb)in short{
            let Some(ua)=orth(va,&q)else{continue};
            let mut basis=q.clone();basis.push(ua.clone());
            let Some(ub)=orth(vb,&basis)else{continue};
            let g=gain(&cross(&ua,&residual))+gain(&cross(&ub,&residual));
            if g>best.as_ref().map_or(1e-10,|v|v.0){best=Some((g,a,b,ua,ub))}
        }
        let Some((g,a,b,ua,ub))=best else{break};
        let ca=cross(&ua,&residual);remove(&ua,&mut residual,&ca);
        let cb=cross(&ub,&residual);remove(&ub,&mut residual,&cb);
        terms.push(a);terms.push(b);q.push(ua);q.push(ub);
        println!("mars-forward,{},{},{g:.10}",degree,terms.len()-1);
    }
    Ok(terms)
}
struct Stats{g:Vec<Vec<f64>>,xy:Vec<[f64;Y]>,yy:f64,n:usize}
fn stats(rows:&[TrainingRow],terms:&[Term])->Stats{
    stats_refs(&rows.iter().collect::<Vec<_>>(),terms)
}
fn stats_refs(rows:&[&TrainingRow],terms:&[Term])->Stats{
    let p=terms.len();let mut g=vec![vec![0.;p];p];let mut xy=vec![[0.;Y];p];let mut yy=0.;
    for r in rows{
        let z:Vec<_>=terms.iter().map(|t|value(t,&r.input.x)).collect();
        for j in 0..p{for i in 0..=j{g[j][i]+=z[j]*z[i]}for k in 0..Y{xy[j][k]+=z[j]*r.normalized_target[k]}}
        yy+=gain(&r.normalized_target);
    }
    let n=rows.len();for j in 0..p{for i in 0..=j{g[j][i]/=n as f64;g[i][j]=g[j][i]}for k in 0..Y{xy[j][k]/=n as f64}}
    Stats{g,xy,yy:yy/n as f64,n}
}
fn fit_subset(s:&Stats,active:&[usize])->Result<(Vec<[f64;Y]>,f64,Vec<f64>),String>{
    let p=active.len();let mut l=vec![vec![0.;p];p];let max=active.iter().map(|&j|s.g[j][j]).fold(0.,f64::max);
    for j in 0..p{for i in 0..=j{
        let v=s.g[active[j]][active[i]]-dot(&l[j][..i],&l[i][..i]);
        if i==j{if !v.is_finite()||v<=max*1e-12{return Err("rank-deficient MARS basis".into())}l[j][i]=v.sqrt()}else{l[j][i]=v/l[i][i]}
    }}
    let mut b:Vec<_>=active.iter().map(|&j|s.xy[j]).collect();
    for j in 0..p{for i in 0..j{let a=l[j][i];for k in 0..Y{b[j][k]-=a*b[i][k]}}for v in &mut b[j]{*v/=l[j][j]}}
    for j in (0..p).rev(){for i in j+1..p{let a=l[i][j];for k in 0..Y{b[j][k]-=a*b[i][k]}}for v in &mut b[j]{*v/=l[j][j]}}
    let rss=(s.yy-active.iter().zip(&b).map(|(&j,v)|dot(&s.xy[j],v)).sum::<f64>()).max(0.);
    let mut inv_diag=vec![0.;p];
    for j in 0..p{
        let mut x=vec![0.;p];for i in j..p{x[i]=((if i==j{1.}else{0.})-dot(&l[i][j..i],&x[j..i]))/l[i][i]}
        inv_diag[j]=dot(&x,&x);
    }
    if !rss.is_finite()||b.iter().flatten().any(|v|!v.is_finite()){return Err("nonfinite MARS fit".into())}
    Ok((b,rss,inv_diag))
}
fn prune(s:&Stats,mut active:Vec<usize>)->Result<(Vec<usize>,Vec<[f64;Y]>,f64),String>{
    let mut best:Option<(Vec<usize>,Vec<[f64;Y]>,f64)>=None;
    loop{
        let (b,rss,inv)=fit_subset(s,&active)?;
        // GCV: intercept + one effective parameter per slope + 3 per knot pair.
        let effective=1.+2.5*(active.len()-1)as f64;
        let gcv=if effective>=s.n as f64{f64::INFINITY}else{rss/(Y as f64*(1.-effective/s.n as f64).powi(2))};
        println!("mars-prune,{},{rss:.10},{gcv:.10}",active.len()-1);
        if gcv<best.as_ref().map_or(f64::INFINITY,|v|v.2){best=Some((active.clone(),b.clone(),gcv))}
        if active.len()==1{break}
        // Exact least-squares RSS increase for deleting one column: sum(beta_j²)/inverse_Gram_jj.
        let j=(1..active.len()).min_by(|&a,&c|(gain(&b[a])/inv[a]).total_cmp(&(gain(&b[c])/inv[c]))).unwrap();
        active.remove(j);
    }
    best.ok_or("no finite pruning GCV".into())
}
struct Model{terms:Vec<Term>,b:Vec<[f64;Y]>,mean:f64,sd:f64}
impl Predictor for Model{fn predict(&self,x:&OriginInput)->[f64;Y]{
    let mut y=[0.;Y];for(t,b)in self.terms.iter().zip(&self.b){let a=value(t,&x.x);for k in 0..Y{y[k]+=a*b[k]}}
    y.map(|v|v*self.sd+self.mean)
}}
pub fn run(data:&Path,id:&str,output:&Path,selection:&str)->Result<(),String>{
    let suffix=selection.strip_prefix("mars-grid-").ok_or("unknown MARS configuration")?;
    let(a,b)=suffix.split_once('-').ok_or("expected degree and max terms")?;
    let degree:usize=a.parse().map_err(|_|"invalid degree")?;let cap:usize=b.parse().map_err(|_|"invalid cap")?;
    if ![1,2].contains(&degree)||![16,32,64].contains(&cap){return Err("unknown MARS config".into())}
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=crate::benchmark::Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);
    println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);
    writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();
    let result=forward(&f.train,cap,degree).and_then(|terms|{
        let s=stats(&f.train,&terms);let (active,b,gcv)=prune(&s,(0..terms.len()).collect())?;
        let chosen=active.iter().map(|&j|terms[j].clone()).collect::<Vec<_>>();
        Ok((terms.len()-1,chosen,b,gcv))
    });
    match result{
        Ok((forward_count,terms,b,gcv))=>{
            println!("mars-config,{id},{selection},{degree},{cap},{forward_count},{},{},{gcv:.12e},{:.6}",terms.len()-1,terms.len()*Y,now.elapsed().as_secs_f64());
            for (i,t) in terms.iter().enumerate(){println!("mars-term,{id},{selection},{i},{t:?}")}
            let model=Model{terms,b,mean:f.target_mean,sd:f.target_sd};score(&model,selection,&f,&mut out)?;
        },
        Err(e)=>println!("blocked,{id},{selection},{e}"),
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
pub(super) fn optimize(f:&super::PreparedFold,rows:&[&TrainingRow],label:&str,degree:usize,cap:usize,out:&mut impl Write)->Result<(),String>{
    if ![1,2].contains(&degree)||![16,32,64].contains(&cap){return Err("invalid C012 MARS configuration".into())}
    let id=f.fold.id;let now=Instant::now();
    let p=rows.len().min(1024);let small:Vec<_>=(0..p).map(|i|rows[i*rows.len()/p]).collect();
    let workspace=8*(p*(cap+1+Y+32)+(cap+1)*(cap+1+Y)*4+rows.len()*4+f.eval.len()*25);
    if workspace>2_000_000_000{println!("blocked,{id},{label},preflight workspace guard");return Ok(())}
    let timer=Instant::now();let terms=match forward_refs(&small,cap,degree){Ok(v)=>v,Err(e)=>{println!("blocked,{id},{label},preflight {e}");return Ok(())}};let forward_s=timer.elapsed().as_secs_f64();
    let timer=Instant::now();let stats=stats_refs(&small,&terms);let(active,b,_)=match prune(&stats,(0..terms.len()).collect()){Ok(v)=>v,Err(e)=>{println!("blocked,{id},{label},preflight {e}");return Ok(())}};
    let probe=Model{terms:active.iter().map(|&j|terms[j].clone()).collect(),b,mean:f.target_mean,sd:f.target_sd};let fit_s=timer.elapsed().as_secs_f64();
    let q=rows.len().min(32);let timer=Instant::now();for i in 0..q{std::hint::black_box(probe.predict(&rows[i*rows.len()/q].input));}let pp=timer.elapsed().as_secs_f64();
    let estimate=forward_s+fit_s*rows.len()as f64/p as f64+pp*f.eval.len()as f64/q as f64;
    println!("optimization-mars-preflight,{id},{label},{p},{q},{workspace},{forward_s:.6},{fit_s:.6},{pp:.6},{estimate:.6}");
    if 2.*estimate+180.>1400.{println!("blocked,{id},{label},preflight runtime guard");return Ok(())}
    let now=Instant::now();let result=forward_refs(rows,cap,degree).and_then(|terms|{
        let s=stats_refs(rows,&terms);let(active,b,gcv)=prune(&s,(0..terms.len()).collect())?;
        let chosen=active.iter().map(|&j|terms[j].clone()).collect::<Vec<_>>();Ok((terms.len()-1,chosen,b,gcv))
    });
    match result{
        Ok((forward_count,terms,b,gcv))=>{
            let fit=now.elapsed().as_secs_f64();let params=terms.len()*Y;
            println!("mars-config,{id},{label},{degree},{cap},{forward_count},{},{params},{gcv:.12e},{fit:.6}",terms.len()-1);
            for(i,t)in terms.iter().enumerate(){println!("mars-term,{id},{label},{i},{t:?}")}
            let model=Model{terms,b,mean:f.target_mean,sd:f.target_sd};let timer=Instant::now();score(&model,label,f,out)?;
            println!("optimization-cost,{id},{label},{fit:.6},{:.6},{params}",timer.elapsed().as_secs_f64());
        },Err(e)=>println!("blocked,{id},{label},{e}")
    }Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn adaptive_hinge_recovers_v_shape(){
        let rows:Vec<_>=(0..101).map(|i|{let mut x=[0.;D];x[5]=(i as f64-50.)/10.;let y=[2.+x[5].abs();Y];
            TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:y,normalized_target:y}}).collect();
        let t=forward(&rows,16,1).unwrap();assert!(t.len()>=3);
        let s=stats(&rows,&t);let(a,b,_)=prune(&s,(0..t.len()).collect()).unwrap();
        let m=Model{terms:a.iter().map(|&j|t[j].clone()).collect(),b,mean:0.,sd:1.};
        for r in &rows{assert!((m.predict(&r.input)[0]-r.target[0]).abs()<1e-7)}
    }
    #[test]fn deletion_formula(){
        let rows:Vec<_>=(0..101).map(|i|{let mut x=[0.;D];x[0]=(i as f64-50.)/10.;let y=[1.+x[0].abs();Y];
            TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:y,normalized_target:y}}).collect();
        let t=vec![vec![],vec![Hinge{feature:0,knot:0.,sign:1.}],vec![Hinge{feature:0,knot:0.,sign:-1.}]];
        let s=stats(&rows,&t);let(b,r,inv)=fit_subset(&s,&[0,1,2]).unwrap();
        let(_,r2,_)=fit_subset(&s,&[0,2]).unwrap();assert!((r2-r-gain(&b[1])/inv[1]).abs()<1e-8);
    }
    #[test]fn optimization_mars_reference_parity(){
        let rows:Vec<_>=(0..101).map(|i|{let mut x=[0.;D];x[5]=(i as f64-50.)/10.;let y=[2.+x[5].abs();Y];TrainingRow{input:OriginInput{origin:0,x,recent:[0.;Y]},target:y,normalized_target:y}}).collect();
        let refs:Vec<_>=rows.iter().collect();let a=forward(&rows,16,1).unwrap();let b=forward_refs(&refs,16,1).unwrap();assert_eq!(a,b);
        let sa=stats(&rows,&a);let sb=stats_refs(&refs,&b);assert_eq!(sa.g,sb.g);assert_eq!(sa.xy,sb.xy);assert_eq!(sa.yy,sb.yy);
    }
}
