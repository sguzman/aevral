//! AEV-005–008: factorial selection of origin-time weather variables.
use crate::{Mlp,benchmark::Metrics,covariates::Weather,weather::{Split,split_for},windows::Calendar};
const H:usize=72;
const Y:usize=24;
#[derive(Debug)]
struct Case { origin:i64, split:Split, x:[f64;H], y:[f64;Y], weather:[f64;4] }
fn cases(bins:&[Option<Weather>],first:i64)->Vec<Case>{
    collect_cases(bins,first,false)
}
fn collect_cases(bins:&[Option<Weather>],first:i64,include_test:bool)->Vec<Case>{
    let mut rows=Vec::new();
    for i in H-1..bins.len().saturating_sub(Y){
        let origin=first+i as i64*60;
        let Some(split)=split_for(origin)else{continue};
        if split==Split::Test&&!include_test{continue}
        let Some(r)=bins[i]else{continue};
        let (Some(dew),Some(pressure),Some(wind))=(r.dew_c,r.pressure_hpa,r.wind_ms)else{continue};
        let v:Option<Vec<_>>=bins[i+1-H..=i+Y].iter().map(|r|r.and_then(|v|v.temperature)).collect();
        if let Some(v)=v{rows.push(Case{origin,split,x:v[..H].try_into().unwrap(),y:v[H..].try_into().unwrap(),weather:[dew,pressure,wind[0],wind[1]]})}
    }rows
}
#[derive(Clone,Copy,Debug)]
struct Scale{mean:f64,sd:f64}
impl Scale{
    fn fit(v:impl Iterator<Item=f64>+Clone)->Result<Self,String>{
        let n=v.clone().count()as f64;
        let mean=v.clone().sum::<f64>()/n;
        let sd=(v.map(|x|(x-mean).powi(2)).sum::<f64>()/n).sqrt();
        if !mean.is_finite()||!sd.is_finite()||sd<=0.{return Err("invalid training scale".into())}
        Ok(Self{mean,sd})
    }
    fn norm(self,v:f64)->f64{(v-self.mean)/self.sd}
    fn restore(self,v:f64)->f64{v*self.sd+self.mean}
}
fn columns(mask:u8)->Vec<usize>{
    (0..4).filter(|i|mask&(1<<(*i).min(2))!=0).collect()
}
fn name(mask:u8)->&'static str{
    ["reference","dew","pressure","dew-pressure","wind","dew-wind","pressure-wind","all"][mask as usize]
}
fn input(r:&Case,mask:u8,temp:Scale,weather:[Scale;4])->Vec<f64>{
    let mut x:Vec<_>=r.x.iter().map(|&v|temp.norm(v)).collect();
    x.extend(Calendar::Both.features(r.origin));
    x.extend(columns(mask).into_iter().map(|i|weather[i].norm(r.weather[i])));x
}
fn report(name:&str,m:&Metrics,daily:f64)->Result<(),String>{
    if !m.mae().is_finite()||!m.rmse().is_finite(){return Err("nonfinite scores".into())}
    println!("aggregate,{name},{},{:.8},{:.8},{:.8}",m.count,m.mae(),m.rmse(),1.-m.mae()/daily);
    for i in 0..Y{let(a,b)=m.lead(i);println!("lead,{name},{},{:.8},{:.8}",i+1,a,b)}
    Ok(())
}
fn train_model(train:&[&Case],tr:&[Vec<f64>],mask:u8,temp:Scale)->Mlp{
    let dim=H+4+columns(mask).len();
        let mut model=Mlp::new(dim,32,Y,42);
        let mut order:Vec<_>=(0..train.len()).collect();let mut rng=42u64;
        for epoch in 0..10{
            for i in (1..order.len()).rev(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;order.swap(i,(rng%(i as u64+1))as usize);}
            let mut loss=0.;
            for &i in &order{loss+=model.train_step(&tr[i],&train[i].y.map(|v|temp.norm(v)),0.005)}
            println!("features={} epoch={} normalized_mse={:.8}",name(mask),epoch+1,loss/train.len()as f64);
        }

    model
}
pub fn run(bins:&[Option<Weather>],first:i64)->Result<(),String>{
    let rows=cases(bins,first);
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let val:Vec<_>=rows.iter().filter(|r|r.split==Split::Validation).collect();
    if train.len()!=121849||val.len()!=30207{return Err(format!("frozen cohort mismatch: {} {}",train.len(),val.len()))}
    let temp=Scale::fit(train.iter().flat_map(|r|r.x))?;
    let weather: [Scale;4]=(0..4).map(|i|Scale::fit(train.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    println!("AEV-005-008 matched train={} validation={}",train.len(),val.len());
    println!("temperature_scale={temp:?} weather_scales={weather:?}");
    let mut daily=Metrics::default();
    for r in &val{daily.add(&r.y,&r.x[H-Y..])}
    report("daily",&daily,daily.mae())?;
    for mask in 0..8{
        let dim=H+4+columns(mask).len();
        let tr:Vec<_>=train.iter().map(|r|input(r,mask,temp,weather)).collect();
        let va:Vec<_>=val.iter().map(|r|input(r,mask,temp,weather)).collect();
        let model=train_model(&train,&tr,mask,temp);
        let mut m=Metrics::default();
        for(i,r)in val.iter().enumerate(){let p:Vec<_>=model.predict(&va[i]).into_iter().map(|v|temp.restore(v)).collect();m.add(&r.y,&p)}
        println!("features={} mask={mask} inputs={dim} parameters={}",name(mask),32*dim+824);
        report(name(mask),&m,daily.mae())?;
    }Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn masks(){assert_eq!(columns(0),vec![]);assert_eq!(columns(1),vec![0]);assert_eq!(columns(2),vec![1]);assert_eq!(columns(4),vec![2,3]);assert_eq!(columns(7),vec![0,1,2,3]);}
    #[test]fn causal_features(){
        let mut b:Vec<_>=(0..98).map(|i|Some(Weather{minute:i*60,temperature:Some(i as f64),dew_c:Some(i as f64),pressure_hpa:Some(1000.),wind_ms:Some([1.,2.])})).collect();
        // Only the origin's covariates are needed, not the future covariates.
        b[95].as_mut().unwrap().dew_c=None;
        let r=cases(&b,0);assert_eq!(r.len(),3);assert_eq!(r[0].weather,[71.,1000.,1.,2.]);assert_eq!(r[0].y[23],95.);
        b[71].as_mut().unwrap().dew_c=None;assert_eq!(cases(&b,0).len(),2);
    }
}

/** Frozen mask-7 evaluation, with unchanged training partition and recipe. */
pub fn locked_test(bins:&[Option<Weather>],first:i64)->Result<(),String>{
    use crate::{benchmark::{Climatology,persistence,daily as daily_prediction},weather::Example};
    let rows=collect_cases(bins,first,true);
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let test:Vec<_>=rows.iter().filter(|r|r.split==Split::Test).collect();
    if train.len()!=121849||test.is_empty(){return Err(format!("unexpected train/test counts {} {}",train.len(),test.len()))}
    let temp=Scale::fit(train.iter().flat_map(|r|r.x))?;
    let weather:[Scale;4]=(0..4).map(|i|Scale::fit(train.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    let expected=[11.612720385128858,6.359004285883184,6.403598625256534,5.0022216207107,1017.3671062499778,7.110340951197967,0.27492170423930146,2.1428563920747172,0.9016637834425706,3.193150037462841];
    let actual=[temp.mean,temp.sd,weather[0].mean,weather[0].sd,weather[1].mean,weather[1].sd,weather[2].mean,weather[2].sd,weather[3].mean,weather[3].sd];
    if actual.iter().zip(expected).any(|(a,b)|(a-b).abs()>1e-8){return Err("training scale differs from frozen validation".into())}
    let tr:Vec<_>=train.iter().map(|r|input(r,7,temp,weather)).collect();
    let model=train_model(&train,&tr,7,temp);
    let ex=|r:&Case|Example{origin:r.origin,split:r.split,x:r.x[H-Y..].try_into().unwrap(),y:r.y};
    let clim_rows:Vec<_>=train.iter().map(|r|ex(r)).collect();
    let clim=Climatology::fit(&clim_rows.iter().collect::<Vec<_>>())?;
    println!("LOCKED_TEST train={} test={} mask=7",train.len(),test.len());
    println!("temperature_scale={temp:?} weather_scales={weather:?}");
    let mut m:[Metrics;4]=std::array::from_fn(|_|Metrics::default());
    let mut by_year:[[Metrics;4];3]=std::array::from_fn(|_|std::array::from_fn(|_|Metrics::default()));
    let edges=[crate::weather::minute("2024-01-01 00:00")?,crate::weather::minute("2025-01-01 00:00")?];
    let mut possible=0usize;let mut temperature_complete=0usize;
    for i in H-1..bins.len().saturating_sub(Y){
        if split_for(first+i as i64*60)!=Some(Split::Test){continue}
        possible+=1;
        if bins[i+1-H..=i+Y].iter().all(|r|r.is_some_and(|v|v.temperature.is_some())){temperature_complete+=1}
    }
    println!("coverage possible={possible} temperature_complete={temperature_complete} joint_complete={}",test.len());
    for (idx,r) in test.iter().enumerate(){
        let e=ex(r);
        let pred:Vec<_>=model.predict(&input(r,7,temp,weather)).into_iter().map(|v|temp.restore(v)).collect();
        let p=[persistence(&e),daily_prediction(&e),clim.predict(&e),pred.try_into().map_err(|_|"bad prediction shape")?];
        let yr=if r.origin+60<edges[0]{0}else if r.origin+60<edges[1]{1}else{2};
        for i in 0..4{m[i].add(&r.y,&p[i]);by_year[yr][i].add(&r.y,&p[i])}
        // Predeclared illustrative case: first eligible origin, not cherry-picked.
        if idx==0{for h in 0..Y{println!("example,{},{},{:.8},{:.8},{:.8}",r.origin,h+1,r.y[h],p[3][h],p[1][h])}}
    }
    let names=["persistence","daily","climatology","frozen-mlp"];
    for i in 0..4{report(names[i],&m[i],m[1].mae())?}
    for (yr,metrics) in by_year.iter().enumerate(){for(i,v)in metrics.iter().enumerate(){if v.count>0{println!("year,{},{},{},{:.8},{:.8}",2023+yr,names[i],v.count,v.mae(),v.rmse())}}}
    Ok(())
}
#[cfg(test)]
mod locked_tests{
    use super::*;
    #[test]fn test_requires_explicit_opt_in(){
        let first=crate::weather::minute("2023-01-01 00:00").unwrap();
        let b:Vec<_>=(0..98).map(|i|Some(Weather{minute:first+i*60,temperature:Some(i as f64),dew_c:Some(5.),pressure_hpa:Some(1000.),wind_ms:Some([1.,2.])})).collect();
        assert!(cases(&b,first).is_empty());
        let r=collect_cases(&b,first,true);assert_eq!(r.len(),3);assert!(r.iter().all(|r|r.split==Split::Test));
    }
}

pub fn architectures(bins:&[Option<Weather>],first:i64)->Result<(),String>{
    use crate::dense::Dense;
    let rows=cases(bins,first);
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let val:Vec<_>=rows.iter().filter(|r|r.split==Split::Validation).collect();
    if train.len()!=121849||val.len()!=30207{return Err("frozen cohort mismatch".into())}
    let temp=Scale::fit(train.iter().flat_map(|r|r.x))?;
    let weather:[Scale;4]=(0..4).map(|i|Scale::fit(train.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    let tr:Vec<_>=train.iter().map(|r|input(r,7,temp,weather)).collect();
    let va:Vec<_>=val.iter().map(|r|input(r,7,temp,weather)).collect();
    let targets:Vec<_>=train.iter().map(|r|r.y.map(|v|temp.norm(v))).collect();
    let mut daily=Metrics::default();
    for r in &val{daily.add(&r.y,&r.x[H-Y..])}
    println!("AEV-010 matched train={} validation={}",train.len(),val.len());
    report("daily",&daily,daily.mae())?;
    let configs:[(&str,&[usize]);5]=[("mlp32",&[80,32,24]),("linear",&[80,24]),("mlp16",&[80,16,24]),("mlp64",&[80,64,24]),("mlp32x32",&[80,32,32,24])];
    for (label,dims) in configs{
        let mut model=Dense::new(dims,42);
        let mut order:Vec<_>=(0..train.len()).collect();let mut rng=42u64;
        let start=std::time::Instant::now();
        for epoch in 0..10{
            for i in (1..order.len()).rev(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;order.swap(i,(rng%(i as u64+1))as usize);}
            let mut loss=0.;
            for &i in &order{loss+=model.train_step(&tr[i],&targets[i],0.005)}
            if !loss.is_finite(){return Err(format!("nonfinite training loss: {label}"))}
            println!("architecture={label} epoch={} normalized_mse={:.8}",epoch+1,loss/train.len()as f64);
        }
        println!("architecture={label} parameters={} training_seconds={:.3}",model.parameters(),start.elapsed().as_secs_f64());
        let mut m=Metrics::default();
        for(i,r)in val.iter().enumerate(){let p:Vec<_>=model.predict(&va[i]).into_iter().map(|v|temp.restore(v)).collect();m.add(&r.y,&p)}
        report(label,&m,daily.mae())?;
        if label=="mlp32"&&(m.mae()-1.39610378).abs()>1e-6{return Err("C001 reference reproduction failed".into())}
    }Ok(())
}

pub fn recurrent_screen(bins:&[Option<Weather>],first:i64,label:&str)->Result<(),String>{
    use crate::recurrent::{Kind,Recurrent};
    let kind=Kind::parse(label).ok_or("unknown recurrent kind")?;
    let rows=cases(bins,first);
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let val:Vec<_>=rows.iter().filter(|r|r.split==Split::Validation).collect();
    if train.len()!=121849||val.len()!=30207{return Err("frozen cohort mismatch".into())}
    let temp=Scale::fit(train.iter().flat_map(|r|r.x))?;
    let weather:[Scale;4]=(0..4).map(|i|Scale::fit(train.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    let tr:Vec<_>=train.iter().map(|r|input(r,7,temp,weather)).collect();
    let va:Vec<_>=val.iter().map(|r|input(r,7,temp,weather)).collect();
    let targets:Vec<_>=train.iter().map(|r|r.y.map(|v|temp.norm(v))).collect();
    let mut daily=Metrics::default();for r in &val{daily.add(&r.y,&r.x[H-Y..])}
    println!("AEV-011 model={label} train={} validation={}",train.len(),val.len());
    report("daily",&daily,daily.mae())?;
    let mut model=Recurrent::new(kind,42);
    let mut order:Vec<_>=(0..train.len()).collect();let mut rng=42u64;
    let start=std::time::Instant::now();
    for epoch in 0..10{
        for i in (1..order.len()).rev(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;order.swap(i,(rng%(i as u64+1))as usize);}
        let(mut loss,mut clips)=(0.,0usize);
        for &i in &order{let(l,c)=model.train_step(&tr[i],&targets[i],0.005);loss+=l;clips+=usize::from(c)}
        if !loss.is_finite(){return Err(format!("nonfinite training loss: {label}"))}
        println!("architecture={label} epoch={} normalized_mse={:.8} clips={clips}",epoch+1,loss/train.len()as f64);
    }
    println!("architecture={label} parameters={} training_seconds={:.3}",model.parameters(),start.elapsed().as_secs_f64());
    let mut m=Metrics::default();
    for(i,r)in val.iter().enumerate(){let p=model.predict(&va[i]).map(|v|temp.restore(v));m.add(&r.y,&p)}
    report(label,&m,daily.mae())
}

pub fn convolution_screen(bins:&[Option<Weather>],first:i64,label:&str)->Result<(),String>{
    use crate::convolution::{Kind,Temporal};
    let kind=Kind::parse(label).ok_or("unknown convolution kind")?;
    let rows=cases(bins,first);
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let val:Vec<_>=rows.iter().filter(|r|r.split==Split::Validation).collect();
    if train.len()!=121849||val.len()!=30207{return Err("frozen cohort mismatch".into())}
    let temp=Scale::fit(train.iter().flat_map(|r|r.x))?;
    let weather:[Scale;4]=(0..4).map(|i|Scale::fit(train.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    let tr:Vec<_>=train.iter().map(|r|input(r,7,temp,weather)).collect();
    let va:Vec<_>=val.iter().map(|r|input(r,7,temp,weather)).collect();
    let targets:Vec<_>=train.iter().map(|r|r.y.map(|v|temp.norm(v))).collect();
    let mut daily=Metrics::default();for r in &val{daily.add(&r.y,&r.x[H-Y..])}
    println!("AEV-012 model={label} train={} validation={}",train.len(),val.len());
    report("daily",&daily,daily.mae())?;
    let mut model=Temporal::new(kind,42);
    let mut order:Vec<_>=(0..train.len()).collect();let mut rng=42u64;
    let start=std::time::Instant::now();
    for epoch in 0..10{
        for i in (1..order.len()).rev(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;order.swap(i,(rng%(i as u64+1))as usize);}
        let(mut loss,mut clips)=(0.,0usize);
        for &i in &order{let(l,c)=model.train_step(&tr[i],&targets[i],0.005);loss+=l;clips+=usize::from(c)}
        if !loss.is_finite(){return Err(format!("nonfinite training loss: {label}"))}
        println!("architecture={label} epoch={} normalized_mse={:.8} clips={clips}",epoch+1,loss/train.len()as f64);
    }
    println!("architecture={label} parameters={} training_seconds={:.3}",model.parameters(),start.elapsed().as_secs_f64());
    let mut m=Metrics::default();
    for(i,r)in val.iter().enumerate(){let p=model.predict(&va[i]).map(|v|temp.restore(v));m.add(&r.y,&p)}
    report(label,&m,daily.mae())
}

pub fn attention_screen(bins:&[Option<Weather>],first:i64,label:&str)->Result<(),String>{
    use crate::attention::Attention;
    let rows=cases(bins,first);
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let val:Vec<_>=rows.iter().filter(|r|r.split==Split::Validation).collect();
    if train.len()!=121849||val.len()!=30207{return Err("frozen cohort mismatch".into())}
    let temp=Scale::fit(train.iter().flat_map(|r|r.x))?;
    let weather:[Scale;4]=(0..4).map(|i|Scale::fit(train.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    let tr:Vec<_>=train.iter().map(|r|input(r,7,temp,weather)).collect();
    let va:Vec<_>=val.iter().map(|r|input(r,7,temp,weather)).collect();
    let targets:Vec<_>=train.iter().map(|r|r.y.map(|v|temp.norm(v))).collect();
    let mut daily=Metrics::default();for r in &val{daily.add(&r.y,&r.x[H-Y..])}
    println!("AEV-013 model={label} train={} validation={}",train.len(),val.len());
    report("daily",&daily,daily.mae())?;
    let mut model=Attention::new(42);
    let mut order:Vec<_>=(0..train.len()).collect();let mut rng=42u64;
    let start=std::time::Instant::now();
    for epoch in 0..10{
        for i in (1..order.len()).rev(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;order.swap(i,(rng%(i as u64+1))as usize);}
        let(mut loss,mut clips)=(0.,0usize);
        for &i in &order{let(l,c)=model.train_step(&tr[i],&targets[i],0.005);loss+=l;clips+=usize::from(c)}
        if !loss.is_finite(){return Err(format!("nonfinite training loss: {label}"))}
        println!("architecture={label} epoch={} normalized_mse={:.8} clips={clips}",epoch+1,loss/train.len()as f64);
    }
    println!("architecture={label} parameters={} training_seconds={:.3}",model.parameters(),start.elapsed().as_secs_f64());
    let mut m=Metrics::default();
    for(i,r)in val.iter().enumerate(){let p=model.predict(&va[i]).map(|v|temp.restore(v));m.add(&r.y,&p)}
    report(label,&m,daily.mae())
}

pub fn architecture_budget(bins:&[Option<Weather>],first:i64,label:&str,seed:u64,adam:bool)->Result<(),String>{
    use crate::architecture::Model;
    if ![42,123,2026].contains(&seed){return Err("seed outside predeclared schedule".into())}
    let expected=match label{"mlp32"=>1.39610378,"mlp64"=>1.38652827,"mlp32x32"=>1.39090582,"rnn"=>1.52718775,"cnn"=>1.54584788,"attention"=>1.55932750,_=>return Err("unknown architecture".into())};
    let rows=cases(bins,first);
    let train:Vec<_>=rows.iter().filter(|r|r.split==Split::Train).collect();
    let val:Vec<_>=rows.iter().filter(|r|r.split==Split::Validation).collect();
    if train.len()!=121849||val.len()!=30207{return Err("frozen cohort mismatch".into())}
    let temp=Scale::fit(train.iter().flat_map(|r|r.x))?;
    let weather:[Scale;4]=(0..4).map(|i|Scale::fit(train.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    let tr:Vec<_>=train.iter().map(|r|input(r,7,temp,weather)).collect();
    let va:Vec<_>=val.iter().map(|r|input(r,7,temp,weather)).collect();
    let targets:Vec<_>=train.iter().map(|r|r.y.map(|v|temp.norm(v))).collect();
    let mut daily=Metrics::default();for r in &val{daily.add(&r.y,&r.x[H-Y..])}
    println!("AEV-014{} model={label} seed={seed} train={} validation={}",if adam{"B"}else{"A"},train.len(),val.len());
    report("daily",&daily,daily.mae())?;
    let mut model=Model::new(label,seed)?;
    let mut opt=crate::adam::Adam::new(model.parameters(),0.001);
    let mut order:Vec<_>=(0..train.len()).collect();let mut rng=seed;
    let mut seconds=0.;let mut total_clips=0usize;
    for epoch in 1..=30{
        let start=std::time::Instant::now();
        for i in (1..order.len()).rev(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;order.swap(i,(rng%(i as u64+1))as usize);}
        let(mut loss,mut clips)=(0.,0usize);
        for &i in &order{let(l,c)=if adam{model.adam_step(&tr[i],&targets[i],&mut opt)}else{model.train_step(&tr[i],&targets[i],0.005)};loss+=l;clips+=usize::from(c)}
        if !loss.is_finite(){return Err(format!("nonfinite training loss: {label} seed {seed}"))}
        seconds+=start.elapsed().as_secs_f64();total_clips+=clips;
        println!("budget,{label},{seed},{epoch},{:.8},{clips}",loss/train.len()as f64);
        if epoch==10||epoch==30{
            let mut m=Metrics::default();
            for(i,r)in val.iter().enumerate(){let p=model.predict(&va[i]).map(|v|temp.restore(v));m.add(&r.y,&p)}
            if !adam&&seed==42&&epoch==10&&(m.mae()-expected).abs()>1e-6{return Err(format!("screen reproduction failed: {label} {}",m.mae()))}
            println!("checkpoint,{label},{seed},{epoch},{},{:.8},{:.8},{:.8},{},{seconds:.3},{total_clips}",m.count,m.mae(),m.rmse(),1.-m.mae()/daily.mae(),model.parameters());
            for h in 0..Y{let(a,b)=m.lead(h);println!("checkpoint-lead,{label},{seed},{epoch},{},{a:.8},{b:.8}",h+1)}
        }
    }Ok(())
}

fn holdout_origin(origin:i64)->bool{
    let start=crate::weather::minute("2026-01-01 00:00").unwrap();
    let end=crate::weather::minute("2026-11-01 00:00").unwrap();
    origin+60>=start&&origin+60*(Y as i64)<end
}
fn holdout_cases(bins:&[Option<Weather>],first:i64)->(Vec<Case>,[usize;3]){
    let(mut rows,mut counts)=(Vec::new(),[0;3]);
    for i in H-1..bins.len().saturating_sub(Y){
        let origin=first+i as i64*60;
        if !holdout_origin(origin){continue}
        counts[0]+=1;
        let v:Option<Vec<_>>=bins[i+1-H..=i+Y].iter().map(|r|r.and_then(|v|v.temperature)).collect();
        let Some(v)=v else{continue};
        counts[1]+=1;
        let Some(r)=bins[i]else{continue};
        let(Some(dew),Some(pressure),Some(wind))=(r.dew_c,r.pressure_hpa,r.wind_ms)else{continue};
        counts[2]+=1;
        rows.push(Case{origin,split:Split::Test,x:v[..H].try_into().unwrap(),y:v[H..].try_into().unwrap(),weather:[dew,pressure,wind[0],wind[1]]});
    }(rows,counts)
}
pub fn holdout_audit(bins:&[Option<Weather>],first:i64,temps:&[Option<crate::weather::Report>])->Result<(),String>{
    if bins.len()!=temps.len(){return Err("archive length mismatch".into())}
    for (c,t) in bins.iter().zip(temps){
        match(c,t){(None,None)=>{},(Some(c),Some(t)) if c.minute==t.minute&&c.temperature==t.celsius=>{},_=>return Err("archive alignment mismatch".into())}
    }
    let historical=cases(bins,first);
    let train:Vec<_>=historical.iter().filter(|r|r.split==Split::Train).collect();
    let val=historical.iter().filter(|r|r.split==Split::Validation).count();
    if train.len()!=121849||val!=30207{return Err("historical cohort mismatch".into())}
    let temp=Scale::fit(train.iter().flat_map(|r|r.x))?;
    let weather:[Scale;4]=(0..4).map(|i|Scale::fit(train.iter().map(move|r|r.weather[i]))).collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
    let expected=[11.612720385128858,6.359004285883184,6.403598625256534,5.0022216207107,1017.3671062499778,7.110340951197967,0.27492170423930146,2.1428563920747172,0.9016637834425706,3.193150037462841];
    let actual=[temp.mean,temp.sd,weather[0].mean,weather[0].sd,weather[1].mean,weather[1].sd,weather[2].mean,weather[2].sd,weather[3].mean,weather[3].sd];
    if actual.iter().zip(expected).any(|(a,b)|(a-b).abs()>1e-8){return Err("training scale differs from frozen validation".into())}
    let (test,counts)=holdout_cases(bins,first);
    if counts[0]!=7273||test.is_empty(){return Err(format!("unexpected holdout coverage {counts:?}"))}
    println!("AEV-015 audit train={} validation={} possible={} temperature_complete={} joint_complete={}",train.len(),val,counts[0],counts[1],counts[2]);
    println!("AEV-015 training_scales={actual:?}");
    for m in 1..=10{
        let lo=crate::weather::minute(&format!("2026-{m:02}-01 00:00"))?;
        let hi=crate::weather::minute(&format!("2026-{:02}-01 00:00",m+1))?;
        let n=test.iter().filter(|r|r.origin+60>=lo&&r.origin+60<hi).count();
        println!("coverage-month,{m},{n}");
    }Ok(())
}
#[cfg(test)]
mod holdout_tests{
    use super::*;
    #[test]fn boundaries(){
        let a=crate::weather::minute("2026-01-01 00:00").unwrap();
        let b=crate::weather::minute("2026-11-01 00:00").unwrap();
        assert!(!holdout_origin(a-120));assert!(holdout_origin(a-60));
        assert!(holdout_origin(b-25*60));assert!(!holdout_origin(b-24*60));
        assert_eq!((a-60..b-24*60).step_by(60).count(),7273);
    }
}
