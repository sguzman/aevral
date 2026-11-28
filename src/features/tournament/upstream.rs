//! Official upstream boosting adapters, one scalar model per forecast lead.
use super::{load,score,Fold,OriginInput,Predictor,D};
use super::cart::Data;
use crate::benchmark::Metrics;
use std::{collections::BTreeMap,ffi::{c_char,c_int,c_void,CStr,CString},io::{BufWriter,Write},path::{Path,PathBuf},process::Command,time::Instant};
const Y:usize=24;const GUARD:f64=1200.;
type Handle=*mut c_void;
type V=[f64;Y];
#[link(name="dl")]
extern "C"{fn dlopen(name:*const c_char,flags:c_int)->Handle;fn dlsym(h:Handle,name:*const c_char)->Handle;fn dlclose(h:Handle)->c_int;}
struct Library(Handle);
impl Library{
    fn open(path:&str)->Result<Self,String>{let p=CString::new(path).map_err(|e|e.to_string())?;let h=unsafe{dlopen(p.as_ptr(),2)};if h.is_null(){Err("cannot load pinned XGBoost library".into())}else{Ok(Self(h))}}
    fn sym<T:Copy>(&self,name:&str)->Result<T,String>{let n=CString::new(name).unwrap();let p=unsafe{dlsym(self.0,n.as_ptr())};if p.is_null(){return Err(format!("missing XGBoost symbol {name}"))}assert_eq!(std::mem::size_of::<T>(),std::mem::size_of::<Handle>());Ok(unsafe{std::mem::transmute_copy(&p)})}
}
impl Drop for Library{fn drop(&mut self){unsafe{dlclose(self.0);}}}
struct Owned{h:Handle,free:unsafe extern "C" fn(Handle)->c_int}
impl Drop for Owned{fn drop(&mut self){unsafe{(self.free)(self.h);}}}
struct Xgb{lib:Library}
impl Xgb{
    fn new()->Result<Self,String>{let lib=Library::open(&std::env::var("AEVRAL_XGBOOST_LIB").map_err(|_|"use locked boosting shell")?)?;let ver:unsafe extern "C" fn(*mut c_int,*mut c_int,*mut c_int)=lib.sym("XGBoostVersion")?;let(mut a,mut b,mut c)=(0,0,0);unsafe{ver(&mut a,&mut b,&mut c)}if(a,b,c)!=(3,0,5){return Err(format!("wrong XGBoost version {a}.{b}.{c}"))}Ok(Self{lib})}
    fn check(&self,code:c_int)->Result<(),String>{if code==0{return Ok(())}let f:unsafe extern "C" fn()->*const c_char=self.lib.sym("XGBGetLastError")?;let p=unsafe{f()};Err(if p.is_null(){"unknown XGBoost error".into()}else{unsafe{CStr::from_ptr(p)}.to_string_lossy().into_owned()})}
    fn matrix(&self,x:&[[f64;D]])->Result<Owned,String>{
        let create:unsafe extern "C" fn(*const f32,u64,u64,f32,*mut Handle,c_int)->c_int=self.lib.sym("XGDMatrixCreateFromMat_omp")?;
        let free=self.lib.sym("XGDMatrixFree")?;let vals:Vec<f32>=x.iter().flat_map(|r|r.iter().map(|v|*v as f32)).collect();let mut h=std::ptr::null_mut();
        self.check(unsafe{create(vals.as_ptr(),x.len()as u64,D as u64,f32::NAN,&mut h,2)})?;Ok(Owned{h,free})
    }
    fn fit(&self,train:&Owned,eval:&Owned,y:&[f64],n_eval:usize,rounds:usize,seed:u64,now:Instant)->Result<(Vec<f64>,usize,usize),String>{
        let info:unsafe extern "C" fn(Handle,*const c_char,*const f32,u64)->c_int=self.lib.sym("XGDMatrixSetFloatInfo")?;
        let yy:Vec<f32>=y.iter().map(|v|*v as f32).collect();self.check(unsafe{info(train.h,b"label\0".as_ptr().cast(),yy.as_ptr(),yy.len()as u64)})?;
        let create:unsafe extern "C" fn(*const Handle,u64,*mut Handle)->c_int=self.lib.sym("XGBoosterCreate")?;
        let free=self.lib.sym("XGBoosterFree")?;let mut h=std::ptr::null_mut();
        self.check(unsafe{create(&train.h,1,&mut h)})?;let model=Owned{h,free};
        let set:unsafe extern "C" fn(Handle,*const c_char,*const c_char)->c_int=self.lib.sym("XGBoosterSetParam")?;
        let seed=seed.to_string();
        for(k,v)in [("objective","reg:squarederror"),("booster","gbtree"),("tree_method","hist"),("device","cpu"),("max_depth","4"),("eta","0.1"),("max_bin","64"),("subsample","1"),("colsample_bytree","1"),("nthread","2"),("seed",&seed),("verbosity","0")]{
            let k=CString::new(k).unwrap();let v=CString::new(v).unwrap();self.check(unsafe{set(model.h,k.as_ptr(),v.as_ptr())})?;
        }
        let update:unsafe extern "C" fn(Handle,c_int,Handle)->c_int=self.lib.sym("XGBoosterUpdateOneIter")?;
        for i in 0..rounds{guard(now)?;self.check(unsafe{update(model.h,i as c_int,train.h)})?}
        let predict:unsafe extern "C" fn(Handle,Handle,*const c_char,*mut *const u64,*mut u64,*mut *const f32)->c_int=self.lib.sym("XGBoosterPredictFromDMatrix")?;
        let json=CString::new(r#"{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":true}"#).unwrap();
        let(mut shape,mut dim,mut out)=(std::ptr::null(),0,std::ptr::null());
        self.check(unsafe{predict(model.h,eval.h,json.as_ptr(),&mut shape,&mut dim,&mut out)})?;
        if dim!=2||shape.is_null()||out.is_null(){return Err("unexpected XGBoost prediction shape".into())}
        let sh=unsafe{std::slice::from_raw_parts(shape,2)};if sh!=[n_eval as u64,1]{return Err(format!("unexpected XGBoost shape {sh:?}"))}
        let values=unsafe{std::slice::from_raw_parts(out,n_eval)}.iter().map(|v|*v as f64).collect();
        let dump:unsafe extern "C" fn(Handle,*const c_char,c_int,*const c_char,*mut u64,*mut *const *const c_char)->c_int=self.lib.sym("XGBoosterDumpModelEx")?;
        let(mut len,mut arr)=(0,std::ptr::null());self.check(unsafe{dump(model.h,b"\0".as_ptr().cast(),0,b"text\0".as_ptr().cast(),&mut len,&mut arr)})?;
        if len==0||arr.is_null(){return Err("empty XGBoost model dump".into())}
        let leaves=unsafe{std::slice::from_raw_parts(arr,len as usize)}.iter().map(|&p|unsafe{CStr::from_ptr(p)}.to_string_lossy().matches("leaf=").count()).sum();
        Ok((values,len as usize,leaves))
    }
}
fn guard(now:Instant)->Result<(),String>{if now.elapsed().as_secs_f64()>=GUARD{Err("fit elapsed guard".into())}else{Ok(())}}
fn command(bin:&str,args:&[String],dir:&Path,now:Instant)->Result<(),String>{
    guard(now)?;let seconds=(GUARD-now.elapsed().as_secs_f64()).floor().max(1.)as u64;
    let o=Command::new("timeout").arg(seconds.to_string()).arg(bin).args(args).current_dir(dir).output().map_err(|e|e.to_string())?;
    if o.status.code()==Some(124){return Err("fit elapsed guard".into())}
    if !o.status.success(){let s=String::from_utf8_lossy(&o.stderr);let t=String::from_utf8_lossy(&o.stdout);return Err(format!("backend command failed {}: {} {}",o.status,s.chars().take(3000).collect::<String>(),t.chars().take(3000).collect::<String>()))}Ok(())
}
struct Scratch(PathBuf);
impl Scratch{fn new()->Result<Self,String>{static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);let id=NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed);let p=std::env::temp_dir().join(format!("aevral-upstream-{}-{id}",std::process::id()));std::fs::create_dir(&p).map_err(|e|e.to_string())?;Ok(Self(p))}}
impl Drop for Scratch{fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
fn write_rows(path:&Path,x:&[[f64;D]],y:Option<&[f64]>)->Result<(),String>{
    let mut w=BufWriter::new(std::fs::File::create(path).map_err(|e|e.to_string())?);
    for(i,r)in x.iter().enumerate(){if let Some(y)=y{write!(w,"{}\t",y[i]).map_err(|e|e.to_string())?}for(j,v)in r.iter().enumerate(){if j>0{write!(w,"\t").map_err(|e|e.to_string())?}write!(w,"{v}").map_err(|e|e.to_string())?}writeln!(w).map_err(|e|e.to_string())?}w.flush().map_err(|e|e.to_string())
}
fn parse_predictions(s:&str,n:usize,header:bool)->Result<Vec<f64>,String>{
    let mut lines=s.lines();
    let mut sample_column=None;let mut pred_column=0;let mut width=1;
    if header{
        let first=lines.next().ok_or("missing CatBoost prediction header")?;
        let fields:Vec<_>=first.trim_end_matches('\r').split('\t').collect();
        width=fields.len();
        let pred:Vec<_>=fields.iter().enumerate().filter(|(_,v)|**v=="RawFormulaVal").map(|(i,_)|i).collect();
        let sample:Vec<_>=fields.iter().enumerate().filter(|(_,v)|**v=="SampleId").map(|(i,_)|i).collect();
        if width!=2||pred.len()!=1||sample.len()!=1{return Err(format!("unexpected CatBoost prediction header: {first:?}"))}
        pred_column=pred[0];sample_column=Some(sample[0]);
    }
    let mut v=Vec::with_capacity(n);
    for (i,line) in lines.enumerate(){
        let fields:Vec<_>=line.split('\t').collect();
        if fields.len()!=width{return Err(format!("prediction width mismatch at row {i}"))}
        if let Some(c)=sample_column{if fields[c].trim().parse::<usize>().ok()!=Some(i){return Err(format!("prediction row identity mismatch at row {i}"))}}
        let p=fields[pred_column].trim().parse::<f64>().map_err(|_|"invalid prediction".to_string())?;
        if !p.is_finite(){return Err("nonfinite prediction".into())}v.push(p)
    }
    if v.len()!=n{return Err("prediction count mismatch".into())}Ok(v)
}
fn read_predictions(path:&Path,n:usize,header:bool)->Result<Vec<f64>,String>{
    let s=std::fs::read_to_string(path).map_err(|e|e.to_string())?;parse_predictions(&s,n,header)
}
#[derive(Clone,Copy,Debug,PartialEq)]
struct LightConfig{rounds:usize,depth:usize,leaves:usize,eta:f64,min_leaf:usize}
impl LightConfig{
    fn old(rounds:usize)->Self{Self{rounds,depth:4,leaves:16,eta:0.1,min_leaf:32}}
    fn args(self,seed:u64)->Vec<String>{vec!["task=train".into(),"data=train.tsv".into(),"output_model=model.txt".into(),"objective=regression".into(),"boosting=gbdt".into(),format!("num_iterations={}",self.rounds),format!("num_leaves={}",self.leaves),format!("max_depth={}",self.depth),format!("learning_rate={}",self.eta),"max_bin=64".into(),format!("min_data_in_leaf={}",self.min_leaf),"num_threads=2".into(),format!("seed={seed}"),"deterministic=true".into(),"force_col_wise=true".into(),"verbosity=-1".into()]}
}
fn optimization_light_config(label:&str)->Result<LightConfig,String>{
    let (rounds,depth,leaves,eta,min_leaf)=match label{
        "r128-d4-l16-eta01-min32"=>(128,4,16,0.1,32),
        "r256-d4-l16-eta005-min32"=>(256,4,16,0.05,32),
        "r128-d6-l32-eta01-min32"=>(128,6,32,0.1,32),
        "r256-d6-l32-eta005-min32"=>(256,6,32,0.05,32),
        "r128-d4-l16-eta01-min64"=>(128,4,16,0.1,64),
        "r128-d3-l8-eta01-min16"=>(128,3,8,0.1,16),
        _=>return Err("unknown frozen C012 LightGBM configuration".into())
    };Ok(LightConfig{rounds,depth,leaves,eta,min_leaf})
}
/// Frozen O04-04 LightGBM member, with independent training and prediction.
pub struct LightGbm {
    models: Vec<String>,
    mean: f64,
    sd: f64,
}
fn light_predict(bin:&str,dir:&Path,n:usize,now:Instant)->Result<Vec<f64>,String>{
    command(bin,&["task=predict","data=eval.tsv","input_model=model.txt","output_result=pred.txt","num_threads=2","verbosity=-1"].map(String::from),dir,now)?;
    read_predictions(&dir.join("pred.txt"),n,false)
}
fn light_model_counts(s:&str,cfg:LightConfig)->Result<(usize,usize),String>{
    let counts:Vec<usize>=s.lines().filter_map(|l|l.strip_prefix("num_leaves=")).map(|v|v.parse::<usize>().map_err(|_|"invalid leaf count".to_string())).collect::<Result<_,_>>()?;
    if counts.is_empty()||counts.len()>cfg.rounds||counts.iter().any(|&n|n==0||n>cfg.leaves){return Err("invalid LightGBM tree counts".into())}
    Ok((counts.len(),counts.iter().sum()))
}
impl LightGbm {
    fn backend()->Result<String,String>{
        if std::env::var("AEVRAL_LIGHTGBM_VERSION").as_deref()!=Ok("4.6.0"){return Err("use pinned LightGBM 4.6.0 boosting shell".into())}
        std::env::var("AEVRAL_LIGHTGBM").map_err(|_|"use locked boosting shell".into())
    }
    fn config()->LightConfig{optimization_light_config("r256-d6-l32-eta005-min32").expect("frozen recipe")}
    pub fn fit(rows:&[super::TrainingRow],mean:f64,sd:f64)->Result<Self,String>{
        let selected=super::optimization::subset(rows,super::optimization::Rung::Final)?;
        Self::fit_selected(&selected,mean,sd)
    }
    fn fit_selected(rows:&[&super::TrainingRow],mean:f64,sd:f64)->Result<Self,String>{
        if rows.is_empty()||!mean.is_finite()||!sd.is_finite()||sd<=0.||rows.iter().any(|r|r.input.x.iter().chain(&r.normalized_target).any(|v|!v.is_finite())){return Err("invalid LightGBM training data".into())}
        let bin=Self::backend()?;
        let scratch=Scratch::new()?;let dir=&scratch.0;let now=Instant::now();
        let x:Vec<_>=rows.iter().map(|r|r.input.x).collect();
        let mut models=Vec::with_capacity(Y);
        for h in 0..Y{
            let y:Vec<_>=rows.iter().map(|r|r.normalized_target[h]).collect();
            write_rows(&dir.join("train.tsv"),&x,Some(&y))?;
            let _=std::fs::remove_file(dir.join("model.txt"));
            command(&bin,&Self::config().args(42),dir,now)?;
            let s=std::fs::read_to_string(dir.join("model.txt")).map_err(|e|e.to_string())?;
            light_model_counts(&s,Self::config())?;
            models.push(s);
        }
        Self::from_models(models,mean,sd)
    }
    pub(crate) fn from_models(models:Vec<String>,mean:f64,sd:f64)->Result<Self,String>{
        if models.len()!=Y||!mean.is_finite()||!sd.is_finite()||sd<=0.{return Err("invalid LightGBM artifact dimensions/scales".into())}
        for s in &models{
            if !s.lines().any(|l|l=="max_feature_idx=79"){return Err("invalid LightGBM feature count".into())}
            light_model_counts(s,Self::config())?;
        }
        Ok(Self{models,mean,sd})
    }
    /// Framed official LightGBM 4.6.0 text models, one per lead.
    pub fn model_bytes(&self)->Result<Vec<u8>,String>{
        let mut out=b"AEVLGBM\x01".to_vec();out.extend(b"4.6.0\0");
        for s in &self.models{
            let n=u32::try_from(s.len()).map_err(|_|"LightGBM model too large")?;
            out.extend(n.to_le_bytes());out.extend(s.as_bytes());
        }
        Ok(out)
    }
    pub fn from_model_bytes(bytes:&[u8],mean:f64,sd:f64)->Result<Self,String>{
        let mut r=bytes.strip_prefix(b"AEVLGBM\x014.6.0\0").ok_or("LightGBM format/backend mismatch")?;
        let mut models=Vec::with_capacity(Y);
        for _ in 0..Y{
            let n=u32::from_le_bytes(r.get(..4).ok_or("truncated LightGBM model length")?.try_into().unwrap())as usize;r=&r[4..];
            if n==0||n>16*1024*1024{return Err("invalid LightGBM model length".into())}
            let s=std::str::from_utf8(r.get(..n).ok_or("truncated LightGBM model")?).map_err(|_|"invalid LightGBM model encoding")?;
            models.push(s.to_owned());r=&r[n..];
        }
        if !r.is_empty(){return Err("trailing LightGBM payload".into())}
        Self::from_models(models,mean,sd)
    }
    pub fn model_texts(&self)->&[String]{&self.models}
    pub fn predict_batch(&self,rows:&[OriginInput])->Result<Vec<V>,String>{
        if rows.is_empty(){return Ok(Vec::new())}
        if rows.iter().any(|r|r.x.iter().any(|v|!v.is_finite())){return Err("nonfinite LightGBM query".into())}
        let bin=Self::backend()?;
        let scratch=Scratch::new()?;let dir=&scratch.0;let now=Instant::now();
        let x:Vec<_>=rows.iter().map(|r|r.x).collect();
        write_rows(&dir.join("eval.tsv"),&x,None)?;
        let mut result=vec![[0.;Y];rows.len()];
        for (h,s) in self.models.iter().enumerate(){
            std::fs::write(dir.join("model.txt"),s).map_err(|e|e.to_string())?;
            let _=std::fs::remove_file(dir.join("pred.txt"));
            for (i,p) in light_predict(&bin,dir,rows.len(),now)?.into_iter().enumerate(){
                let v=p*self.sd+self.mean;if !v.is_finite(){return Err("nonfinite LightGBM output".into())}result[i][h]=v;
            }
        }
        Ok(result)
    }
    pub fn predict(&self,row:&OriginInput)->Result<V,String>{Ok(self.predict_batch(std::slice::from_ref(row))?[0])}
}
fn cli_fit(backend:&str,d:&Data,eval:&[[f64;D]],y:&[f64],rounds:usize,seed:u64,dir:&Path,now:Instant,light:Option<LightConfig>)->Result<(Vec<f64>,usize,usize),String>{
    write_rows(&dir.join("train.tsv"),&d.x,Some(y))?;
    write_rows(&dir.join("eval.tsv"),eval,None)?;
    let _=std::fs::remove_file(dir.join("pred.txt"));let _=std::fs::remove_file(dir.join("model.txt"));let _=std::fs::remove_file(dir.join("model.json"));
    if backend=="lightgbm"{
        let bin=std::env::var("AEVRAL_LIGHTGBM").map_err(|_|"use locked boosting shell")?;
        let args=light.unwrap_or_else(||LightConfig::old(rounds)).args(seed);
        command(&bin,&args,dir,now)?;
        let predictions=light_predict(&bin,dir,eval.len(),now)?;
        let s=std::fs::read_to_string(dir.join("model.txt")).map_err(|e|e.to_string())?;
        let counts:Vec<usize>=s.lines().filter_map(|l|l.strip_prefix("num_leaves=")).map(|s|s.parse::<usize>().map_err(|_|"invalid leaf count".to_string())).collect::<Result<_,_>>()?;
        if counts.is_empty(){return Err("empty LightGBM model".into())}
        Ok((predictions,counts.len(),counts.iter().sum()))
    }else{
        let bin=std::env::var("AEVRAL_CATBOOST").map_err(|_|"use locked boosting shell")?;
        let mut cd=String::from("0\tLabel\n");for i in 0..D{cd.push_str(&format!("{}\tNum\n",i+1))}
        std::fs::write(dir.join("train.cd"),cd).map_err(|e|e.to_string())?;
        let cd=(0..D).map(|i|format!("{i}\tNum\n")).collect::<String>();std::fs::write(dir.join("eval.cd"),cd).map_err(|e|e.to_string())?;
        let args=vec!["fit".into(),"--learn-set".into(),"train.tsv".into(),"--column-description".into(),"train.cd".into(),"--loss-function".into(),"RMSE".into(),"--iterations".into(),rounds.to_string(),"--depth".into(),"4".into(),"--learning-rate".into(),"0.1".into(),"--grow-policy".into(),"SymmetricTree".into(),"--task-type".into(),"CPU".into(),"--thread-count".into(),"2".into(),"--random-seed".into(),seed.to_string(),"--logging-level".into(),"Silent".into(),"--model-file".into(),"model.json".into(),"--model-format".into(),"json".into()];
        command(&bin,&args,dir,now)?;
        command(&bin,&["calc","--input-path","eval.tsv","--column-description","eval.cd","--model-file","model.json","--model-format","json","--output-path","pred.txt","--output-columns","SampleId,RawFormulaVal","--prediction-type","RawFormulaVal","--thread-count","2"].map(String::from),dir,now)?;
        let s=std::fs::read_to_string(dir.join("model.json")).map_err(|e|e.to_string())?;
        let counts:Vec<usize>=s.split("\"leaf_values\"").skip(1).map(|p|{let a=p.split_once('[').ok_or("invalid CatBoost leaf array")?.1.split_once(']').ok_or("invalid CatBoost leaf array")?.0;Ok::<usize,String>(a.split(',').filter(|v|!v.trim().is_empty()).count())}).collect::<Result<_,_>>()?;
        if counts.is_empty()||counts.contains(&0){return Err("empty CatBoost model".into())}
        Ok((read_predictions(&dir.join("pred.txt"),eval.len(),true)?,counts.len(),counts.iter().sum()))
    }
}
fn all_leads(backend:&str,d:&Data,eval:&[[f64;D]],rounds:usize,seed:u64,leads:usize,now:Instant)->Result<Vec<V>,String>{
    let mut result=vec![[0.;Y];eval.len()];
    let xgb=if backend=="xgboost"{Some(Xgb::new()?)}else{None};
    let matrices=if let Some(x)=&xgb{Some((x.matrix(&d.x)?,x.matrix(eval)?))}else{None};
    let scratch=if xgb.is_none(){Some(Scratch::new()?)}else{None};
    for h in 0..leads{
        let timer=Instant::now();let y:Vec<_>=d.y.iter().map(|r|r[h]).collect();
        let(v,trees,leaves)=if let(Some(x),Some((tr,ev)))=(&xgb,&matrices){x.fit(tr,ev,&y,eval.len(),rounds,seed,now)?}
        else{cli_fit(backend,d,eval,&y,rounds,seed,&scratch.as_ref().unwrap().0,now,None)?};
        if v.len()!=eval.len()||v.iter().any(|p|!p.is_finite())||trees==0||trees>rounds||leaves<trees{return Err("invalid prediction or model counts".into())}
        for(i,p)in v.into_iter().enumerate(){result[i][h]=p}
        let elapsed=timer.elapsed().as_secs_f64();println!("upstream-lead,{backend},{},{trees},{leaves},{elapsed:.6}",h+1);
        if h==0{let est=now.elapsed().as_secs_f64()+2.*elapsed*(leads-1)as f64;println!("upstream-preflight,{backend},{rounds},{elapsed:.6},{est:.6}");if est>GUARD{return Err("preflight estimated fit exceeds 1200s".into())}}
    }Ok(result)
}
struct Model{pred:BTreeMap<i64,V>}
impl Predictor for Model{fn predict(&self,x:&OriginInput)->V{*self.pred.get(&x.origin).expect("verified evaluation origin")}}
fn config(label:&str)->Result<(&str,usize),String>{let (backend,rounds)=label.split_once('-').ok_or("invalid upstream configuration")?;if !["xgboost","lightgbm","catboost"].contains(&backend){return Err("unknown backend".into())}let rounds=rounds.parse::<usize>().map_err(|_|"invalid rounds")?;if ![64,128].contains(&rounds){return Err("rounds not in manifest".into())}Ok((backend,rounds))}
pub fn smoke(backend:&str)->Result<(),String>{
    if !["xgboost","lightgbm","catboost"].contains(&backend){return Err("unknown backend".into())}
    let mut x=vec![[0.;D];512];let mut y=vec![[0.;Y];512];for i in 0..512{x[i][0]=i as f64/512.;y[i].fill(if i<256{-2.}else{2.})}
    let d=Data{x,y};let mut ev=vec![[0.;D];2];ev[0][0]=0.2;ev[1][0]=0.8;
    let v=all_leads(backend,&d,&ev,64,42,1,Instant::now())?;if v[0][0]>-1.||v[1][0]<1.{return Err(format!("synthetic predictions incorrect: {v:?}"))}
    println!("upstream-smoke,{backend},{:.8},{:.8},pass",v[0][0],v[1][0]);Ok(())
}
pub fn run(data:&Path,id:&str,output:&Path,label:&str,seed:u64)->Result<(),String>{
    config(label)?;if ![42,123,2026].contains(&seed){return Err("seed not in manifest".into())}
    let f=load(data,Fold::development(id)?)?;
    let(n,m,baseline)=match id{"2019"=>(121849,7455,1.88752763),"2020"=>(129327,7543,2.07087759),"2021"=>(136893,7501,2.14898785),"2022"=>(144417,7639,2.10544725),_=>return Err("annual folds only".into())};
    if(f.train.len(),f.eval.len())!=(n,m){return Err("frozen cohort mismatch".into())}
    let mut daily=Metrics::default();for r in &f.eval{daily.add(&r.target,&r.input.recent)}
    if(daily.mae()-baseline).abs()>1e-6{return Err("daily baseline mismatch".into())}
    println!("cohort,{id},{n},{m},{},{}",f.possible,f.temperature_complete);println!("scales,{id},{:?}",f.scales);println!("baseline,{id},daily,{:.8},{:.8}",daily.mae(),daily.rmse());
    run_prepared(&f,output,label,seed)
}
pub(super) fn run_prepared(f:&super::PreparedFold,output:&Path,label:&str,seed:u64)->Result<(),String>{
    let id=f.fold.id;
    let(backend,rounds)=config(label)?;
    let m=f.eval.len();
    let mut out=BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);writeln!(out,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").map_err(|e|e.to_string())?;
    let now=Instant::now();let d=Data::from_rows(&f.train);
    if super::Fold::retrospective(id).is_ok(){
        let n=d.x.len();
        let sampled=super::neighbors::subset(&f.train,n);
        if n!=24576||sampled.iter().zip(d.x.iter().zip(&d.y)).any(|(r,(x,y))|r.input.x!=*x||r.normalized_target!=*y){return Err("XGBoost actual matrix mismatch".into())}
        super::representation::origins(&output.parent().ok_or("output directory")?.join("backend-fit-origins.txt"),sampled.iter().map(|r|r.input.origin))?;
    }let ev:Vec<_>=f.eval.iter().map(|r|r.input.x).collect();
    match all_leads(backend,&d,&ev,rounds,seed,Y,now){
        Ok(pred)=>{let mut map=BTreeMap::new();for(r,p)in f.eval.iter().zip(pred){let v=std::array::from_fn(|h|p[h]*f.target_sd+f.target_mean);if map.insert(r.input.origin,v).is_some(){return Err("duplicate origin".into())}}if map.len()!=m{return Err("prediction coverage mismatch".into())}
            println!("upstream-config,{id},{label},{seed},{},{rounds},24,2,{:.6}",d.x.len(),now.elapsed().as_secs_f64());score(&Model{pred:map},label,&f,&mut out)?;},
        Err(e) if e.contains("guard")||e.contains("preflight")=>println!("blocked,{id},{label},{e}"),
        Err(e)=>return Err(e)
    }out.flush().map_err(|e|e.to_string())?;Ok(())
}
/// C012: fixed-budget official LightGBM, one scalar regressor per lead.
pub(super) fn optimize(f:&super::PreparedFold,rows:&[&super::TrainingRow],label:&str,configuration:&str,out:&mut impl Write)->Result<(),String>{
    let cfg=optimization_light_config(configuration)?;let id=f.fold.id;let now=Instant::now();
    let n=rows.len();let m=f.eval.len();
    let workspace=8*(n+m)*(D+Y)*4+8*n*cfg.leaves*4;
    if workspace>2_000_000_000{println!("blocked,{id},{label},preflight workspace guard");return Ok(())}
    let d=Data{x:rows.iter().map(|r|r.input.x).collect(),y:rows.iter().map(|r|r.normalized_target).collect()};
    let scratch=Scratch::new()?;
    // Disposable probe uses only training rows and one lead, not evaluation labels.
    let probe_n=n.min(512);let probe=Data{x:(0..probe_n).map(|i|d.x[i*n/probe_n]).collect(),y:(0..probe_n).map(|i|d.y[i*n/probe_n]).collect()};
    let probe_eval:Vec<_>=probe.x.iter().take(32).copied().collect();let y:Vec<_>=probe.y.iter().map(|r|r[0]).collect();
    let timer=Instant::now();
    match cli_fit("lightgbm",&probe,&probe_eval,&y,cfg.rounds,42,&scratch.0,now,Some(cfg)){
        Ok(_)=>{},Err(e)if e.contains("guard")=>{println!("blocked,{id},{label},{e}");return Ok(())},Err(e)=>return Err(e)
    }
    let probe_s=timer.elapsed().as_secs_f64();let estimate=probe_s*n as f64/probe_n as f64*Y as f64;
    println!("optimization-light-preflight,{id},{label},{n},{probe_n},{workspace},{probe_s:.6},{estimate:.6}");
    if 2.*estimate+180.>GUARD{println!("blocked,{id},{label},preflight runtime guard");return Ok(())}
    let ev:Vec<_>=f.eval.iter().map(|r|r.input.x).collect();let mut pred=vec![[0.;Y];m];let mut leaves_total=0usize;
    let timer=Instant::now();
    for h in 0..Y{
        let y:Vec<_>=d.y.iter().map(|r|r[h]).collect();let lead_timer=Instant::now();
        let(v,trees,leaves)=match cli_fit("lightgbm",&d,&ev,&y,cfg.rounds,42,&scratch.0,now,Some(cfg)){
            Ok(v)=>v,Err(e)if e.contains("guard")=>{println!("blocked,{id},{label},{e}");return Ok(())},Err(e)=>return Err(e)
        };
        if v.len()!=m||v.iter().any(|v|!v.is_finite())||trees==0||trees>cfg.rounds||leaves<trees||leaves>trees*cfg.leaves{return Err("invalid C012 LightGBM counts".into())}
        for(i,p)in v.into_iter().enumerate(){pred[i][h]=p;}leaves_total+=leaves;
        println!("upstream-lead,lightgbm,{},{trees},{leaves},{:.6}",h+1,lead_timer.elapsed().as_secs_f64());
    }
    // CLI adapters combine fit and prediction; keep joint cost, do not invent separation.
    let joint=timer.elapsed().as_secs_f64();
    let mut map=BTreeMap::new();for(r,p)in f.eval.iter().zip(pred){if map.insert(r.input.origin,f.restore(p)).is_some(){return Err("duplicate origin".into())}}
    println!("optimization-light-config,{id},{label},{n},{},{},{},{},{},24,{leaves_total},{joint:.6}",cfg.rounds,cfg.depth,cfg.leaves,cfg.eta,cfg.min_leaf);
    let timer=Instant::now();score(&Model{pred:map},label,f,out)?;
    println!("optimization-cost,{id},{label},{joint:.6},{:.6},{leaves_total}",timer.elapsed().as_secs_f64());
    println!("optimization-cost-note,{id},{label},joint-cli-fit-predict-plus-export");Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn configs(){assert_eq!(config("xgboost-64").unwrap(),("xgboost",64));assert!(config("xgboost-32").is_err());assert!(config("fake-64").is_err());}
    #[test]fn prediction_identity(){assert_eq!(parse_predictions("SampleId\tRawFormulaVal\n0\t-2\n1\t2\n",2,true).unwrap(),vec![-2.,2.]);assert!(parse_predictions("SampleId\tRawFormulaVal\n1\t-2\n0\t2\n",2,true).is_err());assert!(parse_predictions("SampleId\tRawFormulaVal\n0\tNaN\n",1,true).is_err());assert!(parse_predictions("RawFormulaVal\n2\n",1,true).is_err());assert_eq!(parse_predictions("-2\n2\n",2,false).unwrap(),vec![-2.,2.]);}
    #[test]fn label_free_export(){let p=std::env::temp_dir().join(format!("aevral-export-test-{}",std::process::id()));let x=vec![[1.;D];2];write_rows(&p,&x,None).unwrap();let s=std::fs::read_to_string(&p).unwrap();assert!(s.lines().all(|r|r.split('\t').count()==D));write_rows(&p,&x,Some(&[5.,6.])).unwrap();let s=std::fs::read_to_string(&p).unwrap();assert!(s.lines().all(|r|r.split('\t').count()==D+1));std::fs::remove_file(p).unwrap();}
    #[test]fn optimization_light_grid_and_legacy_args(){
        let c=optimization_light_config("r128-d4-l16-eta01-min32").unwrap();
        assert_eq!(c,LightConfig::old(128));assert_eq!(c.args(42),LightConfig::old(128).args(42));
        let args=optimization_light_config("r256-d6-l32-eta005-min32").unwrap().args(42);
        for v in ["num_iterations=256","max_depth=6","num_leaves=32","learning_rate=0.05","min_data_in_leaf=32","deterministic=true"]{assert!(args.iter().any(|a|a==v));}
        assert!(optimization_light_config("r999").is_err());
    }
}

#[cfg(test)]
mod deployment_tests {
    use super::*;
    use super::super::TrainingRow;
    #[test]
    #[ignore = "requires pinned LightGBM backend"]
    fn split_lightgbm_parity(){
        let rows:Vec<_>=(0..512).map(|i|{
            let x=std::array::from_fn(|j|((i*(j+1))%127)as f64/127.);
            let y=std::array::from_fn(|j|((i+j)%59)as f64/59.);
            TrainingRow{input:OriginInput{origin:i as i64*60,x,recent:[0.;Y]},target:y,normalized_target:y}
        }).collect();
        let refs:Vec<_>=rows.iter().collect();
        let model=LightGbm::fit_selected(&refs,10.,2.).unwrap();
        let query:Vec<_>=rows.iter().step_by(23).map(|r|r.input.clone()).collect();
        let actual=model.predict_batch(&query).unwrap();
        let restored=LightGbm::from_model_bytes(&model.model_bytes().unwrap(),10.,2.).unwrap();
        assert_eq!(actual,restored.predict_batch(&query).unwrap());
        let d=Data{x:rows.iter().map(|r|r.input.x).collect(),y:rows.iter().map(|r|r.normalized_target).collect()};
        let ev:Vec<_>=query.iter().map(|r|r.x).collect();
        let scratch=Scratch::new().unwrap();let now=Instant::now();
        for h in 0..Y{
            let y:Vec<_>=d.y.iter().map(|r|r[h]).collect();
            let (p,_,_)=cli_fit("lightgbm",&d,&ev,&y,256,42,&scratch.0,now,Some(LightGbm::config())).unwrap();
            for (i,v) in p.iter().enumerate(){assert_eq!(actual[i][h],v*2.+10.);}
        }
        assert!(LightGbm::from_models(vec![],10.,2.).is_err());
        println!("lightgbm-split-parity,24,pass");
    }
}
