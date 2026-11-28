//! C015 chronological combination adapter. Reuse original C013/C014 numerical code.
#[path="../014-probabilistic/probabilistic.rs"] mod probabilistic;
use probabilistic::{ensemble::{self,Features,Row,Fold,Learned,Linear},core::{self,Metrics},regression::Node,Model};
use std::{fs,io::{BufRead,BufReader,BufWriter,Write},path::Path,time::Instant};
const FOUR:&str="fold,origin,lead,target,daily,I,L,R,M";
const COHORTS:&str=include_str!("../../results/C015/cohorts.csv");
fn finite(s:&str)->f64{let v=s.parse::<f64>().expect("numeric field");assert!(v.is_finite());v}
fn bounds(y:u16)->(i64,i64,usize){
    let r:Vec<_>=COHORTS.lines().skip(1).find(|s|s.split(',').next()==Some(&y.to_string())).expect("unknown retrospective year").split(',').collect();
    (r[1].parse().unwrap(),r[2].parse().unwrap(),r[4].parse().unwrap())
}
fn month(y:u16,origin:i64)->usize{
    let(start,end,_)=bounds(y);assert!(origin%60==0&&origin+60>=start&&origin+1440<end);
    let mut d=(origin+60-start)/1440;
    for (i,n) in [31,if y==2024{29}else{28},31,30,31,30,31,31,30,31,30,31].into_iter().enumerate(){if d<n{return i+1}d-=n;}
    panic!("invalid month")
}
fn read_rows(reader:impl BufRead,y:u16,k:usize,count:usize)->Fold{
    assert!(matches!(k,4|8));
    let mut lines=reader.lines();
    assert_eq!(lines.next().unwrap().unwrap(),if k==4{FOUR}else{ensemble::HEADER});
    let mut rows:Vec<Row>=Vec::with_capacity(count*24);
    for line in lines{
        let l=line.unwrap();let p:Vec<_>=l.split(',').collect();assert_eq!(p.len(),5+k);assert_eq!(p[0].parse::<u16>().unwrap(),y);
        let origin=p[1].parse::<i64>().unwrap();let lead=p[2].parse::<usize>().unwrap();
        assert_eq!(lead,rows.len()%24+1);
        if let Some(last)=rows.last(){if lead==1{assert!(origin>last.x.origin)}else{assert_eq!(origin,last.x.origin)}}
        // In four-member mode unused coordinates remain NaN: any accidental
        // eight-member numerical use fails closed, not a substituted forecast.
        let base=std::array::from_fn(|i|if i<k{finite(p[5+i])}else{f64::NAN});
        rows.push(Row{x:Features{origin,lead,month:month(y,origin),base},target:finite(p[3]),daily:finite(p[4])});
    }
    assert_eq!(rows.len(),count*24);Fold{year:y,rows}
}
fn load(root:&Path,y:u16,k:usize)->Fold{
    if y<=2022{return ensemble::load(&root.join(format!("{y}.csv")),y)}
    let(_,_,n)=bounds(y);
    read_rows(BufReader::new(fs::File::open(root.join(format!("{y}-{k}.csv"))).unwrap()),y,k,n)
}
fn calibration(history:&[Fold],assessment:u16)->Vec<&Row>{
    assert!((2023..=2026).contains(&assessment));assert_eq!(history.iter().map(|f|f.year).collect::<Vec<_>>(),(2020..assessment).collect::<Vec<_>>());
    assert!(history.iter().all(|f|!f.rows.is_empty()));history.iter().flat_map(|f|f.rows.iter()).collect()
}
fn linear(out:&mut impl Write,h:usize,m:&Linear){
    write!(out,"linear,{h},{},{},{:.17e},{:.17e}",m.iterations,m.converged,m.objective,m.intercept).unwrap();
    for w in m.weights{write!(out,",{w:.17e}").unwrap()}writeln!(out).unwrap();
}
fn node(out:&mut impl Write,h:usize,path:&str,n:&Node){
    match n{
        Node::Leaf{count,impurity,support}=>{write!(out,"leaf,{h},{path},{count},{impurity:.17e}").unwrap();for x in support{write!(out,",{x:.17e}").unwrap()}writeln!(out).unwrap();}
        Node::Split{count,impurity,objective,feature,threshold,left,right}=>{writeln!(out,"split,{h},{path},{count},{impurity:.17e},{objective:.17e},{feature},{threshold:.17e}").unwrap();node(out,h,&format!("{path}L"),left);node(out,h,&format!("{path}R"),right);}
    }
}
enum Fitted{Mean,Linear(Learned),Distribution(Model)}
fn main(){
    let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),5,"combinations JOINED_DIR OUT_DIR YEAR ID");
    let root=Path::new(&a[1]);let out=Path::new(&a[2]);let y=a[3].parse::<u16>().unwrap();assert!((2023..=2026).contains(&y));
    let id=a[4].as_str();let k=match id{"E08"|"P13"=>4,"E13"|"E14"=>8,_=>panic!("unknown recipe")};
    fs::create_dir_all(out).unwrap();
    let history:Vec<_>=if id=="E08"{Vec::new()}else{(2020..y).map(|v|load(root,v,k)).collect()};
    let refs=if id=="E08"{Vec::new()}else{calibration(&history,y)};
    let now=Instant::now();let fitted=match id{"E08"=>Fitted::Mean,"E13"=>Fitted::Linear(ensemble::fit_linear(13,&refs)),"E14"=>Fitted::Linear(ensemble::fit_linear(14,&refs)),"P13"=>Fitted::Distribution(probabilistic::fit(13,&refs)),_=>unreachable!()};
    let fit=now.elapsed().as_secs_f64();
    let mut param=BufWriter::new(fs::File::create(out.join("parameters.csv")).unwrap());
    writeln!(param,"recipe,{y},{id},calibration_rows,{}",refs.len()).unwrap();
    match &fitted{Fitted::Mean=>writeln!(param,"mean,I,L,R,M").unwrap(),Fitted::Linear(Learned::Global(m))=>linear(&mut param,0,m),Fitted::Linear(Learned::Lead(v))=>for (h,m) in v.iter().enumerate(){linear(&mut param,h+1,m)},Fitted::Distribution(Model::Tree(t))=>{writeln!(param,"tree,{}",t.b).unwrap();for(h,n)in t.roots.iter().enumerate(){node(&mut param,h+1,"R",n)}},_=>unreachable!()}
    param.flush().unwrap();drop(refs);drop(history);
    // The current assessment targets are opened only after calibration is frozen.
    let f=load(root,y,k);
    let mut pred=BufWriter::new(fs::File::create(out.join("predictions.csv")).unwrap());
    writeln!(pred,"fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c").unwrap();
    let mut dist=if id=="P13"{Some(BufWriter::new(fs::File::create(out.join("distribution.csv")).unwrap()))}else{None};
    if let Some(w)=&mut dist{write!(w,"origin,lead,target,median").unwrap();for i in 1..=99{write!(w,",q{i}").unwrap()}writeln!(w).unwrap();}
    let mut metrics=ensemble::Scores::default();let mut prob=Metrics::default();
    let mut pl:[Metrics;24]=std::array::from_fn(|_|Metrics::default());let mut pm:[Metrics;12]=std::array::from_fn(|_|Metrics::default());let mut ps:[Metrics;4]=std::array::from_fn(|_|Metrics::default());
    let now=Instant::now();
    for r in &f.rows{
        let p=match &fitted{Fitted::Mean=>ensemble::static_predict(8,&r.x),Fitted::Linear(m)=>m.predict(&r.x),Fitted::Distribution(m)=>{
            let (v,cross)=m.support(&r.x);assert!(!cross);let v=v.unwrap();assert_eq!(v.len(),99);
            let p=core::distribution(&v,r.target);
            prob.add(r.target,&p);pl[r.x.lead-1].add(r.target,&p);pm[r.x.month-1].add(r.target,&p);ps[ensemble::season(r.x.month)].add(r.target,&p);
            let w=dist.as_mut().unwrap();write!(w,"{},{},{},{}",r.x.origin,r.x.lead,r.target,p.median).unwrap();for x in v{write!(w,",{x:.17e}").unwrap()}writeln!(w).unwrap();p.median
        }};
        assert!(p.is_finite());metrics.add(r,p);writeln!(pred,"{y},{id},{},{},{},{}",r.x.origin,r.x.lead,r.target,p).unwrap();
    }
    pred.flush().unwrap();if let Some(w)=&mut dist{w.flush().unwrap();}
    let elapsed=now.elapsed().as_secs_f64();metrics.validate(f.rows.len()/24);
    println!("score,{y},{id},{},{:.8},{:.8},{elapsed:.6}",metrics.all.count/24,metrics.all.mae(),metrics.all.rmse());
    for(i,m)in metrics.leads.iter().enumerate(){println!("lead,{y},{id},{},{:.8},{:.8}",i+1,m.mae(),m.rmse());}
    for(i,m)in metrics.months.iter().enumerate(){if m.count>0{println!("month,{y},{id},{},{},{:.8},{:.8}",i+1,m.count/24,m.mae(),m.rmse());}}
    if id=="P13"{
        let mut w=BufWriter::new(fs::File::create(out.join("probabilistic-metrics.csv")).unwrap());
        write!(w,"group,index,targets,absolute_sum,squared_sum,crps_sum").unwrap();for i in 1..=9{write!(w,",pinball{i}").unwrap()}for i in 1..=4{write!(w,",coverage{i},width{i},interval_score{i}").unwrap()}writeln!(w).unwrap();
        let mut emit=|group:&str,i:usize,m:&Metrics|{if m.n==0{return}assert_eq!(m.n,m.distribution_n);write!(w,"{group},{i},{},{:.17e},{:.17e},{:.17e}",m.n,m.ae,m.se,m.crps).unwrap();for p in m.pinball{write!(w,",{p:.17e}").unwrap()}for j in 0..4{write!(w,",{:.17e},{:.17e},{:.17e}",m.coverage[j],m.width[j],m.interval_score[j]).unwrap()}writeln!(w).unwrap();};
        emit("all",0,&prob);for (kind,ms) in [("lead",&pl[..]),("month",&pm[..]),("season",&ps[..])]{for(i,m)in ms.iter().enumerate(){emit(kind,i+1,m)}}
        w.flush().unwrap();
    }
    println!("combination-cost,{y},{id},{fit:.6},{elapsed:.6}");
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn calendar(){let(s,e,_)=bounds(2026);assert_eq!(month(2026,s-60),1);assert_eq!(month(2026,e-1500),10);let(s,_,_)=bounds(2024);assert_eq!(month(2024,s+59*1440-60),2);assert_eq!(month(2024,s+60*1440-60),3);}
    #[test]fn four_only(){let mut s=format!("{FOUR}\n");for h in 1..=24{s+=&format!("2023,12097380,{h},3,2,1,2,3,4\n");}let f=read_rows(s.as_bytes(),2023,4,1);assert_eq!(ensemble::static_predict(8,&f.rows[0].x),2.5);assert!(f.rows[0].x.base[4].is_nan());}
    #[test]#[should_panic]fn future_rejected(){calibration(&[Fold{year:2023,rows:Vec::new()}],2023);}
}
