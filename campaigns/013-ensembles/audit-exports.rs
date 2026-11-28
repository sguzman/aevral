//! Independent output audit. Does not import the ensemble engine.
use std::{collections::{BTreeMap,BTreeSet},fs,io::{BufRead,BufReader},path::Path};
type Key=(String,String,String,usize);
#[derive(Default)]
struct Sum{n:usize,a:f64,s:f64}
impl Sum{fn add(&mut self,e:f64){self.n+=1;self.a+=e.abs();self.s+=e*e;}}
fn val(s:&str)->f64{let n:f64=s.parse().unwrap();assert!(n.is_finite());n}
fn lines(p:&Path)->impl Iterator<Item=String>{BufReader::new(fs::File::open(p).unwrap()).lines().map(Result::unwrap)}
fn main(){
 let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),3);
 let root=Path::new(&args[1]);let out=Path::new(&args[2]);
 let mut labels:Vec<String>=["daily","I","L","R","M","S","A","H","X"].iter().map(|s|s.to_string()).collect();
 labels.extend((1..=24).map(|i|format!("E{i:02}")));
 let mut sums:BTreeMap<Key,Sum>=BTreeMap::new();
 for (year,n,start) in [("2021",7501,11046240i64),("2022",7639,11571840)]{
  let mut src=lines(&root.join(format!("{year}.csv")));
  let mut pred=lines(&out.join(format!("predictions-{year}.csv")));
  let header=src.next().unwrap();assert_eq!(header,"fold,origin,lead,target,daily,I,L,R,M,S,A,H,X");
  let expect=format!("{},{}",header,labels[9..].join(","));assert_eq!(pred.next().unwrap(),expect);
  let mut count=0;
  for line in src{
   let p=pred.next().expect("missing export row");let a:Vec<_>=line.split(',').collect();let b:Vec<_>=p.split(',').collect();assert_eq!(a.len(),13);assert_eq!(b.len(),37);
   assert_eq!(&a[..3],&b[..3]);assert_eq!(a[0],year);
   for i in 3..13{assert_eq!(val(a[i]),val(b[i]),"changed source column");}
   let h:usize=a[2].parse().unwrap();assert_eq!(h,count%24+1);
   let d=(a[1].parse::<i64>().unwrap()+60-start).div_euclid(1440);assert!((0..365).contains(&d));
   let m=[31,59,90,120,151,181,212,243,273,304,334,365].iter().position(|&end|d<end).unwrap()+1;
   let y=val(b[3]);
   for (i,label) in labels.iter().enumerate(){
    let e=val(b[4+i])-y;
    for (kind,index) in [("score",0),("lead",h),("month",m)]{
     sums.entry((year.into(),label.clone(),kind.into(),index)).or_default().add(e);
    }
   }
   count+=1;
  }
  assert!(pred.next().is_none());assert_eq!(count,n*24);println!("export-rows,{year},{count},pass");
 }
 let mut seen=BTreeSet::new();let mut score_count=0;
 for l in lines(&out.join("metrics.csv")){
  let p:Vec<_>=l.split(',').collect();let kind=p[0];let (index,count,mae,rmse)=match kind{
   "score"=>{assert_eq!(p.len(),8);assert!(val(p[6])>=0.&&val(p[7])>=0.);score_count+=1;(0,p[3],p[4],p[5])},
   "lead"|"month"=>{assert_eq!(p.len(),7);(p[3].parse().unwrap(),p[4],p[5],p[6])},_=>panic!("bad metric kind")};
  let k=(p[1].into(),p[2].into(),kind.into(),index);assert!(seen.insert(k.clone()));
  let s=&sums[&k];let n=count.parse::<usize>().unwrap();
  assert_eq!(s.n,if kind=="lead"{n}else{n*24});
  assert!((s.a/s.n as f64-val(mae)).abs()<1e-7);
  assert!(((s.s/s.n as f64).sqrt()-val(rmse)).abs()<1e-7);
 }
 assert_eq!(score_count,66);assert_eq!(seen.len(),sums.len());assert_eq!(seen.len(),66*37);
 println!("metric-audit,{score_count},{},pass",seen.len());
 let mut co=lines(&out.join("coefficients.csv"));assert_eq!(co.next().unwrap(),"fold,recipe,group,calibration_rows,fallback,iterations,converged,objective,intercept,I,L,R,M,S,A,H,X");
 let mut coeffs=BTreeMap::new();
 for l in co{
  let p:Vec<_>=l.split(',').collect();assert_eq!(p.len(),17);
  let year=p[0];assert!(matches!(year,"2021"|"2022"));let id:u8=p[1].strip_prefix('E').unwrap().parse().unwrap();
  assert!(matches!(id,11..=19|21|22|24));
  let group=p[2];let n:usize=p[3].parse().unwrap();let full=if year=="2021"{181032}else{361056};
  let fallback:bool=p[4].parse().unwrap();let iter:usize=p[5].parse().unwrap();let converged:bool=p[6].parse().unwrap();
  assert!(val(p[7])>=0.);let intercept=val(p[8]);let w:Vec<_>=p[9..].iter().map(|s|val(s)).collect();
  assert!((w.iter().sum::<f64>()-1.).abs()<1e-8);
  if matches!(id,11..=14|21|22|24){assert!(w.iter().all(|&x|x>=-1e-12));assert!(intercept.abs()<1e-12);assert!((1..=2000).contains(&iter));assert!(converged||iter==2000);}
  else if id==19{assert_eq!(iter,0);assert_eq!(w,[1.,0.,0.,0.,0.,0.,0.,0.]);}
  else{assert_eq!(iter,1);assert!(converged);}
  if matches!(id,11|12|15|16|21|22|24){assert!(w[3..].iter().all(|&x|x==0.));}
  match id{
   12|14|16|18|24=>{let h:usize=group.strip_prefix("lead-").unwrap().parse().unwrap();assert!((1..=24).contains(&h));assert_eq!(n,full/24);assert!(!fallback);},
   21|22=>{let (prefix,k)=if id==21{("spread-",2)}else{("season-",4)};let g:usize=group.strip_prefix(prefix).unwrap().parse().unwrap();assert!(g<k&&n<=full);assert_eq!(fallback,n<1000);},
   _=>{assert_eq!(group,"global");assert_eq!(n,full);assert!(!fallback);}
  }
  assert!(coeffs.insert((year.to_owned(),id,group.to_owned()),n).is_none());
 }
 for year in ["2021","2022"]{
  let full=if year=="2021"{181032}else{361056};
  for id in [11,13,15,17,19]{assert_eq!(coeffs[&(year.into(),id,"global".into())],full);}
  for id in [12,14,16,18,24]{for h in 1..=24{assert_eq!(coeffs[&(year.into(),id,format!("lead-{h}"))],full/24);}}
  for (id,prefix,k) in [(21,"spread",2),(22,"season",4)]{assert_eq!((0..k).map(|g|coeffs[&(year.into(),id,format!("{prefix}-{g}"))]).sum::<usize>(),full);}
 }
 assert_eq!(coeffs.len(),262);println!("coefficient-audit,{},pass",coeffs.len());
 let mut params=BTreeSet::new();
 for l in lines(&out.join("parameters.csv")){
  let p:Vec<_>=l.split(',').collect();assert!(matches!(p[1],"2021"|"2022"));
  let k=match p[0]{
   "calibration"=>{assert_eq!(p.len(),4);assert_eq!((p[2],p[3]),if p[1]=="2021"{("2020","181032")}else{("2020+2021","361056")});format!("{},calibration",p[1])},
   "static"=>{assert_eq!(p.len(),5);let expected=match p[2]{"E01"|"E20"=>("Mean","ILR"),"E02"=>("Median","ILR"),"E03"=>("Mean","ILRMSAHX"),"E04"=>("Median","ILRMSAHX"),"E05"=>("Mean","IL"),"E06"=>("Mean","IR"),"E07"=>("Mean","LR"),"E08"=>("Mean","ILRM"),"E09"=>("Mean","AHX"),"E10"=>("Mean","IS"),_=>panic!("static id")};assert_eq!((p[3],p[4]),expected);format!("{},static,{}",p[1],p[2])},
   "bias"=>{assert_eq!(p.len(),5);assert!(matches!(p[2],"E19"|"E20"|"E24"));assert!((1..=24).contains(&p[3].parse::<usize>().unwrap()));val(p[4]);format!("{},bias,{},{}",p[1],p[2],p[3])},
   "spread-threshold"=>{assert_eq!(p.len(),4);assert_eq!(p[2],"E21");assert!(val(p[3])>=0.);format!("{},spread",p[1])},
   "selected"=>{assert_eq!(p.len(),4);assert_eq!(p[2],"E23");assert_eq!(p[3].len(),3);assert_eq!(p[3].chars().collect::<BTreeSet<_>>().len(),3);assert!(p[3].chars().all(|c|"ILRMSAHX".contains(c)));format!("{},selected",p[1])},
   "correlation"=>{assert_eq!(p.len(),6);assert_eq!(p[2],"E23");let a="ILRMSAHX".find(p[3]).unwrap();let b="ILRMSAHX".find(p[4]).unwrap();assert!(p[3].len()==1&&p[4].len()==1&&a<b);assert!(p[5]=="undefined"||val(p[5]).abs()<=1.);format!("{},correlation,{},{}",p[1],p[3],p[4])},
   _=>panic!("unknown parameter")
  };assert!(params.insert(k));
 }
 assert_eq!(params.len(),228);println!("parameter-audit,{},pass",params.len());
}
