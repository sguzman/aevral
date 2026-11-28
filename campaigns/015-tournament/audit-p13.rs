//! Independent P13 reconstruction and scoring. No producer/model imports.
use std::{collections::BTreeMap,fs::File,io::{BufRead,BufReader},path::Path};
const YEARS:&str=include_str!("../../results/C015/cohorts.csv");
const Q:[f64;9]=[0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975];
const C:[f64;4]=[0.50,0.80,0.90,0.95];
fn num(s:&str)->f64{let v=s.parse::<f64>().unwrap();assert!(v.is_finite());v}
fn count(s:&str)->usize{s.parse().unwrap()}
fn close(a:f64,b:f64,tol:f64){assert!((a-b).abs()<=tol,"mismatch {a} {b}")}
fn quant(v:&[f64],p:f64)->f64{let q=p*v.len() as f64;let i=if (q-q.round()).abs()<1e-10{q.round()}else{q.ceil()};v[(i as usize).clamp(1,v.len())-1]}
fn reader(p:&Path)->BufReader<File>{BufReader::new(File::open(p).unwrap())}
fn bounds(y:u16)->(i64,i64,usize){
    match y{2020=>(10519200,11046240,7543),2021=>(11046240,11571840,7501),2022=>(11571840,12097440,7639),_=>{
        let l=YEARS.lines().skip(1).find(|l|l.split(',').next()==Some(&y.to_string())).unwrap();
        let p:Vec<_>=l.split(',').collect();(p[1].parse().unwrap(),p[2].parse().unwrap(),count(p[4]))
    }}
}
fn month(y:u16,t:i64)->usize{let(s,e,_)=bounds(y);assert!(t%60==0&&t+60>=s&&t+1440<e);let mut d=(t+60-s)/1440;for(i,n)in [31,if y==2020||y==2024{29}else{28},31,30,31,30,31,31,30,31,30,31].into_iter().enumerate(){if d<n{return i+1}d-=n;}panic!("bad month")}
#[derive(Clone,Copy)]
struct Row{t:i64,h:usize,y:f64,c:f64,d:f64}
fn read(root:&Path,y:u16)->Vec<Row>{
    let historic=y<=2022;let path=root.join(if historic{format!("{y}.csv")}else{format!("{y}-4.csv")});
    let mut lines=reader(&path).lines();assert_eq!(lines.next().unwrap().unwrap(),if historic{"fold,origin,lead,target,daily,I,L,R,M,S,A,H,X"}else{"fold,origin,lead,target,daily,I,L,R,M"});
    let mut rows:Vec<Row>=Vec::new();
    for l in lines{let l=l.unwrap();let p:Vec<_>=l.split(',').collect();assert_eq!(p.len(),if historic{13}else{9});assert_eq!(p[0].parse::<u16>().unwrap(),y);
        let t=p[1].parse::<i64>().unwrap();month(y,t);let h=count(p[2]);assert_eq!(h,rows.len()%24+1);
        if let Some(r)=rows.last(){assert!(if h==1{t>r.t}else{t==r.t})}
        let b:Vec<_>=p[5..9].iter().map(|s|num(s)).collect();let c=b.iter().sum::<f64>()/4.;let d=b.iter().copied().fold(f64::NEG_INFINITY,f64::max)-b.iter().copied().fold(f64::INFINITY,f64::min);
        rows.push(Row{t,h,y:num(p[3]),c,d});
    }
    assert_eq!(rows.len(),bounds(y).2*24);rows
}
enum Node{Split{n:usize,f:usize,t:f64},Leaf{n:usize,v:Vec<f64>}}
type Tree=BTreeMap<(usize,String),Node>;
fn tree(path:&Path,y:u16,n:usize)->Tree{
    let mut l=reader(path).lines();assert_eq!(l.next().unwrap().unwrap(),format!("recipe,{y},P13,calibration_rows,{n}"));assert_eq!(l.next().unwrap().unwrap(),"tree,true");
    let mut out=Tree::new();
    for row in l{let row=row.unwrap();let p:Vec<_>=row.split(',').collect();let h=count(p[1]);assert!((1..=24).contains(&h));let path=p[2];assert!(path.starts_with('R')&&path.len()<=3&&path[1..].chars().all(|c|matches!(c,'L'|'R')));let n=count(p[3]);assert!(n>=128);let imp=num(p[4]);assert!(imp>=-1e-12);
        let node=match p[0]{"leaf"=>{assert_eq!(p.len(),104);let v:Vec<_>=p[5..].iter().map(|s|num(s)).collect();assert!(v.windows(2).all(|w|w[0]<=w[1]));Node::Leaf{n,v}},"split"=>{assert_eq!(p.len(),8);assert!(path.len()<3&&n>=256&&num(p[5])<imp-1e-12);let f=count(p[6]);assert!(f<=1);Node::Split{n,f,t:num(p[7])}},_=>panic!("unknown node")};
        assert!(out.insert((h,path.into()),node).is_none());
    }out
}
fn route(tree:&Tree,r:&Row)->Vec<String>{let mut path="R".to_owned();let mut visited=vec![path.clone()];loop{match &tree[&(r.h,path.clone())]{Node::Leaf{..}=>return visited,Node::Split{f,t,..}=>{path.push(if (if *f==0{r.c}else{r.d})<=*t{'L'}else{'R'});visited.push(path.clone());}}}}
fn check_cal(tree:&Tree,rows:&[Row]){
    let mut counts:BTreeMap<(usize,String),usize>=BTreeMap::new();
    let mut residuals:BTreeMap<(usize,String),Vec<f64>>=BTreeMap::new();
    for r in rows{let p=route(tree,r);for v in &p{*counts.entry((r.h,v.clone())).or_default()+=1;}residuals.entry((r.h,p.last().unwrap().clone())).or_default().push(r.y-r.c);}
    assert_eq!(counts.len(),tree.len());
    for(k,node)in tree{match node{Node::Split{n,..}=>assert_eq!(counts[k],*n),Node::Leaf{n,v}=>{assert_eq!(counts[k],*n);let mut r=residuals.remove(k).unwrap();r.sort_by(f64::total_cmp);for(j,x)in v.iter().enumerate(){close(*x,quant(&r,(j as f64+0.5)/99.),1e-12)}}}}
    assert!(residuals.is_empty());
}
#[derive(Default)]
struct Sum{n:usize,v:[f64;24]}
impl Sum{
    fn add(&mut self,y:f64,v:&[f64]){
        self.n+=1;let e=v[49]-y;self.v[0]+=e.abs();self.v[1]+=e*e;
        // Independent CRPS via successive sorted-support gaps, not producer coefficients.
        let a=v.iter().map(|x|(x-y).abs()).sum::<f64>()/99.;
        let b=v.windows(2).enumerate().map(|(i,w)|((i+1)*(98-i))as f64*(w[1]-w[0])).sum::<f64>()/(99.*99.);
        self.v[2]+=a-b;
        for(i,p)in Q.iter().enumerate(){let e=y-quant(v,*p);self.v[3+i]+=if e>=0.{p*e}else{(p-1.)*e};}
        for(i,c)in C.iter().enumerate(){let l=quant(v,(1.-c)/2.);let u=quant(v,(1.+c)/2.);let w=u-l;self.v[12+3*i]+=f64::from(l<=y&&y<=u);self.v[13+3*i]+=w;self.v[14+3*i]+=w+2./(1.-c)*((l-y).max(0.)+(y-u).max(0.));}
    }
}
fn main(){
    let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),4,"audit-p13 JOINED_DIR OUTPUT_DIR YEAR");let root=Path::new(&a[1]);let out=Path::new(&a[2]);let y=a[3].parse::<u16>().unwrap();assert!((2023..=2026).contains(&y));
    let cal:Vec<_>=(2020..y).flat_map(|v|read(root,v)).collect();let nodes=tree(&out.join("parameters.csv"),y,cal.len());check_cal(&nodes,&cal);drop(cal);
    let rows=read(root,y);let mut d=reader(&out.join("distribution.csv")).lines();let mut header="origin,lead,target,median".to_string();for i in 1..=99{header+=&format!(",q{i}");}assert_eq!(d.next().unwrap().unwrap(),header);
    let mut sums:BTreeMap<(&str,usize),Sum>=BTreeMap::new();
    for r in &rows{
        let line=d.next().unwrap().unwrap();let p:Vec<_>=line.split(',').collect();assert_eq!(p.len(),103);assert_eq!(p[0].parse::<i64>().unwrap(),r.t);assert_eq!(count(p[1]),r.h);assert_eq!(num(p[2]),r.y);
        let path=route(&nodes,r);let Node::Leaf{v,..}=&nodes[&(r.h,path.last().unwrap().clone())]else{panic!("not leaf")};
        let v:Vec<_>=v.iter().map(|v|r.c+v).collect();close(num(p[3]),v[49],1e-12);for(i,x)in v.iter().enumerate(){close(num(p[4+i]),*x,1e-12);}
        let m=month(y,r.t);let s=match m{12|1|2=>1,3..=5=>2,6..=8=>3,_=>4};
        for k in [("all",0),("lead",r.h),("month",m),("season",s)]{sums.entry(k).or_default().add(r.y,&v);}
    }assert!(d.next().is_none());
    let mut metrics=reader(&out.join("probabilistic-metrics.csv")).lines();assert_eq!(metrics.next().unwrap().unwrap().split(',').count(),27);
    let mut seen=0;for l in metrics{let l=l.unwrap();let p:Vec<_>=l.split(',').collect();assert_eq!(p.len(),27);let group=match p[0]{"all"=>"all","lead"=>"lead","month"=>"month","season"=>"season",_=>panic!("unknown metric group")};let k=(group,count(p[1]));let s=sums.remove(&k).expect("unexpected or duplicate metric");assert_eq!(count(p[2]),s.n);for(i,v)in s.v.iter().enumerate(){close(num(p[3+i])/s.n as f64,*v/s.n as f64,1e-7);}seen+=1;}
    assert!(sums.is_empty());println!("p13-reconstruction,{y},{},{seen},pass",rows.len());
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn gap_crps(){let mut s=Sum::default();let v=vec![3.;99];s.add(1.,&v);assert_eq!(s.v[2],2.);assert_eq!(s.v[0],2.);assert_eq!(s.v[1],4.);}
    #[test]fn rank_ties(){assert_eq!(quant(&[1.,2.,3.,4.],0.5),2.);assert_eq!(quant(&[1.,2.,3.,4.],0.95),4.);}
    #[test]fn tree_route(){let mut t=Tree::new();t.insert((1,"R".into()),Node::Split{n:256,f:0,t:0.});for p in ["RL","RR"]{t.insert((1,p.into()),Node::Leaf{n:128,v:vec![0.;99]});}let r=Row{t:0,h:1,y:0.,c:0.,d:0.};assert_eq!(route(&t,&r),vec!["R","RL"]);}
}
