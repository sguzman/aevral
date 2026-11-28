//! Join only audited seed42 prediction exports, in frozen C013 member order.
//! Four-member joins remain usable when any of S/A/H/X is unavailable.
use std::{fs::File,io::{BufRead,BufReader,BufWriter,Write},path::Path};
const TARGET:&str="origin,lead,target_c";
const PRED:&str="fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c";
const MEMBERS:[(&str,&str);8]=[("I","mlp64"),("L","O04-04"),("R","O05-02"),("M","O01-06"),("S","O02-06"),("A","O06-06"),("H","O08-06"),("X","O07-06")];
fn finite(s:&str)->Result<f64,String>{let x=s.parse::<f64>().map_err(|_|"bad number")?;if !x.is_finite(){return Err("nonfinite number".into())}Ok(x)}
fn next_line(it:&mut impl Iterator<Item=std::io::Result<String>>)->Result<String,String>{it.next().ok_or("short input")?.map_err(|e|e.to_string())}
fn join(target:impl BufRead, inputs:Vec<impl BufRead>,year:&str,count:usize,k:usize,mut out:impl Write)->Result<(),String>{
    if !matches!(k,4|8)||inputs.len()!=k+1{return Err("wrong member count".into())}
    let mut ts=target.lines();let mut streams:Vec<_>=inputs.into_iter().map(|r|r.lines()).collect();
    if next_line(&mut ts)?!=TARGET{return Err("bad target header".into())}
    for s in &mut streams{if next_line(s)?!=PRED{return Err("bad prediction header".into())}}
    write!(out,"fold,origin,lead,target,daily").map_err(|e|e.to_string())?;
    for (id,_) in &MEMBERS[..k]{write!(out,",{id}").map_err(|e|e.to_string())?;}
    writeln!(out).map_err(|e|e.to_string())?;
    let mut prior=None;let mut n=0;
    for line in ts {
        let l=line.map_err(|e|e.to_string())?;let p:Vec<_>=l.split(',').collect();
        if p.len()!=3{return Err("invalid target row".into())}
        let origin=p[0].parse::<i64>().map_err(|_|"bad origin")?;
        let lead=p[1].parse::<usize>().map_err(|_|"bad lead")?;
        if lead!=n%24+1{return Err("bad lead order".into())}
        if lead==1{if prior.is_some_and(|o|origin<=o){return Err("unordered origin".into())}prior=Some(origin);}
        else if prior!=Some(origin){return Err("origin changed mid-trajectory".into())}
        let y=finite(p[2])?;
        write!(out,"{year},{origin},{lead},{y}").map_err(|e|e.to_string())?;
        for (i,s) in streams.iter_mut().enumerate(){
            let l=next_line(s)?;let v:Vec<_>=l.split(',').collect();
            let label=if i==0{"daily"}else{MEMBERS[i-1].1};
            if v.len()!=6||v[0]!=year||v[1]!=label||v[2]!=p[0]||v[3]!=p[1]||finite(v[4])?!=y{return Err(format!("member alignment mismatch {label}"))}
            write!(out,",{}",finite(v[5])?).map_err(|e|e.to_string())?;
        }
        writeln!(out).map_err(|e|e.to_string())?;n+=1;
    }
    if n!=count*24||n==0||streams.iter_mut().any(|s|s.next().is_some()){return Err("row count mismatch".into())}
    out.flush().map_err(|e|e.to_string())
}
fn open(path:&Path)->Result<BufReader<File>,String>{File::open(path).map(BufReader::new).map_err(|e|format!("{}: {e}",path.display()))}
fn run()->Result<(),String>{
    let a:Vec<_>=std::env::args().collect();
    if a.len()!=7{return Err("usage: join YEAR four|eight TARGET INPUT_DIR COUNT OUTPUT".into())}
    let k=match a[2].as_str(){"four"=>4,"eight"=>8,_=>return Err("invalid member group".into())};
    if !["2023","2024","2025","2026"].contains(&a[1].as_str()){return Err("outside C015".into())}
    let dir=Path::new(&a[4]);
    let mut inputs=vec![open(&dir.join("daily.csv"))?];
    for (id,_) in &MEMBERS[..k]{inputs.push(open(&dir.join(format!("{id}.csv")))?);}
    let count=a[5].parse::<usize>().map_err(|_|"bad count")?;
    join(open(Path::new(&a[3]))?,inputs,&a[1],count,k,BufWriter::new(File::create(&a[6]).map_err(|e|e.to_string())?))?;
    println!("c015-joined,{},{k},{count},{},pass",a[1],count*24);Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("{e}");std::process::exit(1)}}
#[cfg(test)]
mod tests{
    use super::*;
    fn fixture(k:usize)->(String,Vec<String>){
        let mut t=format!("{TARGET}\n");
        let mut v=vec![format!("{PRED}\n");k+1];
        for h in 1..=24{
            t+=&format!("12097380,{h},3\n");
            for (i,s) in v.iter_mut().enumerate(){let label=if i==0{"daily"}else{MEMBERS[i-1].1};*s+=&format!("2023,{label},12097380,{h},3,{}\n",i+1);}
        }(t,v)
    }
    #[test]fn joins_both_groups(){for k in [4,8]{let(t,v)=fixture(k);let mut out=Vec::new();join(t.as_bytes(),v.iter().map(|s|s.as_bytes()).collect(),"2023",1,k,&mut out).unwrap();let text=String::from_utf8(out).unwrap();assert_eq!(text.lines().count(),25);assert_eq!(text.lines().nth(1).unwrap().split(',').count(),5+k);}}
    #[test]fn rejects_corruption(){let(t,v)=fixture(4);for bad in [v[2].replace("O04-04","wrong"),v[2].replace(",3,3\n",",4,3\n"),v[2].replace(",3,3\n",",3,NaN\n"),v[2].replace("12097380","12097440")]{let mut v=v.clone();v[2]=bad;assert!(join(t.as_bytes(),v.iter().map(|s|s.as_bytes()).collect(),"2023",1,4,Vec::new()).is_err());}}
}
