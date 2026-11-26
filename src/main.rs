use aevral::Mlp;
fn main() {
    let args:Vec<_>=std::env::args().collect();
    if args.get(1).map(String::as_str)==Some("audit"){
        if let Err(e)=audit(&args[2..]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if std::env::args().nth(1).as_deref() != Some("smoke") {
        eprintln!("Usage: aevral smoke (synthetic implementation check only)");
        std::process::exit(2);
    }
    let mut model = Mlp::new(24,32,24,42);
    let series = |t: usize| (t as f64 * std::f64::consts::TAU / 24.0).sin();
    let example = |t: usize| {
        let x: Vec<_> = (t-23..=t).map(series).collect();
        let y: Vec<_> = (t+1..=t+24).map(series).collect();
        (x,y)
    };
    let score = |m: &Mlp| {
        (1000..1024).map(|t| {
            let (x,y) = example(t);
            m.predict(&x).iter().zip(y).map(|(a,b)| (a-b).abs()).sum::<f64>()/24.0
        }).sum::<f64>()/24.0
    };
    let before = score(&model);
    for _ in 0..20 {
        for t in 23..240 {
            let (x,y) = example(t);
            model.train_step(&x,&y,0.025);
        }
    }
    let after = score(&model);
    println!("SYNTHETIC ONLY: held-out-origin MAE before={before:.6}, after={after:.6}");
    if !after.is_finite() || after >= before { std::process::exit(1); }
}

fn audit(args:&[String])->Result<(),String>{
    use aevral::weather::{Archive,minute,examples};
    use std::io::BufReader;
    if args.len()!=3{return Err("Usage: aevral audit FIRST_YEAR LAST_YEAR DATA_DIR".into())}
    let first=args[0].parse::<i64>().map_err(|e|e.to_string())?;
    let last=args[1].parse::<i64>().map_err(|e|e.to_string())?;
    if first<2000||last>2025||first>last{return Err("invalid year range".into())}
    let mut a=Archive::default();
    for y in first..=last{
        let p=std::path::Path::new(&args[2]).join(format!("SEA-{y}.csv"));
        let f=std::fs::File::open(&p).map_err(|e|format!("{}: {e}",p.display()))?;
        a.ingest(BufReader::new(f)).map_err(|e|format!("{}: {e}",p.display()))?;
    }
    if a.is_empty(){return Err("empty archive".into())}
    let start=minute(&format!("{first}-01-01 00:00"))?+60;
    let end=minute(&format!("{}-01-01 00:00",last+1))?-60;
    let (_,report)=examples(&a.hourly(start,end),start);
    println!("KSEA UTC strict-window audit; years={first}..={last}");
    println!("unique_reports={} duplicate_rows={}",a.len(),a.duplicates);
    println!("{report:#?}");
    if report.eligible==0{return Err("no complete forecasting examples".into())}Ok(())
}
