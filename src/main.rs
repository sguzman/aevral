use aevral::Mlp;
fn main() {
    let args:Vec<_>=std::env::args().collect();
    if matches!(args.get(1).map(String::as_str),Some("audit"|"validate"|"windows"|"calendar"|"covariates"|"features"|"locked-test"|"architectures"|"rnn"|"gru"|"lstm"|"cnn"|"tcn")){
        if let Err(e)=audit(&args[2..],&args[1]){eprintln!("{e}");std::process::exit(1)}
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

fn audit(args:&[String],mode:&str)->Result<(),String>{
    use aevral::weather::{Archive,minute,examples};
    use std::io::BufReader;
    if args.len()!=3{return Err("Usage: aevral audit FIRST_YEAR LAST_YEAR DATA_DIR".into())}
    let first=args[0].parse::<i64>().map_err(|e|e.to_string())?;
    let last=args[1].parse::<i64>().map_err(|e|e.to_string())?;
    if first<2000||last>2025||first>last{return Err("invalid year range".into())}
    if mode=="locked-test" && (first!=2000||last!=2025){return Err("locked test requires exact 2000–2025 archive".into())}
    if mode!="audit" && mode!="locked-test" && last>2022{return Err("validation must not open locked test years".into())}
    let mut a=Archive::default();
    let mut cov=aevral::covariates::Archive::default();
    for y in first..=last{
        let p=std::path::Path::new(&args[2]).join(format!("SEA-{y}.csv"));
        let f=std::fs::File::open(&p).map_err(|e|format!("{}: {e}",p.display()))?;
        a.ingest_policy(BufReader::new(f),mode=="locked-test").map_err(|e|format!("{}: {e}",p.display()))?;
        if matches!(mode,"covariates"|"features"|"locked-test"|"architectures"|"rnn"|"gru"|"lstm"|"cnn"|"tcn"){cov.ingest_policy(BufReader::new(std::fs::File::open(&p).map_err(|e|e.to_string())?),mode=="locked-test").map_err(|e|format!("{}: {e}",p.display()))?;}
    }
    if a.is_empty(){return Err("empty archive".into())}
    let start=minute(&format!("{first}-01-01 00:00"))?+60;
    let end=minute(&format!("{}-01-01 00:00",last+1))?-60;
    if mode=="locked-test"{
        let conflicts:Vec<_>=a.conflicts.union(&cov.conflicts).copied().collect();
        println!("quarantined_test_timestamps={} utc_minutes={conflicts:?}",conflicts.len());
        for t in conflicts{a.quarantine(t);cov.quarantine(t);}
    }
    let bins=a.hourly(start,end);
    if mode=="locked-test"{let c=cov.hourly(start,end);aevral::covariates::audit(&c,start,&bins)?;return aevral::features::locked_test(&c,start)}
    if matches!(mode,"cnn"|"tcn"){let c=cov.hourly(start,end);aevral::covariates::audit(&c,start,&bins)?;return aevral::features::convolution_screen(&c,start,mode)}
    if matches!(mode,"rnn"|"gru"|"lstm"){let c=cov.hourly(start,end);aevral::covariates::audit(&c,start,&bins)?;return aevral::features::recurrent_screen(&c,start,mode)}
    if mode=="architectures"{let c=cov.hourly(start,end);aevral::covariates::audit(&c,start,&bins)?;return aevral::features::architectures(&c,start)}
    if mode=="features"{let c=cov.hourly(start,end);aevral::covariates::audit(&c,start,&bins)?;return aevral::features::run(&c,start)}
    if mode=="covariates"{println!("invalid_dew_pressure_direction_speed={:?}",cov.invalid);return aevral::covariates::audit(&cov.hourly(start,end),start,&bins)}
    if mode=="calendar"{return aevral::windows::calendar(&bins,start)}
    if mode=="windows"{return aevral::windows::run(&bins,start)}
    let (rows,report)=examples(&bins,start);
    println!("KSEA UTC strict-window audit; years={first}..={last}");
    println!("unique_reports={} duplicate_rows={}",a.len(),a.duplicates);
    println!("{report:#?}");
    if report.eligible==0{return Err("no complete forecasting examples".into())}
    if mode=="validate" {aevral::benchmark::run(&rows)?;} Ok(())
}
