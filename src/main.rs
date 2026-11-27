use aevral::Mlp;
fn main() {
    let args:Vec<_>=std::env::args().collect();
    if args.get(1).map(String::as_str)==Some("forest"){
        if args.len()!=7{eprintln!("Usage: aevral forest DATA_DIR FOLD OUTPUT.csv CONFIG SEED");std::process::exit(2)}
        let seed=args[6].parse::<u64>().unwrap_or_else(|_|{eprintln!("invalid seed");std::process::exit(2)});
        if let Err(e)=aevral::features::tournament::forest::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5],seed){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("upstream-smoke"){
        if args.len()!=3{eprintln!("Usage: aevral upstream-smoke BACKEND");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::upstream::smoke(&args[2]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("upstream"){
        if args.len()!=7{eprintln!("Usage: aevral upstream DATA_DIR FOLD OUTPUT.csv CONFIG SEED");std::process::exit(2)}
        let seed=args[6].parse::<u64>().unwrap_or_else(|_|{eprintln!("invalid seed");std::process::exit(2)});
        if let Err(e)=aevral::features::tournament::upstream::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5],seed){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("boosting"){
        if args.len()!=6{eprintln!("Usage: aevral boosting DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::boosting::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("svr-smoke"){
        if args.len()!=2{eprintln!("Usage: aevral svr-smoke");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::svr::smoke(){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("svr"){
        if args.len()!=6{eprintln!("Usage: aevral svr DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::svr::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("kfeat"){
        if args.len()!=7{eprintln!("Usage: aevral kfeat DATA_DIR FOLD OUTPUT.csv CONFIG SEED");std::process::exit(2)}
        let seed=args[6].parse::<u64>().unwrap_or(0);
        if let Err(e)=aevral::features::tournament::kernel_features::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5],seed){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("gp"){
        if args.len()!=6{eprintln!("Usage: aevral gp DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::gp::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("krr"){
        if args.len()!=6{eprintln!("Usage: aevral krr DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::kernel_ridge::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("local"){
        if args.len()!=6{eprintln!("Usage: aevral local DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::local::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("neighbors"){
        if args.len()!=6{eprintln!("Usage: aevral neighbors DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::neighbors::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("cart"){
        if args.len()!=6{eprintln!("Usage: aevral cart DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::cart::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("switching"){
        if args.len()!=6{eprintln!("Usage: aevral switching DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::switching::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("kalman"){
        if args.len()!=6{eprintln!("Usage: aevral kalman DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::kalman::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("smoothing"){
        if args.len()!=6{eprintln!("Usage: aevral smoothing DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::smoothing::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("arima"){
        if args.len()!=6{eprintln!("Usage: aevral arima DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::arima::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("autoreg"){
        if args.len()!=6{eprintln!("Usage: aevral autoreg DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::autoreg::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("mars"){
        if args.len()!=6{eprintln!("Usage: aevral mars DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::mars::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("basis"){
        if args.len()!=6{eprintln!("Usage: aevral basis DATA_DIR FOLD OUTPUT.csv FAMILY[:LAMBDA]");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::basis::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("latent"){
        if args.len()!=6{eprintln!("Usage: aevral latent DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::latent::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("robust"){
        if args.len()!=6{eprintln!("Usage: aevral robust DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::robust::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("sparse"){
        if args.len()!=6{eprintln!("Usage: aevral sparse DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::sparse::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("linear"){
        if args.len()!=6{eprintln!("Usage: aevral linear DATA_DIR FOLD OUTPUT.csv CONFIG");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::linear::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4]),&args[5]){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if args.get(1).map(String::as_str)==Some("tournament"){
        if args.len()!=5{eprintln!("Usage: aevral tournament DATA_DIR FOLD OUTPUT.csv");std::process::exit(2)}
        if let Err(e)=aevral::features::tournament::run(std::path::Path::new(&args[2]),&args[3],std::path::Path::new(&args[4])){eprintln!("{e}");std::process::exit(1)}
        return
    }
    if matches!(args.get(1).map(String::as_str),Some("audit"|"validate"|"windows"|"calendar"|"covariates"|"features"|"locked-test"|"architectures"|"rnn"|"gru"|"lstm"|"cnn"|"tcn"|"attention"|"architecture-budget"|"architecture-adam"|"holdout-audit"|"holdout-score")){
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
    if (matches!(mode,"architecture-budget"|"architecture-adam")&&args.len()!=5)||(!matches!(mode,"architecture-budget"|"architecture-adam")&&args.len()!=3){return Err("Usage: aevral MODE FIRST_YEAR LAST_YEAR DATA_DIR [MODEL SEED]".into())}
    let first=args[0].parse::<i64>().map_err(|e|e.to_string())?;
    let last=args[1].parse::<i64>().map_err(|e|e.to_string())?;
    if first<2000||last>2026||first>last{return Err("invalid year range".into())}
    if matches!(mode,"holdout-audit"|"holdout-score") && (first!=2000||last!=2026){return Err("holdout requires 2000–2026 archive".into())}
    if mode=="locked-test" && (first!=2000||last!=2025){return Err("locked test requires exact 2000–2025 archive".into())}
    if mode!="audit" && mode!="locked-test" && !matches!(mode,"holdout-audit"|"holdout-score") && last>2022{return Err("validation must not open locked test years".into())}
    if last==2026&&!matches!(mode,"holdout-audit"|"holdout-score"){return Err("2026 reserved for holdout".into())}
    let holdout=matches!(mode,"holdout-audit"|"holdout-score");
    let quarantine=mode=="locked-test"||holdout;
    let mut a=Archive::default();
    let mut cov=aevral::covariates::Archive::default();
    for y in first..=last{
        let p=std::path::Path::new(&args[2]).join(format!("SEA-{y}.csv"));
        let f=std::fs::File::open(&p).map_err(|e|format!("{}: {e}",p.display()))?;
        a.ingest_policy(BufReader::new(f),quarantine).map_err(|e|format!("{}: {e}",p.display()))?;
        if matches!(mode,"covariates"|"features"|"locked-test"|"architectures"|"rnn"|"gru"|"lstm"|"cnn"|"tcn"|"attention"|"architecture-budget"|"architecture-adam"|"holdout-audit"|"holdout-score"){cov.ingest_policy(BufReader::new(std::fs::File::open(&p).map_err(|e|e.to_string())?),quarantine).map_err(|e|format!("{}: {e}",p.display()))?;}
    }
    if a.is_empty(){return Err("empty archive".into())}
    let start=minute(&format!("{first}-01-01 00:00"))?+60;
    let end=if holdout{minute("2026-11-01 00:00")?-60}else{minute(&format!("{}-01-01 00:00",last+1))?-60};
    if quarantine{
        let conflicts:Vec<_>=a.conflicts.union(&cov.conflicts).copied().collect();
        println!("quarantined_test_timestamps={} utc_minutes={conflicts:?}",conflicts.len());
        for t in conflicts{a.quarantine(t);cov.quarantine(t);}
    }
    let bins=a.hourly(start,end);
    if holdout{let c=cov.hourly(start,end);aevral::features::holdout_audit(&c,start,&bins)?;if mode=="holdout-score"{return aevral::features::holdout_score(&c,start)}return Ok(())}
    if mode=="locked-test"{let c=cov.hourly(start,end);aevral::covariates::audit(&c,start,&bins)?;return aevral::features::locked_test(&c,start)}
    if matches!(mode,"architecture-budget"|"architecture-adam"){let c=cov.hourly(start,end);aevral::covariates::audit(&c,start,&bins)?;let seed=args[4].parse::<u64>().map_err(|e|e.to_string())?;return aevral::features::architecture_budget(&c,start,&args[3],seed,mode=="architecture-adam")}
    if mode=="attention"{let c=cov.hourly(start,end);aevral::covariates::audit(&c,start,&bins)?;return aevral::features::attention_screen(&c,start,mode)}
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
