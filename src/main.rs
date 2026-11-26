use aevral::Mlp;
fn main() {
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
