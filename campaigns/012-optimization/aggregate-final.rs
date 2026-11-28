//! Reproduce C012 annual summary from the committed registry; no fitting.
use std::{collections::{BTreeMap, BTreeSet}, fs, path::Path};
fn num(r: &[&str], i: usize) -> f64 {
    let v: f64 = r[i].parse().unwrap();
    assert!(v.is_finite() && v >= 0.0);
    v
}
fn mean(rs: &[&[&str]], i: usize, square: bool) -> f64 {
    rs.iter().map(|r| { let v = num(r, i); if square { v*v } else { v } }).sum::<f64>() / rs.len() as f64
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "usage: aggregate-final REGISTRY FINAL_MATRIX OUTPUT_DIR");
    let source = fs::read_to_string(&args[1]).unwrap();
    let mut lines = source.lines();
    assert_eq!(lines.next().unwrap(), "rung,config,fold,seed,track,fit_budget,origins,mae_c,rmse_c,natural_origins,natural_mae_c,natural_rmse_c,fit_s,predict_s,fit_predict_s,parameters,wall_s,max_rss_kib,code_sha,ci_run,job_id,status,rescore,archive");
    let all: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
    assert_eq!(all.len(),128);
    assert!(all.iter().all(|r| r.len()==24));
    assert_eq!(all.iter().map(|r| &r[..4]).collect::<BTreeSet<_>>().len(),128);
    assert_eq!(all.iter().map(|r| r[20]).collect::<BTreeSet<_>>().len(),128);
    let matrix = fs::read_to_string(&args[2]).unwrap();
    let expected: BTreeSet<_> = matrix.lines().skip(1).map(|s| s.to_owned()).collect();
    let actual: BTreeSet<_> = all.iter().filter(|r| r[0]=="final").map(|r| r[1..4].join(",")).collect();
    assert_eq!(expected.len(),48);
    assert_eq!(actual,expected);
    let mut groups: BTreeMap<&str,Vec<&[&str]>>=BTreeMap::new();
    for r in all.iter().filter(|r| matches!(r[0],"3"|"final")) {
        assert_eq!(r[5],"32768");
        assert!(r[21]=="measured" && r[22]=="ci-rescored" || r[21]=="blocked" && r[22]=="not-applicable" && r[6..16].iter().all(|s|s.is_empty()));
        groups.entry(r[1]).or_default().push(r);
    }
    assert_eq!(groups.len(),8);
    assert_eq!(groups.values().map(Vec::len).sum::<usize>(),56);
    let mut summary=String::from("config,track,measured,planned,coverage,matched_origins,mae_c,rmse_c,natural_origins,natural_mae_c,natural_rmse_c,median_fit_predict_s,max_rss_kib\n");
    let mut folds=String::from("config,fold,measured,planned,origins,mae_c,rmse_c,min_seed_mae_c,max_seed_mae_c\n");
    for (id,rs) in groups {
        let stochastic=id.starts_with("O01-")||id.starts_with("O02-")||id.starts_with("O03-");
        let seeds: BTreeSet<&str> = if stochastic { ["42","123","2026"].into_iter().collect() } else { ["42"].into_iter().collect() };
        assert_eq!(rs.len(),4*seeds.len());
        let (mut n,mut nn,mut ae,mut se,mut nae,mut nse)=(0.,0.,0.,0.,0.,0.);
        let mut costs=Vec::new();
        let mut rss: f64=0.;
        for (year,w,nw) in [("2019",7455.,7948.),("2020",7543.,7976.),("2021",7501.,7969.),("2022",7639.,8063.)] {
            let planned: Vec<_>=rs.iter().copied().filter(|r|r[2]==year).collect();
            assert_eq!(planned.iter().map(|r|r[3]).collect::<BTreeSet<_>>(),seeds);
            assert_eq!(planned.len(),seeds.len());
            let measured: Vec<_>=planned.into_iter().filter(|r|r[21]=="measured").collect();
            assert!(!measured.is_empty());
            let natural=if id.starts_with("O02-"){nw}else{w};
            assert!(measured.iter().all(|r|num(r,6)==w&&num(r,9)==natural));
            let a=mean(&measured,7,false); let s=mean(&measured,8,true);
            let min=measured.iter().map(|r|num(r,7)).fold(f64::INFINITY,f64::min);
            let max=measured.iter().map(|r|num(r,7)).fold(0.,f64::max);
            folds+=&format!("{id},{year},{},{},{w:.0},{a:.8},{:.8},{min:.8},{max:.8}\n",measured.len(),seeds.len(),s.sqrt());
            n+=w; nn+=natural; ae+=w*a; se+=w*s;
            nae+=natural*mean(&measured,10,false); nse+=natural*mean(&measured,11,true);
            for r in measured { costs.push(num(r,14)); rss=rss.max(num(r,17)); }
        }
        costs.sort_by(f64::total_cmp);
        let k=costs.len();
        let median=(costs[(k-1)/2]+costs[k/2])/2.;
        let coverage=if k==rs.len(){"complete"}else{"available-case"};
        assert!(rs.iter().all(|r|r[4]==rs[0][4]));
        summary+=&format!("{id},{},{k},{},{coverage},{n:.0},{:.8},{:.8},{nn:.0},{:.8},{:.8},{median:.8},{rss:.0}\n",rs[0][4],rs.len(),ae/n,(se/n).sqrt(),nae/nn,(nse/nn).sqrt());
    }
    let out=Path::new(&args[3]);
    fs::create_dir_all(out).unwrap();
    fs::write(out.join("annual-summary.csv"),summary).unwrap();
    fs::write(out.join("annual-folds.csv"),folds).unwrap();
}
