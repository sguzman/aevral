//! C015 retrospective-only data adapter; no model fitting or scoring in the cohort gate.
use super::{collect, representation, Fold, PreparedFold, H, Y};
use crate::{covariates::{self, Weather}, weather::{minute, Archive}};
use std::{fs::{self, File}, io::{BufReader, BufWriter, Write}, path::Path};

type Bin = Option<Weather>;

/// Preserve the exposed-holdout conflict policy from C001/C002:
/// quarantine conflicting timestamps from 2023 onward in both archives.
/// Pre-2023 conflicting reports still fail ingestion.
pub(super) fn load_bins(data: &Path, id: &str) -> Result<(i64, Vec<Bin>, Fold), String> {
    let fold = Fold::retrospective(id)?;
    let last: i32 = id.parse::<i32>().map_err(|e| e.to_string())?;
    let mut temperature = Archive::default();
    let mut weather = covariates::Archive::default();
    for year in 2000..=last {
        let path = data.join(format!("SEA-{year}.csv"));
        let open = || File::open(&path).map(BufReader::new)
            .map_err(|e| format!("{}: {e}", path.display()));
        temperature.ingest_policy(open()?, true)?;
        weather.ingest_policy(open()?, true)?;
    }
    let conflicts: Vec<_> = temperature.conflicts.union(&weather.conflicts).copied().collect();
    for &t in &conflicts {
        temperature.quarantine(t);
        weather.quarantine(t);
    }
    println!("retrospective-quarantine,{id},{}", conflicts.len());
    let first = minute("2000-01-01 00:00")? + 60;
    let reference = temperature.hourly(first, fold.end - 60);
    let bins = weather.hourly(first, fold.end - 60);
    covariates::audit(&bins, first, &reference)?;
    Ok((first, bins, fold))
}

fn complete(bins: &[Bin], i: usize, history: usize) -> bool {
    i + 1 >= history && i + Y < bins.len()
        && bins[i + 1 - history..=i + Y].iter().all(|b|
            b.and_then(|w| w.temperature).is_some_and(f64::is_finite))
}
fn weather_values(b: Bin) -> Option<[f64; 4]> {
    let w = b?;
    let [u, v] = w.wind_ms?;
    Some([w.dew_c?, w.pressure_hpa?, u, v])
}
#[derive(Debug)]
struct Audit {
    train: Vec<usize>,
    matched: Vec<usize>,
    natural: Vec<usize>,
    possible: usize,
    temperature_complete: usize,
}
/// Independent eligibility enumeration: classify by the entire target interval,
/// do not call Fold::partition, collect, or the representation builders.
fn enumerate(bins: &[Bin], first: i64, fold: Fold) -> Audit {
    let mut a = Audit { train: vec![], matched: vec![], natural: vec![], possible: 0, temperature_complete: 0 };
    for i in 0..bins.len().saturating_sub(Y) {
        let origin = first + i as i64 * 60;
        let assessment = origin + 60 >= fold.start && origin + 24 * 60 < fold.end;
        let training = origin + 24 * 60 < fold.start;
        if !assessment && !training { continue; }
        let met = weather_values(bins[i]).is_some_and(|x| x.iter().all(|v| v.is_finite()));
        if assessment && complete(bins, i, 24) && met { a.natural.push(i); }
        if i + 1 < H { continue; }
        if assessment { a.possible += 1; }
        if !complete(bins, i, H) { continue; }
        if assessment { a.temperature_complete += 1; }
        if !met { continue; }
        if training { a.train.push(i); } else { a.matched.push(i); }
    }
    a
}
fn values<'a>(bins: &'a [Bin], indices: &'a [usize], col: usize) -> impl Iterator<Item=f64> + Clone + 'a {
    // Distinct column walks reproduce population scale semantics without sharing Scale::fit.
    indices.iter().flat_map(move |&i| {
        let n = if col == 0 { H } else { 1 };
        (0..n).map(move |j| if col == 0 {
            bins[i + 1 - H + j].unwrap().temperature.unwrap()
        } else { weather_values(bins[i]).unwrap()[col-1] })
    })
}
fn audit_scales(bins: &[Bin], a: &Audit, f: &PreparedFold) -> Result<(), String> {
    for col in 0..5 {
        let v = values(bins, &a.train, col);
        let n = v.clone().count() as f64;
        let mean = v.clone().sum::<f64>() / n;
        let sd = (v.map(|x| (x-mean).powi(2)).sum::<f64>() / n).sqrt();
        if !mean.is_finite() || !sd.is_finite() || sd <= 0.
            || (mean-f.scales[2*col]).abs() > 1e-10
            || (sd-f.scales[2*col+1]).abs() > 1e-10 {
            return Err(format!("independent scale mismatch column {col}"));
        }
    }
    Ok(())
}
fn verify(bins: &[Bin], first: i64, f: &PreparedFold) -> Result<Audit, String> {
    let a = enumerate(bins, first, f.fold);
    if (a.train.len(), a.matched.len(), a.possible, a.temperature_complete)
        != (f.train.len(), f.eval.len(), f.possible, f.temperature_complete) {
        return Err("independent cohort counts mismatch".into());
    }
    let target = |i: usize| -> [f64; Y] {
        std::array::from_fn(|j| bins[i+j+1].unwrap().temperature.unwrap())
    };
    for (&i, r) in a.train.iter().zip(&f.train) {
        if r.input.origin != first+i as i64*60 || r.target != target(i) {
            return Err("independent training identity mismatch".into());
        }
    }
    for (&i, r) in a.matched.iter().zip(&f.eval) {
        if r.input.origin != first+i as i64*60 || r.target != target(i) {
            return Err("independent assessment identity mismatch".into());
        }
    }
    if a.matched.iter().any(|i| a.natural.binary_search(i).is_err()) {
        return Err("suffix natural coverage omits matched origins".into());
    }
    audit_scales(bins, &a, f)?;
    // Also check the existing separately implemented feature builder for exact input parity.
    representation::verify(f, bins, first)?;
    Ok(a)
}
fn subset(n: usize, budget: usize) -> Result<Vec<usize>, String> {
    if n < budget || budget == 0 { return Err("insufficient fit cohort".into()); }
    (0..budget).map(|i| i.checked_mul(n).map(|x| x/budget).ok_or("subset overflow".into())).collect()
}
fn pressure_mask(bins: &[Bin], i: usize) -> [bool; 3] {
    [3usize, 6, 12].map(|lag| i.checked_sub(lag).and_then(|k| bins[k])
        .and_then(|w| w.pressure_hpa).is_some())
}
fn write_file(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    write(&mut w).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())
}
pub fn cohort(data: &Path, id: &str, out: &Path) -> Result<(), String> {
    let (first, bins, fold) = load_bins(data, id)?;
    let f = collect(&bins, first, fold)?;
    let a = verify(&bins, first, &f)?;
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let origin = |i: usize| first+i as i64*60;
    for (name, ix) in [("train", &a.train), ("matched", &a.matched), ("natural", &a.natural)] {
        representation::origins(&out.join(format!("{name}-origins.txt")), ix.iter().map(|&i| origin(i)))?;
    }
    for budget in [8192, 24576, 32768] {
        let ix = subset(a.train.len(), budget)?;
        representation::origins(&out.join(format!("fit-{budget}-origins.txt")), ix.iter().map(|&i| origin(a.train[i])))?;
    }
    write_file(&out.join("matched-targets.csv"), |w| {
        writeln!(w, "origin,lead,target_c")?;
        for r in &f.eval { for j in 0..Y { writeln!(w, "{},{},{}", r.input.origin, j+1, r.target[j])?; } }
        Ok(())
    })?;
    write_file(&out.join("natural-targets.csv"), |w| {
        writeln!(w, "origin,lead,target_c")?;
        for &i in &a.natural { for j in 0..Y { writeln!(w, "{},{},{}", origin(i), j+1, bins[i+j+1].unwrap().temperature.unwrap())?; } }
        Ok(())
    })?;
    write_file(&out.join("pressure-masks.csv"), |w| {
        writeln!(w, "cohort,origin,lag3,lag6,lag12")?;
        for (label, ix) in [("train", &a.train), ("matched", &a.matched)] {
            for &i in ix {
                let [x,y,z] = pressure_mask(&bins,i).map(u8::from);
                writeln!(w, "{label},{},{x},{y},{z}", origin(i))?;
            }
        }
        Ok(())
    })?;
    let row = format!("{id},{},{},{},{},{},{},{}", fold.start,fold.end,a.train.len(),a.matched.len(),a.natural.len(),a.possible,a.temperature_complete);
    write_file(&out.join("cohort.csv"), |w| {
        writeln!(w, "fold,start,end,train,matched,natural,possible,temperature_complete")?;
        writeln!(w, "{row}")
    })?;
    write_file(&out.join("scales.csv"), |w| {
        writeln!(w, "column,mean,sd")?;
        for (i,name) in ["temperature","dew","pressure","wind_u","wind_v"].iter().enumerate() {
            writeln!(w, "{name},{:.17e},{:.17e}", f.scales[2*i],f.scales[2*i+1])?;
        }
        Ok(())
    })?;
    println!("retrospective-cohort,{row}");
    println!("retrospective-cohort-audit,{id},pass");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weather::Split;
    fn fixture() -> (Fold, i64, Vec<Bin>) {
        let f = Fold::retrospective("2024").unwrap();
        let first = f.start-200*60;
        let b = (0..400).map(|i| Some(Weather {
            minute:first+i*60,temperature:Some((i%31) as f64),dew_c:Some((i%17) as f64),
            pressure_hpa:Some(1000.+(i%13) as f64),wind_ms:Some([(i%11) as f64,(i%7) as f64]),
        })).collect();
        (f,first,b)
    }
    #[test] fn explicit_folds_and_boundaries() {
        for id in ["2023","2024","2025","2026"] {
            let f=Fold::retrospective(id).unwrap();
            assert!(Fold::development(id).is_err());
            assert_eq!(f.partition(f.start-1500),Some(Split::Train));
            assert_eq!(f.partition(f.start-1440),None);
            assert_eq!(f.partition(f.start-60),Some(Split::Validation));
            assert_eq!(f.partition(f.end-1500),Some(Split::Validation));
            assert_eq!(f.partition(f.end-1440),None);
        }
        assert!(Fold::retrospective("2022").is_err());
        let leap=Fold::retrospective("2024").unwrap();
        assert_eq!(leap.end-leap.start,366*1440);
        let partial=Fold::retrospective("2026").unwrap();
        assert_eq!(partial.end,minute("2026-11-01 00:00").unwrap());
    }
    #[test] fn independent_cohort_and_future_perturbation() {
        let (f,first,mut b)=fixture();
        let x=collect(&b,first,f).unwrap();verify(&b,first,&x).unwrap();
        for v in &mut b[200..] { v.as_mut().unwrap().temperature=Some(99.); }
        let y=collect(&b,first,f).unwrap();verify(&b,first,&y).unwrap();
        assert_eq!(x.scales,y.scales);
        assert!(x.train.iter().zip(&y.train).all(|(a,b)|a.input.x==b.input.x&&a.target==b.target));
        assert_ne!(x.eval[0].target,y.eval[0].target);
    }
    #[test] fn natural_and_missing_pressure() {
        let (f,first,mut b)=fixture();
        b[200].as_mut().unwrap().temperature=None;
        b[280].as_mut().unwrap().pressure_hpa=None;
        let x=collect(&b,first,f).unwrap();
        let a=verify(&b,first,&x).unwrap();
        assert!(a.natural.len()>a.matched.len());
        assert_eq!(pressure_mask(&b,283),[false,true,true]);
        assert_eq!(subset(10,4).unwrap(),vec![0,2,5,7]);
        assert!(subset(3,4).is_err());
    }
}
