//! Educational single-hidden-layer MLP, with direct multi-output regression.
#[derive(Clone, Debug)]
pub struct Mlp {
    inputs: usize,
    hidden: usize,
    outputs: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

impl Mlp {
    pub fn new(inputs: usize, hidden: usize, outputs: usize, seed: u64) -> Self {
        assert!(inputs > 0 && hidden > 0 && outputs > 0);
        let mut state = seed.max(1);
        let mut sample = |scale: f64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let uniform = (state >> 11) as f64 / (1u64 << 53) as f64;
            (2.0 * uniform - 1.0) * scale
        };
        let s1 = (6.0 / (inputs + hidden) as f64).sqrt();
        let s2 = (6.0 / (hidden + outputs) as f64).sqrt();
        Self {
            inputs, hidden, outputs,
            w1: (0..inputs * hidden).map(|_| sample(s1)).collect(),
            b1: vec![0.0; hidden],
            w2: (0..hidden * outputs).map(|_| sample(s2)).collect(),
            b2: vec![0.0; outputs],
        }
    }
    fn activations(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(x.len(), self.inputs);
        let h: Vec<_> = (0..self.hidden).map(|j| {
            (self.b1[j] + (0..self.inputs).map(|i| self.w1[j*self.inputs+i]*x[i]).sum::<f64>()).tanh()
        }).collect();
        let y = (0..self.outputs).map(|k| {
            self.b2[k] + (0..self.hidden).map(|j| self.w2[k*self.hidden+j]*h[j]).sum::<f64>()
        }).collect();
        (h,y)
    }
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        self.activations(x).1
    }
    /// One SGD step on mean squared error; returns pre-update loss.
    pub fn train_step(&mut self, x: &[f64], target: &[f64], lr: f64) -> f64 {
        assert_eq!(target.len(), self.outputs);
        assert!(lr.is_finite() && lr > 0.0);
        let (h,y) = self.activations(x);
        let errors: Vec<_> = y.iter().zip(target).map(|(a,b)| a-b).collect();
        let dy: Vec<_> = errors.iter().map(|e| 2.0*e/self.outputs as f64).collect();
        // Backpropagate before modifying output weights.
        let dh: Vec<_> = (0..self.hidden).map(|j| {
            (1.0-h[j]*h[j])*(0..self.outputs).map(|k| self.w2[k*self.hidden+j]*dy[k]).sum::<f64>()
        }).collect();
        for k in 0..self.outputs {
            for j in 0..self.hidden { self.w2[k*self.hidden+j] -= lr*dy[k]*h[j]; }
            self.b2[k] -= lr*dy[k];
        }
        for j in 0..self.hidden {
            for i in 0..self.inputs { self.w1[j*self.inputs+i] -= lr*dh[j]*x[i]; }
            self.b1[j] -= lr*dh[j];
        }
        errors.iter().map(|e| e*e).sum::<f64>() / self.outputs as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_and_shape() {
        let a = Mlp::new(24,32,24,42);
        let b = Mlp::new(24,32,24,42);
        assert_eq!(a.predict(&[0.5;24]), b.predict(&[0.5;24]));
        assert_eq!(a.predict(&[0.5;24]).len(),24);
    }
    #[test]
    fn learns_single_example() {
        let mut m = Mlp::new(2,8,2,42);
        let x = [0.2,-0.7]; let y = [0.3,-0.4];
        let before = m.train_step(&x,&y,0.03);
        for _ in 0..200 { m.train_step(&x,&y,0.03); }
        let after = m.train_step(&x,&y,0.03);
        assert!(after < before * 0.01);
    }
    #[test]
    fn gradient_matches_finite_difference() {
        let m = Mlp::new(2,3,2,42);
        let x = [0.2,-0.7]; let y = [0.3,-0.4];
        let loss = |n: &Mlp| n.predict(&x).iter().zip(y).map(|(a,b)| (a-b).powi(2)).sum::<f64>()/2.0;
        let eps = 1e-6;
        let mut updated = m.clone();
        updated.train_step(&x,&y,eps);
        for i in 0..m.w1.len() {
            let (mut plus,mut minus) = (m.clone(),m.clone());
            plus.w1[i] += eps; minus.w1[i] -= eps;
            let numeric = (loss(&plus)-loss(&minus))/(2.0*eps);
            let analytic = (m.w1[i]-updated.w1[i])/eps;
            assert!((numeric-analytic).abs() < 1e-6);
        }
        for i in 0..m.w2.len() {
            let (mut plus,mut minus) = (m.clone(),m.clone());
            plus.w2[i] += eps; minus.w2[i] -= eps;
            let numeric = (loss(&plus)-loss(&minus))/(2.0*eps);
            let analytic = (m.w2[i]-updated.w2[i])/eps;
            assert!((numeric-analytic).abs() < 1e-6);
        }
    }
}

pub mod weather;

pub mod benchmark;
