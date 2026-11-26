//! Arbitrary-depth dense regressors, including the no-hidden-layer linear case.
#[derive(Clone,Debug)]
pub struct Dense {dims:Vec<usize>,weights:Vec<Vec<f64>>,biases:Vec<Vec<f64>>}
impl Dense{
    pub fn new(dims:&[usize],seed:u64)->Self{
        assert!(dims.len()>=2&&dims.iter().all(|&d|d>0));
        let mut state=seed.max(1);
        let mut sample=|scale:f64|{state^=state<<13;state^=state>>7;state^=state<<17;let u=(state>>11)as f64/(1u64<<53)as f64;(2.*u-1.)*scale};
        let weights=dims.windows(2).map(|d|{
            let scale=(6./(d[0]+d[1])as f64).sqrt();
            (0..d[0]*d[1]).map(|_|sample(scale)).collect()
        }).collect();
        let biases=dims[1..].iter().map(|&d|vec![0.;d]).collect();
        Self{dims:dims.to_vec(),weights,biases}
    }
    pub fn parameters(&self)->usize{self.weights.iter().map(Vec::len).sum::<usize>()+self.biases.iter().map(Vec::len).sum::<usize>()}
    fn activations(&self,x:&[f64])->Vec<Vec<f64>>{
        assert_eq!(x.len(),self.dims[0]);
        let mut a=vec![x.to_vec()];
        for l in 0..self.weights.len(){
            let n=self.dims[l];let prev=&a[l];
            let h=(0..self.dims[l+1]).map(|j|{
                let v=self.biases[l][j]+(0..n).map(|i|self.weights[l][j*n+i]*prev[i]).sum::<f64>();
                if l+1==self.weights.len(){v}else{v.tanh()}
            }).collect();a.push(h);
        }a
    }
    pub fn predict(&self,x:&[f64])->Vec<f64>{self.activations(x).pop().unwrap()}
    pub fn train_step(&mut self,x:&[f64],target:&[f64],lr:f64)->f64{
        assert!(lr.is_finite()&&lr>0.);
        let a=self.activations(x);let n=*self.dims.last().unwrap();assert_eq!(target.len(),n);
        let errors:Vec<_>=a.last().unwrap().iter().zip(target).map(|(p,t)|p-t).collect();
        let mut d:Vec<_>=errors.iter().map(|e|2.*e/n as f64).collect();
        for l in (0..self.weights.len()).rev(){
            let n=self.dims[l];let m=self.dims[l+1];
            let prev:Vec<f64>=if l==0{vec![]}else{(0..n).map(|i|(1.-a[l][i]*a[l][i])*(0..m).map(|j|self.weights[l][j*n+i]*d[j]).sum::<f64>()).collect()};
            for j in 0..m{for i in 0..n{self.weights[l][j*n+i]-=lr*d[j]*a[l][i]}self.biases[l][j]-=lr*d[j]}
            d=prev;
        }errors.iter().map(|e|e*e).sum::<f64>()/target.len()as f64
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn original_equivalence(){
        let mut a=Dense::new(&[4,3,2],42);let mut b=crate::Mlp::new(4,3,2,42);
        let x=[1.,2.,-1.,0.1];let y=[0.2,0.4];
        for _ in 0..10{assert_eq!(a.predict(&x),b.predict(&x));assert_eq!(a.train_step(&x,&y,0.005),b.train_step(&x,&y,0.005));}
    }
    #[test]fn gradient_all_layers(){
        for dims in [&[2,2][..],&[2,3,2,2][..]]{
            let a=Dense::new(dims,42);let x=[0.2,-0.7];let y=[0.3,-0.4];let eps=1e-6;
            let loss=|m:&Dense|m.predict(&x).iter().zip(y).map(|(a,b)|(a-b).powi(2)).sum::<f64>()/2.;
            let mut updated=a.clone();updated.train_step(&x,&y,eps);
            for l in 0..a.weights.len(){
                for i in 0..a.weights[l].len(){let(mut p,mut m)=(a.clone(),a.clone());p.weights[l][i]+=eps;m.weights[l][i]-=eps;assert!(((loss(&p)-loss(&m))/(2.*eps)-(a.weights[l][i]-updated.weights[l][i])/eps).abs()<1e-6);}
                for i in 0..a.biases[l].len(){let(mut p,mut m)=(a.clone(),a.clone());p.biases[l][i]+=eps;m.biases[l][i]-=eps;assert!(((loss(&p)-loss(&m))/(2.*eps)-(a.biases[l][i]-updated.biases[l][i])/eps).abs()<1e-6);}
            }
        }
    }
}
