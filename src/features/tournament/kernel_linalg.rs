//! Checked dense Cholesky for bounded C007 kernel designs. Row-major lower triangular factor.
pub(super) struct Cholesky{n:usize,l:Vec<f64>}
impl Cholesky{
    pub(super) fn factor(a:&[f64],n:usize)->Result<Self,String>{
        if n==0||n.checked_mul(n)!=Some(a.len())||a.iter().any(|v|!v.is_finite()){return Err("invalid Cholesky matrix".into())}
        let mut l=vec![0.;a.len()];
        for i in 0..n{for j in 0..=i{
            if (a[i*n+j]-a[j*n+i]).abs()>1e-10*(1.+a[i*n+j].abs().max(a[j*n+i].abs())){return Err("asymmetric Cholesky matrix".into())}
            let mut v=a[i*n+j];for k in 0..j{v-=l[i*n+k]*l[j*n+k];}
            if i==j{if !v.is_finite()||v<=0.{return Err(format!("nonpositive Cholesky pivot {i}: {v}"))}l[i*n+i]=v.sqrt();}
            else{l[i*n+j]=v/l[j*n+j];}
        }}
        if l.iter().any(|v|!v.is_finite()){return Err("nonfinite Cholesky factor".into())}
        Ok(Self{n,l})
    }
    pub(super) fn forward(&self,b:&[f64])->Result<Vec<f64>,String>{
        if b.len()!=self.n||b.iter().any(|v|!v.is_finite()){return Err("invalid Cholesky RHS".into())}
        let mut x=b.to_vec();for i in 0..self.n{let mut v=x[i];for j in 0..i{v-=self.l[i*self.n+j]*x[j];}x[i]=v/self.l[i*self.n+i];}
        if x.iter().any(|v|!v.is_finite()){return Err("nonfinite forward solve".into())}Ok(x)
    }
    pub(super) fn solve(&self,b:&[f64])->Result<Vec<f64>,String>{
        let mut x=self.forward(b)?;for i in (0..self.n).rev(){let mut v=x[i];for j in i+1..self.n{v-=self.l[j*self.n+i]*x[j];}x[i]=v/self.l[i*self.n+i];}
        if x.iter().any(|v|!v.is_finite()){return Err("nonfinite Cholesky solve".into())}Ok(x)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn exact_residual(){let a=[4.,2.,2.,3.];let c=Cholesky::factor(&a,2).unwrap();let x=c.solve(&[8.,7.]).unwrap();assert!((x[0]-1.25).abs()<1e-12);assert!((x[1]-1.5).abs()<1e-12);for i in 0..2{assert!((a[i*2]*x[0]+a[i*2+1]*x[1]-[8.,7.][i]).abs()<1e-12);}}
    #[test]fn bad_matrices(){assert!(Cholesky::factor(&[1.,2.,2.,1.],2).is_err());assert!(Cholesky::factor(&[1.,0.,1.,2.],2).is_err());assert!(Cholesky::factor(&[f64::NAN],1).is_err());assert!(Cholesky::factor(&[0.],1).is_err());}
}
