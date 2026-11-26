//! Bias-corrected Adam with one timestep per example, not per tensor.
pub struct Adam{m:Vec<f64>,v:Vec<f64>,b1:f64,b2:f64,cursor:usize,lr:f64}
impl Adam{
    pub fn new(n:usize,lr:f64)->Self{assert!(n>0&&lr.is_finite()&&lr>0.);Self{m:vec![0.;n],v:vec![0.;n],b1:1.,b2:1.,cursor:n,lr}}
    pub fn begin(&mut self){assert_eq!(self.cursor,self.m.len());self.cursor=0;self.b1*=0.9;self.b2*=0.999;}
    pub fn update(&mut self,w:&mut f64,g:f64){let i=self.cursor;self.cursor+=1;self.m[i]=0.9*self.m[i]+0.1*g;self.v[i]=0.999*self.v[i]+0.001*g*g;let m=self.m[i]/(1.-self.b1);let v=self.v[i]/(1.-self.b2);*w-=self.lr*m/(v.sqrt()+1e-8)}
    pub fn finish(&self){assert_eq!(self.cursor,self.m.len());}
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn constant_gradient_bias_correction(){
        let mut a=Adam::new(2,0.001);let(mut x,mut y)=(1.,2.);
        for t in 1..=10{a.begin();a.update(&mut x,2.);a.update(&mut y,-3.);a.finish();assert!((x-(1.-t as f64*0.001)).abs()<1e-10);assert!((y-(2.+t as f64*0.001)).abs()<1e-10)}
    }
    #[test]fn changing_gradient_reference(){
        let mut a=Adam::new(1,0.001);let mut w=0.;
        a.begin();a.update(&mut w,1.);a.finish();
        a.begin();a.update(&mut w,-2.);a.finish();
        let expected=-0.001/(1.+1e-8)-0.001*(-0.11/0.19)/((0.004999_f64/0.001999).sqrt()+1e-8);
        assert!((w-expected).abs()<1e-12);
    }
}
