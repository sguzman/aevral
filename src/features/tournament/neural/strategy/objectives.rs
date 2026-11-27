//! Fixed C010 loss definitions, output derivatives in normalized units.
#[derive(Clone,Copy,Debug)]
pub(super) enum Objective{Mse,Mae,Huber(f64),Median,Quantiles}
impl Objective{
    pub(super) fn value_grad(self,p:&[f64],t:&[f64])->(f64,Vec<f64>){
        let n=p.len();assert!(n>0);
        assert_eq!(n,if matches!(self,Self::Quantiles){3*t.len()}else{t.len()});
        let mut loss=0.;let mut grad=Vec::with_capacity(n);
        for(i,&v)in p.iter().enumerate(){let e=v-t[i%t.len()];let sign=if e>0.{1.}else if e<0.{-1.}else{0.};
            let(l,g)=match self{
                Self::Mse=>(e*e,2.*e),
                Self::Mae=>(e.abs(),sign),
                Self::Huber(delta)=>{assert!(delta.is_finite()&&delta>0.);if e.abs()<=delta{(0.5*e*e,e)}else{(delta*(e.abs()-0.5*delta),delta*sign)}},
                Self::Median|Self::Quantiles=>{let tau=if matches!(self,Self::Median){0.5}else{[0.1,0.5,0.9][i/t.len()]};if e>0.{((1.-tau)*e,1.-tau)}else if e<0.{(-tau*e,-tau)}else{(0.,0.)}}
            };loss+=l;grad.push(g/n as f64);
        }(loss/n as f64,grad)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn derivatives_and_edges(){for o in [Objective::Mse,Objective::Mae,Objective::Huber(0.4),Objective::Median,Objective::Quantiles]{let t=[0.2,-0.3];let p=if matches!(o,Objective::Quantiles){vec![-0.7,-0.8,0.3,0.2,0.7,0.9]}else{vec![0.3,-0.8]};let(_,g)=o.value_grad(&p,&t);for i in 0..p.len(){let(mut a,mut b)=(p.clone(),p.clone());a[i]+=1e-6;b[i]-=1e-6;let d=(o.value_grad(&a,&t).0-o.value_grad(&b,&t).0)/2e-6;assert!((d-g[i]).abs()<1e-8,"{o:?} {i}");}}
        for o in [Objective::Mae,Objective::Huber(0.4),Objective::Median]{assert_eq!(o.value_grad(&[1.],&[1.]),(0.,vec![0.]));}
        assert_eq!(Objective::Mae.value_grad(&[2.],&[0.]),(2.,vec![1.]));assert_eq!(Objective::Median.value_grad(&[2.],&[0.]),(1.,vec![0.5]));assert!((Objective::Huber(1.).value_grad(&[2.],&[0.]).0-1.5).abs()<1e-12);
    }
}
