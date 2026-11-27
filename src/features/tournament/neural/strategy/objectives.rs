//! Fixed C010 loss definitions, output derivatives in normalized units.
#[derive(Clone,Copy,Debug)]
pub(super) enum Objective{Mse,Mae,Huber(f64),Median,Quantiles,Early,Late,AuxMean,AuxChange(f64),Bound(f64,f64),Slope,Curvature}
impl Objective{
    pub(super) fn value_grad(self,p:&[f64],t:&[f64])->(f64,Vec<f64>){
        if matches!(self,Self::Early|Self::Late|Self::AuxMean|Self::AuxChange(_)|Self::Bound(..)|Self::Slope|Self::Curvature){return self.regularized(p,t)}
        let n=p.len();assert!(n>0);
        assert_eq!(n,if matches!(self,Self::Quantiles){3*t.len()}else{t.len()});
        let mut loss=0.;let mut grad=Vec::with_capacity(n);
        for(i,&v)in p.iter().enumerate(){let e=v-t[i%t.len()];let sign=if e>0.{1.}else if e<0.{-1.}else{0.};
            let(l,g)=match self{
                Self::Mse=>(e*e,2.*e),
                Self::Mae=>(e.abs(),sign),
                Self::Huber(delta)=>{assert!(delta.is_finite()&&delta>0.);if e.abs()<=delta{(0.5*e*e,e)}else{(delta*(e.abs()-0.5*delta),delta*sign)}},
                Self::Median|Self::Quantiles=>{let tau=if matches!(self,Self::Median){0.5}else{[0.1,0.5,0.9][i/t.len()]};if e>0.{((1.-tau)*e,1.-tau)}else if e<0.{(-tau*e,-tau)}else{(0.,0.)}}
                _=>unreachable!()
            };loss+=l;grad.push(g/n as f64);
        }(loss/n as f64,grad)
    }
    fn regularized(self,p:&[f64],t:&[f64])->(f64,Vec<f64>){
        let n=t.len();assert_eq!(n,24);
        let aux=match self{Self::AuxMean=>1,Self::AuxChange(_)=>2,_=>0};assert_eq!(p.len(),n+aux);
        let e:Vec<_>=p[..n].iter().zip(t).map(|(a,b)|a-b).collect();
        let mut g=vec![0.;p.len()];let mut v=0.;
        for i in 0..n{
            let w=match self{Self::Early=>(n-i)as f64/300.,Self::Late=>(i+1)as f64/300.,_=>1./n as f64};
            v+=w*e[i]*e[i];g[i]=2.*w*e[i];
        }
        match self{
            Self::AuxMean|Self::AuxChange(_)=>{
                let mean=t.iter().sum::<f64>()/n as f64;
                let err=p[n]-mean;let w=0.25/aux as f64;v+=w*err*err;g[n]=2.*w*err;
                if let Self::AuxChange(z0)=self{let err=p[n+1]-(t[n-1]-z0);v+=w*err*err;g[n+1]=2.*w*err;}
            }
            Self::Bound(lo,hi)=>{assert!(lo.is_finite()&&hi.is_finite()&&lo<hi);for i in 0..n{let d=p[i]-p[i].clamp(lo,hi);v+=d*d/n as f64;g[i]+=2.*d/n as f64;}}
            Self::Slope=>{let w=0.25/(n-1)as f64;for i in 1..n{let d=e[i]-e[i-1];v+=w*d*d;g[i]+=2.*w*d;g[i-1]-=2.*w*d;}}
            Self::Curvature=>{let w=0.25/(n-2)as f64;for i in 2..n{let d=e[i]-2.*e[i-1]+e[i-2];v+=w*d*d;g[i]+=2.*w*d;g[i-1]-=4.*w*d;g[i-2]+=2.*w*d;}}
            Self::Early|Self::Late=>{},
            _=>unreachable!()
        }(v,g)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn derivatives_and_edges(){for o in [Objective::Mse,Objective::Mae,Objective::Huber(0.4),Objective::Median,Objective::Quantiles]{let t=[0.2,-0.3];let p=if matches!(o,Objective::Quantiles){vec![-0.7,-0.8,0.3,0.2,0.7,0.9]}else{vec![0.3,-0.8]};let(_,g)=o.value_grad(&p,&t);for i in 0..p.len(){let(mut a,mut b)=(p.clone(),p.clone());a[i]+=1e-6;b[i]-=1e-6;let d=(o.value_grad(&a,&t).0-o.value_grad(&b,&t).0)/2e-6;assert!((d-g[i]).abs()<1e-8,"{o:?} {i}");}}
        for o in [Objective::Mae,Objective::Huber(0.4),Objective::Median]{assert_eq!(o.value_grad(&[1.],&[1.]),(0.,vec![0.]));}
        assert_eq!(Objective::Mae.value_grad(&[2.],&[0.]),(2.,vec![1.]));assert_eq!(Objective::Median.value_grad(&[2.],&[0.]),(1.,vec![0.5]));assert!((Objective::Huber(1.).value_grad(&[2.],&[0.]).0-1.5).abs()<1e-12);
    }
    #[test]fn regularized_gradients_and_reference(){
        let t:Vec<_>=(0..24).map(|i|(i as f64*0.3).sin()).collect();
        for o in [Objective::Early,Objective::Late,Objective::AuxMean,Objective::AuxChange(0.7),Objective::Bound(-0.5,0.5),Objective::Slope,Objective::Curvature]{
            let extra=match o{Objective::AuxMean=>1,Objective::AuxChange(_)=>2,_=>0};
            let p:Vec<_>=(0..24+extra).map(|i|0.8+(i as f64*0.23).cos()).collect();let(_,g)=o.value_grad(&p,&t);
            for i in 0..p.len(){let(mut a,mut b)=(p.clone(),p.clone());a[i]+=1e-6;b[i]-=1e-6;let d=(o.value_grad(&a,&t).0-o.value_grad(&b,&t).0)/2e-6;assert!((d-g[i]).abs()<1e-7,"{o:?} {i}");}
        }
        let t=[0.;24];let mut p=t;p[0]=1.;assert!((Objective::Early.value_grad(&p,&t).0-24./300.).abs()<1e-12);assert!((Objective::Late.value_grad(&p,&t).0-1./300.).abs()<1e-12);
        assert_eq!(Objective::Bound(-1.,1.).value_grad(&t,&t),(0.,vec![0.;24]));
        let mut p=vec![0.;26];p[25]=0.5;assert_eq!(Objective::AuxChange(-0.5).value_grad(&p,&t),(0.,vec![0.;26]));
        assert!((Objective::Slope.value_grad(&[1.;24],&t).0-1.).abs()<1e-12);assert!((Objective::Curvature.value_grad(&[1.;24],&t).0-1.).abs()<1e-12);
    }
}
