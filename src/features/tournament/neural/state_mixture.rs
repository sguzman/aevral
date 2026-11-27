//! Stable trainable diagonal state space and soft mixture of experts.
use super::residual::{Affine,sample};
use crate::adam::Adam;
fn sigmoid(x:f64)->f64{1./(1.+(-x).exp())}
fn loss(y:&[f64],t:&[f64])->(f64,Vec<f64>){assert_eq!(y.len(),24);assert_eq!(t.len(),24);let e:Vec<_>=y.iter().zip(t).map(|(a,b)|a-b).collect();(e.iter().map(|v|v*v).sum::<f64>()/24.,e.iter().map(|v|v/12.).collect())}
fn norm_clip<'a>(g:impl Iterator<Item=&'a f64>)->Option<(f64,bool)>{let n=g.map(|v|v*v).sum::<f64>().sqrt();if !n.is_finite(){return None}Some((if n>5.{5./n}else{1.},n>5.))}
#[derive(Clone)]
pub(super) struct StateSpace{alpha:Vec<f64>,drive:Vec<f64>,bias:Vec<f64>,head:Affine}
impl StateSpace{
    pub(super) fn new(h:usize,seed:u64)->Self{let mut s=seed.max(1);let alpha=(0..h).map(|_|sample(&mut s,1.)).collect();let drive=(0..h).map(|_|sample(&mut s,(6./(1+h)as f64).sqrt())).collect();let head=Affine::new(h+9,24,&mut s);Self{alpha,drive,bias:vec![0.;h],head}}
    pub(super) fn parameters(&self)->usize{3*self.alpha.len()+self.head.w.len()}
    pub(super) fn finite(&self)->bool{self.alpha.iter().chain(&self.drive).chain(&self.bias).chain(&self.head.w).all(|v|v.is_finite())}
    fn forward(&self,x:&[f64])->(Vec<Vec<f64>>,Vec<f64>,Vec<f64>){
        assert_eq!(x.len(),80);let h=self.alpha.len();let a:Vec<_>=self.alpha.iter().map(|&v|sigmoid(v)).collect();let mut states=vec![vec![0.;h];73];
        for t in 0..72{for i in 0..h{states[t+1][i]=a[i]*states[t][i]+(1.-a[i])*(self.drive[i]*x[t]+self.bias[i]);}}
        let mut read=states[72].clone();read.extend_from_slice(&x[71..]);let y=self.head.forward(&read);(states,read,y)
    }
    pub(super) fn predict(&self,x:&[f64])->Vec<f64>{self.forward(x).2}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,[Vec<f64>;4]){
        let(states,read,y)=self.forward(x);let(v,dy)=loss(&y,target);let(mut dh,gh)=self.head.backward(&read,&dy);let h=self.alpha.len();let(mut ga,mut gw,mut gb)=(vec![0.;h],vec![0.;h],vec![0.;h]);
        for i in 0..h{let a=sigmoid(self.alpha[i]);for t in (0..72).rev(){let d=dh[i];let u=self.drive[i]*x[t]+self.bias[i];ga[i]+=d*(states[t][i]-u)*a*(1.-a);gw[i]+=d*(1.-a)*x[t];gb[i]+=d*(1.-a);dh[i]*=a;}}
        (v,[ga,gw,gb,gh])
    }
    pub(super) fn step(&mut self,x:&[f64],y:&[f64],opt:&mut Adam)->(f64,bool){
        let(v,g)=self.gradients(x,y);let Some((s,c))=norm_clip(g.iter().flatten())else{return(f64::NAN,false)};if !v.is_finite(){return(f64::NAN,false)}
        opt.begin();for(w,g)in [&mut self.alpha,&mut self.drive,&mut self.bias,&mut self.head.w].into_iter().zip(g){for(w,d)in w.iter_mut().zip(g){opt.update(w,s*d)}}opt.finish();(v,c)
    }
}
#[derive(Clone)]
pub(super) struct Mixture{gate:Affine,experts:Vec<[Affine;2]>}
struct Cache{p:Vec<f64>,hidden:Vec<Vec<f64>>,expert:Vec<Vec<f64>>,y:Vec<f64>}
impl Mixture{
    pub(super) fn new(e:usize,w:usize,seed:u64)->Self{
        let mut s=seed.max(1);let gate=Affine::new(80,e,&mut s);let experts=(0..e).map(|_|[Affine::new(80,w,&mut s),Affine::new(w,24,&mut s)]).collect();Self{gate,experts}
    }
    pub(super) fn parameters(&self)->usize{self.gate.w.len()+self.experts.iter().flatten().map(|l|l.w.len()).sum::<usize>()}
    pub(super) fn finite(&self)->bool{self.gate.w.iter().chain(self.experts.iter().flatten().flat_map(|l|&l.w)).all(|v|v.is_finite())}
    fn forward(&self,x:&[f64])->Cache{
        let z=self.gate.forward(x);let m=z.iter().copied().fold(f64::NEG_INFINITY,f64::max);let mut p:Vec<_>=z.iter().map(|v|(v-m).exp()).collect();let sum=p.iter().sum::<f64>();for v in &mut p{*v/=sum;}
        let mut hidden=Vec::new();let mut expert=Vec::new();let mut y=vec![0.;24];
        for(e,l)in self.experts.iter().enumerate(){let h:Vec<_>=l[0].forward(x).into_iter().map(f64::tanh).collect();let v=l[1].forward(&h);for j in 0..24{y[j]+=p[e]*v[j];}hidden.push(h);expert.push(v);}Cache{p,hidden,expert,y}
    }
    pub(super) fn predict(&self,x:&[f64])->Vec<f64>{self.forward(x).y}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,Vec<f64>,Vec<[Vec<f64>;2]>){
        let c=self.forward(x);let(v,dy)=loss(&c.y,target);
        let dp:Vec<_>=c.expert.iter().map(|y|dy.iter().zip(y).map(|(d,y)|d*y).sum::<f64>()).collect();let mean=dp.iter().zip(&c.p).map(|(d,p)|d*p).sum::<f64>();
        let dz:Vec<_>=dp.iter().zip(&c.p).map(|(d,p)|p*(d-mean)).collect();let gg=self.gate.backward(x,&dz).1;
        let mut grads=Vec::new();for(e,l)in self.experts.iter().enumerate(){let d:Vec<_>=dy.iter().map(|v|v*c.p[e]).collect();let(mut dh,g2)=l[1].backward(&c.hidden[e],&d);for(d,h)in dh.iter_mut().zip(&c.hidden[e]){*d*=1.-h*h;}let g1=l[0].backward(x,&dh).1;grads.push([g1,g2]);}(v,gg,grads)
    }
    pub(super) fn step(&mut self,x:&[f64],y:&[f64],opt:&mut Adam)->(f64,bool){
        let(v,gg,g)=self.gradients(x,y);let Some((s,c))=norm_clip(gg.iter().chain(g.iter().flatten().flatten()))else{return(f64::NAN,false)};if !v.is_finite(){return(f64::NAN,false)}
        opt.begin();for(w,d)in self.gate.w.iter_mut().zip(gg){opt.update(w,s*d)}for(e,g)in self.experts.iter_mut().zip(g){for(l,g)in e.iter_mut().zip(g){for(w,d)in l.w.iter_mut().zip(g){opt.update(w,s*d)}}}opt.finish();(v,c)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    fn fixture()->(Vec<f64>,Vec<f64>){((0..80).map(|i|(i as f64*0.13).sin()*0.2).collect(),(0..24).map(|i|0.1+(i as f64*0.25).cos()*0.05).collect())}
    #[test]fn state_space_gradients_and_stability(){let(x,y)=fixture();for h in [16,32]{let a=StateSpace::new(h,42);let(v,g)=a.gradients(&x,&y);
        let eval=|m:&StateSpace|loss(&m.predict(&x),&y).0;
        for(l,grad)in g.iter().enumerate(){for i in [0,grad.len()/2,grad.len()-1]{let(mut p,mut m)=(a.clone(),a.clone());            fn weights(a:&mut StateSpace,l:usize)->&mut Vec<f64>{match l{0=>&mut a.alpha,1=>&mut a.drive,2=>&mut a.bias,_=>&mut a.head.w}}
            weights(&mut p,l)[i]+=1e-6;weights(&mut m,l)[i]-=1e-6;assert!(((eval(&p)-eval(&m))/2e-6-grad[i]).abs()<1e-6);}}
        assert_eq!(a.predict(&x),StateSpace::new(h,42).predict(&x));assert_ne!(a.predict(&x),StateSpace::new(h,123).predict(&x));
        let(states,_,_)=a.forward(&[1.;80]);for i in 0..h{let decay=sigmoid(a.alpha[i]);assert!(decay>0.&&decay<1.);assert!((states[72][i]-(a.drive[i]+a.bias[i])*(1.-decay.powi(72))).abs()<1e-12);}
        let mut a=a;let mut opt=Adam::new(a.parameters(),0.001);for _ in 0..150{assert!(a.step(&x,&y,&mut opt).0.is_finite());}assert!(a.finite()&&eval(&a)<v*0.1);
    }}
    #[test]fn mixture_gradients_and_soft_routing(){let(x,y)=fixture();for(e,w)in [(2,64),(4,64),(4,32)]{let a=Mixture::new(e,w,42);let(v,gg,g)=a.gradients(&x,&y);let eval=|m:&Mixture|loss(&m.predict(&x),&y).0;
        for i in [0,gg.len()/2,gg.len()-1]{let(mut p,mut m)=(a.clone(),a.clone());p.gate.w[i]+=1e-6;m.gate.w[i]-=1e-6;assert!(((eval(&p)-eval(&m))/2e-6-gg[i]).abs()<1e-6);}
        for(j,grads)in g.iter().enumerate(){for(l,grad)in grads.iter().enumerate(){assert!(grad.iter().any(|x|x.abs()>1e-12));for i in [0,grad.len()/2,grad.len()-1]{let(mut p,mut m)=(a.clone(),a.clone());p.experts[j][l].w[i]+=1e-6;m.experts[j][l].w[i]-=1e-6;assert!(((eval(&p)-eval(&m))/2e-6-grad[i]).abs()<1e-6);}}}
        let c=a.forward(&x);assert!((c.p.iter().sum::<f64>()-1.).abs()<1e-12);assert!(c.p.iter().all(|v|*v>0.));for j in 0..24{let(lo,hi)=c.expert.iter().map(|v|v[j]).fold((f64::INFINITY,f64::NEG_INFINITY),|(a,b),v|(a.min(v),b.max(v)));assert!(c.y[j]>=lo-1e-12&&c.y[j]<=hi+1e-12);}
        assert_eq!(a.predict(&x),Mixture::new(e,w,42).predict(&x));assert_ne!(a.predict(&x),Mixture::new(e,w,123).predict(&x));
        let mut a=a;let mut opt=Adam::new(a.parameters(),0.001);for _ in 0..100{assert!(a.step(&x,&y,&mut opt).0.is_finite());}assert!(a.finite()&&eval(&a)<v*0.1);
    }}
}
