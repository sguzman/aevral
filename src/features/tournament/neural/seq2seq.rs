//! C009 free-running recurrent encoder-decoder; shared decoder parameters.
use super::residual::Affine;
use crate::adam::Adam;
#[derive(Clone)]
pub(super) struct Seq2Seq{h:usize,layers:[Affine;3]}
struct Cache{enc:Vec<Vec<f64>>,dec:Vec<Vec<f64>>,states:Vec<Vec<f64>>,y:Vec<f64>}
impl Seq2Seq{
    pub(super) fn new(h:usize,seed:u64)->Self{let mut s=seed.max(1);Self{h,layers:[Affine::new(6+h,h,&mut s),Affine::new(10+h,h,&mut s),Affine::new(h,1,&mut s)]}}
    pub(super) fn parameters(&self)->usize{self.layers.iter().map(|l|l.w.len()).sum()}
    pub(super) fn finite(&self)->bool{self.layers.iter().flat_map(|l|&l.w).all(|v|v.is_finite())}
    fn forward(&self,x:&[f64])->Cache{
        assert_eq!(x.len(),80);let mut enc=Vec::with_capacity(12);let mut dec=Vec::with_capacity(24);
        let mut h=vec![0.;self.h];
        for t in 0..12{let mut input=x[t*6..t*6+6].to_vec();input.extend_from_slice(&h);
            h=self.layers[0].forward(&input).into_iter().map(f64::tanh).collect();enc.push(input);}
        let mut states=Vec::with_capacity(24);let mut y=Vec::with_capacity(24);let mut previous=x[71];
        for t in 0..24{let mut input=vec![previous];input.extend_from_slice(&x[72..80]);input.push((t+1)as f64/24.);input.extend_from_slice(&h);
            h=self.layers[1].forward(&input).into_iter().map(f64::tanh).collect();previous=self.layers[2].forward(&h)[0];
            dec.push(input);states.push(h.clone());y.push(previous);}
        Cache{enc,dec,states,y}
    }
    pub(super) fn predict(&self,x:&[f64])->Vec<f64>{self.forward(x).y}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,[Vec<f64>;3]){
        assert_eq!(target.len(),24);let c=self.forward(x);let mut g=self.layers.each_ref().map(|l|vec![0.;l.w.len()]);
        let mut dh=vec![0.;self.h];let mut dy=0.;let mut loss=0.;
        fn add(a:&mut[f64],b:&[f64]){for(a,b)in a.iter_mut().zip(b){*a+=b;}}
        for t in (0..24).rev(){let err=c.y[t]-target[t];loss+=err*err/24.;dy+=2.*err/24.;
            let(d,g2)=self.layers[2].backward(&c.states[t],&[dy]);add(&mut g[2],&g2);add(&mut dh,&d);
            for(d,h)in dh.iter_mut().zip(&c.states[t]){*d*=1.-h*h;}
            let(d,g1)=self.layers[1].backward(&c.dec[t],&dh);add(&mut g[1],&g1);dy=d[0];dh=d[10..].to_vec();
        }
        for t in (0..12).rev(){let h=if t==11{&c.dec[0][10..]}else{&c.enc[t+1][6..]};
            for(d,h)in dh.iter_mut().zip(h){*d*=1.-h*h;}
            let(d,g0)=self.layers[0].backward(&c.enc[t],&dh);add(&mut g[0],&g0);dh=d[6..].to_vec();
        }(loss,g)
    }
    pub(super) fn step(&mut self,x:&[f64],y:&[f64],opt:&mut Adam)->(f64,bool){
        let(loss,g)=self.gradients(x,y);let norm=g.iter().flatten().map(|v|v*v).sum::<f64>().sqrt();
        if !loss.is_finite()||!norm.is_finite(){return(f64::NAN,false)}let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        opt.begin();for(l,g)in self.layers.iter_mut().zip(g){for(w,d)in l.w.iter_mut().zip(g){opt.update(w,d*scale)}}opt.finish();(loss,clipped)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn rollout_gradient_all_groups(){let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let y=[0.1;24];let a=Seq2Seq::new(3,42);
        let loss=|m:&Seq2Seq|m.predict(&x).iter().zip(y).map(|(p,t)|(p-t).powi(2)).sum::<f64>()/24.;
        let(_,g)=a.gradients(&x,&y);for l in 0..3{for(i,d)in g[l].iter().enumerate(){let(mut p,mut m)=(a.clone(),a.clone());p.layers[l].w[i]+=1e-6;m.layers[l].w[i]-=1e-6;assert!(((loss(&p)-loss(&m))/2e-6-d).abs()<1e-6,"{l} {i}");}}
        let c=a.forward(&x);assert_eq!(c.dec[0][0],x[71]);for t in 1..24{assert_eq!(c.dec[t][0],c.y[t-1]);}
        assert_eq!(a.predict(&x),Seq2Seq::new(3,42).predict(&x));assert_ne!(a.predict(&x),Seq2Seq::new(3,123).predict(&x));
        let mut a=a;let before=loss(&a);let mut opt=Adam::new(a.parameters(),0.001);for _ in 0..200{assert!(a.step(&x,&y,&mut opt).0.is_finite());}assert!(a.finite()&&loss(&a)<before*0.1);
    }
    #[test]fn registered_widths(){for h in [16,32]{let a=Seq2Seq::new(h,42);assert_eq!(a.parameters(),h*(7+h)+h*(11+h)+h+1);assert_eq!(a.predict(&[0.;80]).len(),24);}}
}
