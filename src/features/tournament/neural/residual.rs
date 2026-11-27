//! Trainable residual MLP; explicit backpropagation, shared Adam timestep.
use crate::adam::Adam;
#[derive(Clone)]
pub(super) struct Affine { n:usize, m:usize, pub(super) w:Vec<f64> }
pub(super) fn sample(state:&mut u64,scale:f64)->f64 {
    *state^=*state<<13; *state^=*state>>7; *state^=*state<<17;
    (2.*((*state>>11) as f64/9007199254740992.)-1.)*scale
}
impl Affine {
    pub(super) fn new(n:usize,m:usize,state:&mut u64)->Self {
        let scale=(6./(n+m)as f64).sqrt();
        Self{n,m,w:(0..m*(n+1)).map(|i|if i%(n+1)==n{0.}else{sample(state,scale)}).collect()}
    }
    pub(super) fn forward(&self,x:&[f64])->Vec<f64>{
        assert_eq!(x.len(),self.n);
        self.w.chunks_exact(self.n+1).map(|r|r[self.n]+r[..self.n].iter().zip(x).map(|(a,b)|a*b).sum::<f64>()).collect()
    }
    pub(super) fn backward(&self,x:&[f64],dy:&[f64])->(Vec<f64>,Vec<f64>){
        assert_eq!(x.len(),self.n);assert_eq!(dy.len(),self.m);
        let mut dx=vec![0.;self.n];let mut g=vec![0.;self.w.len()];
        for(j,row)in self.w.chunks_exact(self.n+1).enumerate(){let d=dy[j];let a=j*(self.n+1);
            for i in 0..self.n{dx[i]+=row[i]*d;g[a+i]=x[i]*d;}
            g[a+self.n]=d;
        }(dx,g)
    }
}
#[derive(Clone)]
pub(super) struct Residual { layers:Vec<Affine> }
impl Residual {
    pub(super) fn new(n:usize,w:usize,b:usize,o:usize,seed:u64)->Self{
        let mut s=seed.max(1);let mut layers=vec![Affine::new(n,w,&mut s)];
        for _ in 0..b{layers.push(Affine::new(w,w,&mut s));layers.push(Affine::new(w,w,&mut s));}
        layers.push(Affine::new(w,o,&mut s));Self{layers}
    }
    pub(super) fn parameters(&self)->usize{self.layers.iter().map(|l|l.w.len()).sum()}
    pub(super) fn finite(&self)->bool{self.layers.iter().flat_map(|l|&l.w).all(|v|v.is_finite())}
    fn cache(&self,x:&[f64])->Vec<Vec<f64>>{
        let mut a=vec![x.to_vec(),self.layers[0].forward(x).into_iter().map(f64::tanh).collect()];
        for l in (1..self.layers.len()-1).step_by(2){
            let z:Vec<_>=self.layers[l].forward(&a[l]).into_iter().map(f64::tanh).collect();
            let h:Vec<_>=self.layers[l+1].forward(&z).iter().zip(&a[l]).map(|(u,v)|u+v).collect();
            a.push(z);a.push(h);
        }
        a.push(self.layers.last().unwrap().forward(a.last().unwrap()));a
    }
    pub(super) fn predict(&self,x:&[f64])->Vec<f64>{self.cache(x).pop().unwrap()}
    fn gradients(&self,x:&[f64],y:&[f64])->(f64,Vec<Vec<f64>>){
        let a=self.cache(x);let end=self.layers.len()-1;
        let errors:Vec<_>=a.last().unwrap().iter().zip(y).map(|(p,t)|p-t).collect();
        let n=y.len()as f64;let loss=errors.iter().map(|e|e*e).sum::<f64>()/n;
        let dy:Vec<_>=errors.iter().map(|e|2.*e/n).collect();
        let(mut d,g)=self.layers[end].backward(&a[end],&dy);
        let mut grads=vec![Vec::new();self.layers.len()];grads[end]=g;
        for l in (1..end).step_by(2).rev(){
            let(mut dz,g2)=self.layers[l+1].backward(&a[l+1],&d);grads[l+1]=g2;
            for(v,z)in dz.iter_mut().zip(&a[l+1]){*v*=1.-z*z;}
            let(dh,g1)=self.layers[l].backward(&a[l],&dz);grads[l]=g1;
            for(v,h)in d.iter_mut().zip(dh){*v+=h;}
        }
        for(v,h)in d.iter_mut().zip(&a[1]){*v*=1.-h*h;}
        grads[0]=self.layers[0].backward(x,&d).1;(loss,grads)
    }
    pub(super) fn step(&mut self,x:&[f64],y:&[f64],opt:&mut Adam)->(f64,bool){
        let(loss,g)=self.gradients(x,y);
        let norm=g.iter().flatten().map(|v|v*v).sum::<f64>().sqrt();
        if !norm.is_finite()||!loss.is_finite(){return(f64::NAN,false)}
        let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        opt.begin();for(l,grad)in self.layers.iter_mut().zip(g){for(w,v)in l.w.iter_mut().zip(grad){opt.update(w,v*scale)}}opt.finish();(loss,clipped)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn residual_gradient_every_parameter(){
        let a=Residual::new(3,4,2,2,42);let x=[0.2,-0.7,0.5];let y=[-0.1,0.3];
        let(_,g)=a.gradients(&x,&y);let loss=|a:&Residual|a.predict(&x).iter().zip(y).map(|(p,t)|(p-t).powi(2)).sum::<f64>()/2.;
        for(l,layer)in a.layers.iter().enumerate(){for i in 0..layer.w.len(){let(mut p,mut m)=(a.clone(),a.clone());p.layers[l].w[i]+=1e-6;m.layers[l].w[i]-=1e-6;assert!(((loss(&p)-loss(&m))/2e-6-g[l][i]).abs()<1e-6,"layer {l} parameter {i}");}}
    }
    #[test]fn skip_identity(){let mut a=Residual::new(3,4,2,2,42);for l in &mut a.layers[1..5]{l.w.fill(0.);}
        let x=[0.2,-0.7,0.5];let h:Vec<_>=a.layers[0].forward(&x).into_iter().map(f64::tanh).collect();
        assert_eq!(a.predict(&x),a.layers[5].forward(&h));}
    #[test]fn seed_and_overfit(){let mut a=Residual::new(3,4,2,2,42);let x=[0.2,-0.7,0.5];let y=[-0.1,0.3];assert_eq!(a.predict(&x),Residual::new(3,4,2,2,42).predict(&x));assert_ne!(a.predict(&x),Residual::new(3,4,2,2,123).predict(&x));
        let mut opt=Adam::new(a.parameters(),0.001);for _ in 0..400{assert!(a.step(&x,&y,&mut opt).0.is_finite());}
        assert!(a.finite());assert!(a.predict(&x).iter().zip(y).all(|(p,t)|(p-t).abs()<1e-3));}
}
