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
    /// Versioned, little-endian, exact inference state (not optimizer state).
    /// Shape is checked against the caller's expected architecture before allocation.
    pub fn to_bytes(&self)->Result<Vec<u8>,String>{
        if !self.finite(){return Err("nonfinite dense model".into())}
        let mut out=b"AEVDENSE\x01".to_vec();
        out.extend(u32::try_from(self.dims.len()).map_err(|_|"dense dimension count")?.to_le_bytes());
        for &d in &self.dims{out.extend(u32::try_from(d).map_err(|_|"dense dimension")?.to_le_bytes());}
        for (w,b) in self.weights.iter().zip(&self.biases){
            for &v in w.iter().chain(b){out.extend(v.to_le_bytes());}
        }
        Ok(out)
    }
    pub fn from_bytes(bytes:&[u8],expected:&[usize])->Result<Self,String>{
        if expected.len()<2||expected.len()>64||expected.iter().any(|&d|d==0||d>u32::MAX as usize){return Err("invalid expected dense architecture".into())}
        let mut input=bytes.strip_prefix(b"AEVDENSE\x01").ok_or("dense magic/version mismatch")?;
        fn take<const N:usize>(input:&mut &[u8])->Result<[u8;N],String>{
            let v=input.get(..N).ok_or("truncated dense model")?.try_into().unwrap();*input=&input[N..];Ok(v)
        }
        let n=u32::from_le_bytes(take(&mut input)?) as usize;
        if n!=expected.len(){return Err("dense architecture length mismatch".into())}
        for &d in expected{if u32::from_le_bytes(take(&mut input)?) as usize!=d{return Err("dense architecture mismatch".into())}}
        let total=expected.windows(2).try_fold(0usize,|a,d|a.checked_add(d[0].checked_mul(d[1])?.checked_add(d[1])?)).ok_or("dense parameter overflow")?;
        if total.checked_mul(8)!=Some(input.len()){return Err("dense payload length mismatch".into())}
        fn values(input:&mut &[u8],n:usize)->Result<Vec<f64>,String>{
            (0..n).map(|_|{let v=f64::from_le_bytes(take(input)?);if !v.is_finite(){return Err("nonfinite dense parameter".into())}Ok(v)}).collect()
        }
        let mut weights=Vec::new();let mut biases=Vec::new();
        for d in expected.windows(2){weights.push(values(&mut input,d[0]*d[1])?);biases.push(values(&mut input,d[1])?);}
        Ok(Self{dims:expected.to_vec(),weights,biases})
    }
    pub fn finite(&self)->bool{self.weights.iter().chain(&self.biases).flatten().all(|v|v.is_finite())}
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
    pub fn adam_step(&mut self,x:&[f64],target:&[f64],opt:&mut crate::adam::Adam)->(f64,bool){
        self.adam_step_with(x,target,opt,|p,t|{assert_eq!(p.len(),t.len());let n=p.len();let errors:Vec<_>=p.iter().zip(t).map(|(p,t)|p-t).collect();(errors.iter().map(|e|e*e).sum::<f64>()/n as f64,errors.iter().map(|e|2.*e/n as f64).collect())})
    }
    fn gradients_with(&self,x:&[f64],target:&[f64],objective:impl Fn(&[f64],&[f64])->(f64,Vec<f64>))->(f64,Vec<Vec<f64>>,Vec<Vec<f64>>){
        let a=self.activations(x);let(value,mut d)=objective(a.last().unwrap(),target);assert_eq!(d.len(),*self.dims.last().unwrap());
        let mut gw:Vec<_>=self.weights.iter().map(|w|vec![0.;w.len()]).collect();
        let mut gb:Vec<_>=self.biases.iter().map(|b|vec![0.;b.len()]).collect();
        for l in (0..self.weights.len()).rev(){
            let n=self.dims[l];let m=self.dims[l+1];
            let prev:Vec<f64>=if l==0{vec![]}else{(0..n).map(|i|(1.-a[l][i]*a[l][i])*(0..m).map(|j|self.weights[l][j*n+i]*d[j]).sum::<f64>()).collect()};
            for j in 0..m{for i in 0..n{gw[l][j*n+i]=d[j]*a[l][i]}gb[l][j]=d[j];}
            d=prev;
        }(value,gw,gb)
    }
    pub fn adam_step_with(&mut self,x:&[f64],target:&[f64],opt:&mut crate::adam::Adam,objective:impl Fn(&[f64],&[f64])->(f64,Vec<f64>))->(f64,bool){
        let(value,gw,gb)=self.gradients_with(x,target,objective);
        if !value.is_finite()||gw.iter().chain(&gb).flatten().any(|g|!g.is_finite()){return(f64::NAN,false)}
        opt.begin();for l in (0..self.weights.len()).rev(){let n=self.dims[l];let m=self.dims[l+1];for j in 0..m{for i in 0..n{opt.update(&mut self.weights[l][j*n+i],gw[l][j*n+i])}opt.update(&mut self.biases[l][j],gb[l][j])}}opt.finish();(value,false)
    }
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
    /// C012: mean batch gradient, optional inverted hidden mask, coupled L2 on weights.
    pub(crate) fn optimization_batch(&mut self,rows:&[(&[f64],&[f64])],opt:&mut crate::adam::Adam,decay:f64,masks:Option<&[Vec<f64>]>) -> f64 {
        assert!(!rows.is_empty()&&decay>=0.&&decay.is_finite());
        assert!(masks.is_none()||self.weights.len()==2&&masks.unwrap().len()==rows.len());
        let mut gw:Vec<_>=self.weights.iter().map(|w|vec![0.;w.len()]).collect();
        let mut gb:Vec<_>=self.biases.iter().map(|b|vec![0.;b.len()]).collect();
        let mut loss=0.0_f64;
        let objective=|p:&[f64],t:&[f64]| -> (f64,Vec<f64>){assert_eq!(p.len(),t.len());let n=p.len();let e:Vec<_>=p.iter().zip(t).map(|(p,t)|p-t).collect();(e.iter().map(|e|e*e).sum::<f64>()/n as f64,e.iter().map(|e|2.*e/n as f64).collect())};
        for (j,(x,y)) in rows.iter().enumerate(){
            let (v,w,b)=if let Some(masks)=masks{
                let mask=&masks[j];let mut a=self.activations(x);assert_eq!(mask.len(),a[1].len());
                let hidden=a[1].clone();for (h,m) in a[1].iter_mut().zip(mask){*h*=m;}
                let h=&a[1];let n=h.len();let m=self.dims[2];
                a[2]=(0..m).map(|k|self.biases[1][k]+(0..n).map(|i|self.weights[1][k*n+i]*h[i]).sum::<f64>()).collect();
                let (v,dy)=objective(&a[2],y);
                let dh:Vec<_>=(0..n).map(|i|(1.-hidden[i]*hidden[i])*mask[i]*(0..m).map(|k|self.weights[1][k*n+i]*dy[k]).sum::<f64>()).collect();
                let w0:Vec<f64>=dh.iter().flat_map(|z|x.iter().map(move |x|z*x)).collect();
                let w1:Vec<f64>=dy.iter().flat_map(|z|a[1].iter().map(move |x|z*x)).collect();
                (v,vec![w0,w1],vec![dh,dy])
            }else{self.gradients_with(x,y,objective)};
            loss+=v;for (a,b) in gw.iter_mut().zip(w){for (a,b) in a.iter_mut().zip(b){*a+=b;}}
            for (a,b) in gb.iter_mut().zip(b){for (a,b) in a.iter_mut().zip(b){*a+=b;}}
        }
        let count=rows.len() as f64;
        if !loss.is_finite()||gw.iter().chain(&gb).flatten().any(|g|!g.is_finite()){return f64::NAN;}
        opt.begin();for l in (0..self.weights.len()).rev(){let n=self.dims[l];let m=self.dims[l+1];for j in 0..m{for i in 0..n{let k=j*n+i;let grad=gw[l][k]/count+decay*self.weights[l][k];opt.update(&mut self.weights[l][k],grad)}opt.update(&mut self.biases[l][j],gb[l][j]/count)}}opt.finish();loss/count
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
    #[test]fn custom_output_gradient_all_parameters(){
        let a=Dense::new(&[2,3,4],42);let x=[0.2,-0.7];let y=[0.3,-0.4];
        let objective=|p:&[f64],t:&[f64]|{let e:Vec<_>=p.iter().enumerate().map(|(i,v)|v-t[i%2]).collect();(e.iter().map(|v|v.abs()).sum::<f64>()/4.,e.iter().map(|v|v.signum()/4.).collect())};
        let(_,gw,gb)=a.gradients_with(&x,&y,objective);let eps=1e-6;let loss=|m:&Dense|objective(&m.predict(&x),&y).0;
        for l in 0..a.weights.len(){for i in 0..a.weights[l].len(){let(mut p,mut m)=(a.clone(),a.clone());p.weights[l][i]+=eps;m.weights[l][i]-=eps;assert!(((loss(&p)-loss(&m))/(2.*eps)-gw[l][i]).abs()<1e-7);}for i in 0..a.biases[l].len(){let(mut p,mut m)=(a.clone(),a.clone());p.biases[l][i]+=eps;m.biases[l][i]-=eps;assert!(((loss(&p)-loss(&m))/(2.*eps)-gb[l][i]).abs()<1e-7);}}
    }
    #[test]fn optimization_batch_single_and_duplicate_parity(){
        let x=[0.2,-0.7];let y=[0.3,-0.4];let mut a=Dense::new(&[2,3,2],42);let mut b=a.clone();let mut c=a.clone();
        let mut oa=crate::adam::Adam::new(a.parameters(),0.001);let mut ob=crate::adam::Adam::new(b.parameters(),0.001);let mut oc=crate::adam::Adam::new(c.parameters(),0.001);
        for _ in 0..4{let v=a.adam_step(&x,&y,&mut oa).0;assert_eq!(v,b.optimization_batch(&[(&x,&y)],&mut ob,0.,None));assert_eq!(v,c.optimization_batch(&[(&x,&y),(&x,&y)],&mut oc,0.,None));assert_eq!(a.predict(&x),b.predict(&x));assert_eq!(a.predict(&x),c.predict(&x));}
    }
    #[test]fn optimization_dropout_and_decay(){
        let x=[0.2,-0.7];let y=[0.3,-0.4];let mut a=Dense::new(&[2,3,2],42);let mut b=a.clone();let mut oa=crate::adam::Adam::new(a.parameters(),0.001);let mut ob=crate::adam::Adam::new(b.parameters(),0.001);
        a.optimization_batch(&[(&x,&y)],&mut oa,0.,None);b.optimization_batch(&[(&x,&y)],&mut ob,0.,Some(&[vec![1.;3]]));assert_eq!(a.predict(&x),b.predict(&x));
        let mut c=Dense::new(&[2,3,2],42);let old=c.clone();let mut oc=crate::adam::Adam::new(c.parameters(),0.001);
        c.optimization_batch(&[(&x,&y)],&mut oc,0.0001,Some(&[vec![0.;3]]));assert_ne!(old.weights[0],c.weights[0]);assert_eq!(old.biases[0],c.biases[0]);assert!(c.finite());
    }
}

#[cfg(test)]
mod persistence_tests{
    use super::*;
    #[test]
    fn exact_dense_roundtrip_and_rejections(){
        let mut m=Dense::new(&[80,64,24],42);
        let x=std::array::from_fn::<_,80,_>(|i|i as f64/80.);
        for _ in 0..5{m.train_step(&x,&[0.2;24],0.005);}
        let bytes=m.to_bytes().unwrap();
        let restored=Dense::from_bytes(&bytes,&[80,64,24]).unwrap();
        assert_eq!(m.predict(&x),restored.predict(&x));
        assert_eq!(bytes,restored.to_bytes().unwrap());
        assert!(Dense::from_bytes(&bytes,&[80,32,24]).is_err());
        assert!(Dense::from_bytes(&bytes[..bytes.len()-1],&[80,64,24]).is_err());
        let mut bad=bytes.clone();bad.push(0);
        assert!(Dense::from_bytes(&bad,&[80,64,24]).is_err());
        let mut bad=bytes.clone();bad[8]=2;
        assert!(Dense::from_bytes(&bad,&[80,64,24]).is_err());
        let mut bad=bytes;let n=bad.len();bad[n-8..].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(Dense::from_bytes(&bad,&[80,64,24]).is_err());
    }
}
