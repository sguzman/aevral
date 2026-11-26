//! Explicit BPTT for a compact patch-sequence recurrent screen.
const D:usize=16;
const P:usize=6;
const S:usize=12;
const O:usize=24;
const A:usize=8;
const R:usize=P+D+1;
const F:usize=D+A+1;
#[derive(Clone,Copy,Debug)]
pub enum Kind{Rnn,Gru,Lstm}
impl Kind{
    fn gates(self)->usize{match self{Self::Rnn=>1,Self::Gru=>3,Self::Lstm=>4}}
    pub fn parse(s:&str)->Option<Self>{match s{"rnn"=>Some(Self::Rnn),"gru"=>Some(Self::Gru),"lstm"=>Some(Self::Lstm),_=>None}}
}
type V=[f64;D];
#[derive(Clone,Copy,Default)]
struct Frame{h:V,c:V,g:[V;4],rec:V}
#[derive(Clone)]
pub struct Recurrent{kind:Kind,w:Vec<f64>,head:Vec<f64>}
fn sigmoid(x:f64)->f64{1./(1.+(-x).exp())}
impl Recurrent{
    pub fn new(kind:Kind,seed:u64)->Self{
        let mut state=seed.max(1);
        let mut sample=|s:f64|{state^=state<<13;state^=state>>7;state^=state<<17;((state>>11)as f64/(1u64<<53)as f64*2.-1.)*s};
        let w=(0..kind.gates()*D*R).map(|i|if i%R==R-1{0.}else{sample((6./(P+2*D)as f64).sqrt())}).collect();
        let head=(0..O*F).map(|i|if i%F==F-1{0.}else{sample((6./(D+A+O)as f64).sqrt())}).collect();
        Self{kind,w,head}
    }
    pub fn parameters(&self)->usize{self.w.len()+self.head.len()}
    fn at(g:usize,j:usize)->usize{(g*D+j)*R}
    fn affine(&self,g:usize,j:usize,x:&[f64],h:&V)->(f64,f64){
        let a=Self::at(g,j);
        let v=self.w[a+R-1]+(0..P).map(|i|self.w[a+i]*x[i]).sum::<f64>();
        let r=(0..D).map(|i|self.w[a+P+i]*h[i]).sum::<f64>();(v,r)
    }
    fn forward(&self,x:&[f64])->([Frame;S+1],[f64;O]){
        assert_eq!(x.len(),72+A);
        let mut frames=[Frame::default();S+1];
        for t in 0..S{
            let p=frames[t];let mut f=Frame::default();let patch=&x[t*P..(t+1)*P];
            for j in 0..D{
                match self.kind{
                    Kind::Rnn=>{let(a,b)=self.affine(0,j,patch,&p.h);f.h[j]=(a+b).tanh();}
                    Kind::Gru=>{
                        for g in 0..2{let(a,b)=self.affine(g,j,patch,&p.h);f.g[g][j]=sigmoid(a+b)}
                        let(a,b)=self.affine(2,j,patch,&p.h);f.rec[j]=b;
                        let(z,r)=(f.g[0][j],f.g[1][j]);let n=(a+r*b).tanh();f.g[2][j]=n;f.h[j]=(1.-z)*p.h[j]+z*n;
                    }
                    Kind::Lstm=>{
                        for g in 0..4{let(a,b)=self.affine(g,j,patch,&p.h);f.g[g][j]=if g==3{(a+b).tanh()}else{sigmoid(a+b)}}
                        let(i,forget,o,n)=(f.g[0][j],f.g[1][j],f.g[2][j],f.g[3][j]);
                        f.c[j]=forget*p.c[j]+i*n;f.h[j]=o*f.c[j].tanh();
                    }
                }
            }frames[t+1]=f;
        }
        let h=frames[S].h;
        let y=std::array::from_fn(|k|{let a=k*F;self.head[a+F-1]+(0..D).map(|j|self.head[a+j]*h[j]).sum::<f64>()+(0..A).map(|j|self.head[a+D+j]*x[72+j]).sum::<f64>()});
        (frames,y)
    }
    pub fn predict(&self,x:&[f64])->[f64;O]{self.forward(x).1}
    fn gradients(&self,x:&[f64],target:&[f64])->(f64,Vec<f64>,Vec<f64>){
        assert_eq!(target.len(),O);
        let(frames,y)=self.forward(x);
        let mut gw=vec![0.;self.w.len()];let mut go=vec![0.;self.head.len()];let(mut dh,mut dc)=([0.;D],[0.;D]);let mut loss=0.;
        for k in 0..O{
            let e=y[k]-target[k];loss+=e*e/O as f64;let d=2.*e/O as f64;let a=k*F;
            for j in 0..D{go[a+j]=d*frames[S].h[j];dh[j]+=d*self.head[a+j];}
            for j in 0..A{go[a+D+j]=d*x[72+j];}go[a+F-1]=d;
        }
        for t in (0..S).rev(){
            let p=frames[t];let f=frames[t+1];let patch=&x[t*P..(t+1)*P];
            let(mut prev,mut prev_c)=([0.;D],[0.;D]);
            for j in 0..D{
                let mut d=[0.;4];let mut recurrent=[0.;4];
                match self.kind{
                    Kind::Rnn=>{d[0]=dh[j]*(1.-f.h[j]*f.h[j]);recurrent[0]=d[0];}
                    Kind::Gru=>{
                        let(z,r,n)=(f.g[0][j],f.g[1][j],f.g[2][j]);
                        prev[j]+=dh[j]*(1.-z);
                        d[0]=dh[j]*(n-p.h[j])*z*(1.-z);
                        d[2]=dh[j]*z*(1.-n*n);
                        d[1]=d[2]*f.rec[j]*r*(1.-r);
                        recurrent=[d[0],d[1],d[2]*r,0.];
                    }
                    Kind::Lstm=>{
                        let(i,forget,o,n)=(f.g[0][j],f.g[1][j],f.g[2][j],f.g[3][j]);
                        let tc=f.c[j].tanh();let cgrad=dc[j]+dh[j]*o*(1.-tc*tc);
                        d=[cgrad*n*i*(1.-i),cgrad*p.c[j]*forget*(1.-forget),dh[j]*tc*o*(1.-o),cgrad*i*(1.-n*n)];
                        recurrent=d;prev_c[j]=cgrad*forget;
                    }
                }
                for g in 0..self.kind.gates(){
                    let a=Self::at(g,j);
                    for i in 0..P{gw[a+i]+=d[g]*patch[i]}
                    for i in 0..D{gw[a+P+i]+=recurrent[g]*p.h[i];prev[i]+=recurrent[g]*self.w[a+P+i]}
                    gw[a+R-1]+=d[g];
                }
            }dh=prev;dc=prev_c;
        }(loss,gw,go)
    }
    pub fn adam_step(&mut self,x:&[f64],y:&[f64],opt:&mut crate::adam::Adam)->(f64,bool){
        let(loss,gw,go)=self.gradients(x,y);
        let norm=gw.iter().chain(&go).map(|v|v*v).sum::<f64>().sqrt();
        let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        opt.begin();
        for(w,g)in self.w.iter_mut().zip(gw){opt.update(w,scale*g)}
        for(w,g)in self.head.iter_mut().zip(go){opt.update(w,scale*g)}
        opt.finish();(loss,clipped)
    }
    pub fn train_step(&mut self,x:&[f64],y:&[f64],lr:f64)->(f64,bool){
        let(loss,gw,go)=self.gradients(x,y);
        let norm=gw.iter().chain(&go).map(|v|v*v).sum::<f64>().sqrt();
        let clipped=norm>5.;let scale=if clipped{5./norm}else{1.};
        for(w,g)in self.w.iter_mut().zip(gw){*w-=lr*scale*g}
        for(w,g)in self.head.iter_mut().zip(go){*w-=lr*scale*g}
        (loss,clipped)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn gradient_all_parameters(){
        let x:Vec<_>=(0..80).map(|i|(i as f64*0.13).sin()*0.2).collect();let y=[0.1;24];let eps=1e-6;
        for kind in [Kind::Rnn,Kind::Gru,Kind::Lstm]{
            let m=Recurrent::new(kind,42);let(_,gw,go)=m.gradients(&x,&y);
            let loss=|a:&Recurrent|a.predict(&x).iter().zip(y).map(|(p,t)|(p-t).powi(2)).sum::<f64>()/24.;
            for (head,grad)in [(false,gw),(true,go)]{
                for (i,g) in grad.iter().enumerate(){
                    let(mut a,mut b)=(m.clone(),m.clone());
                    if head{a.head[i]+=eps;b.head[i]-=eps}else{a.w[i]+=eps;b.w[i]-=eps}
                    assert!(((loss(&a)-loss(&b))/(2.*eps)-g).abs()<1e-6,"{kind:?} {head} {i}");
                }
            }
            assert_eq!(m.predict(&x),Recurrent::new(kind,42).predict(&x));
            let mut m=m;let before=loss(&m);for _ in 0..100{m.train_step(&x,&y,0.02);}assert!(loss(&m)<before);
        }
    }
}
