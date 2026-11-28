//! Operational P13 adapter. Reuse the exact frozen research implementation,
//! including split ordering, quantile ranks, residual arithmetic and support size.
//! Do not maintain a second copy of the numerical algorithm.
#[allow(dead_code)]
#[path="../../../campaigns/014-probabilistic/probabilistic.rs"]
mod frozen;
use frozen::{ensemble::{Features,Row},regression::{Tree,Node},core};
use super::Y;
pub const SUPPORT:usize=core::M;
/// All four members must be out-of-fold predictions in I,L,R,M order.
/// The caller must establish source/model provenance for each fit cutoff.
pub struct CalibrationOrigin{
    pub origin:i64,
    pub fit_cutoff:i64,
    pub target:[f64;Y],
    pub members:[[f64;Y];4],
}
pub struct P13{tree:Tree}
fn features(origin:i64,h:usize,members:&[[f64;Y];4])->Features{
    let mut base=[f64::NAN;8];
    for k in 0..4{base[k]=members[k][h];}
    // P13 uses only lead, center and spread, not month or absolute origin.
    Features{origin,lead:h+1,month:1,base}
}
impl P13{
    pub fn fit(rows:&[CalibrationOrigin],cutoff:i64)->Result<Self,String>{
        if rows.is_empty()||cutoff<=0||cutoff%60!=0||cutoff>=crate::weather::minute_limit()||
            rows.windows(2).any(|w|w[0].origin>=w[1].origin){return Err("invalid calibration cohort".into())}
        let mut data=Vec::with_capacity(rows.len().checked_mul(Y).ok_or("calibration size overflow")?);
        for r in rows{
            if r.origin<0||r.origin%60!=0||r.fit_cutoff<=0||r.fit_cutoff%60!=0||
                r.origin.checked_add(60).is_none_or(|t|r.fit_cutoff>t)||
                r.origin.checked_add(1440).is_none_or(|t|t>=cutoff)||
                r.target.iter().chain(r.members.iter().flatten()).any(|v|!v.is_finite()||v.abs()>1e6){
                return Err("invalid or noncausal calibration origin".into())
            }
            for h in 0..Y{data.push(Row{x:features(r.origin,h,&r.members),target:r.target[h],daily:0.});}
        }
        let refs:Vec<_>=data.iter().collect();
        let tree=Tree::fit(&refs,true);Ok(Self{tree})
    }
    pub fn support(&self,members:&[[f64;Y];4])->Result<[[f64;SUPPORT];Y],String>{
        if members.iter().flatten().any(|v|!v.is_finite()||v.abs()>1e6){return Err("invalid P13 members".into())}
        let mut out=[[0.;SUPPORT];Y];
        for h in 0..Y{
            let v=self.tree.support(&features(0,h,members));
            if v.len()!=SUPPORT||v.iter().any(|x|!x.is_finite())||v.windows(2).any(|w|w[0]>w[1]){return Err("invalid P13 support".into())}
            out[h].copy_from_slice(&v);
        }
        Ok(out)
    }
    pub fn median(s:&[f64;SUPPORT])->f64{core::quantile(s,0.5)}
    pub fn band(s:&[f64;SUPPORT],coverage:f64)->Result<(f64,f64),String>{
        if !coverage.is_finite()||coverage<=0.||coverage>=1.||s.iter().any(|x|!x.is_finite())||s.windows(2).any(|w|w[0]>w[1]){return Err("invalid interval request".into())}
        Ok((core::quantile(s,(1.-coverage)/2.),core::quantile(s,(1.+coverage)/2.)))
    }
    /// Exact tree parameters, in canonical preorder. Full calibration source
    /// provenance belongs in the enclosing artifact, not this inference payload.
    pub fn model_bytes(&self)->Result<Vec<u8>,String>{
        fn write_node(n:&Node,out:&mut Vec<u8>)->Result<(),String>{
            match n{
                Node::Leaf{count,impurity,support}=>{
                    out.push(0);out.extend((*count as u64).to_le_bytes());out.extend(impurity.to_le_bytes());
                    for &x in support{out.extend(x.to_le_bytes());}
                }
                Node::Split{count,impurity,objective,feature,threshold,left,right}=>{
                    out.push(1);out.extend((*count as u64).to_le_bytes());out.extend(impurity.to_le_bytes());
                    out.extend(objective.to_le_bytes());out.push(*feature as u8);out.extend(threshold.to_le_bytes());
                    write_node(left,out)?;write_node(right,out)?;
                }
            }Ok(())
        }
        let mut out=b"AEVP13\x01".to_vec();
        for n in &self.tree.roots{write_node(n,&mut out)?;}
        Ok(out)
    }
    pub fn from_model_bytes(bytes:&[u8])->Result<Self,String>{
        fn take<const N:usize>(r:&mut &[u8])->Result<[u8;N],String>{let v=r.get(..N).ok_or("truncated P13 tree")?.try_into().unwrap();*r=&r[N..];Ok(v)}
        fn finite(r:&mut &[u8])->Result<f64,String>{let v=f64::from_le_bytes(take(r)?);if !v.is_finite(){return Err("nonfinite P13 parameter".into())}Ok(v)}
        fn read_node(r:&mut &[u8],depth:usize)->Result<(Node,usize),String>{
            let tag=take::<1>(r)?[0];let count=usize::try_from(u64::from_le_bytes(take(r)?)).map_err(|_|"P13 count overflow")?;
            let impurity=finite(r)?;
            if count==0||impurity< -1e-10{return Err("invalid P13 node".into())}
            let node=match tag{
                0=>{
                    let mut support=Vec::with_capacity(SUPPORT);
                    for _ in 0..SUPPORT{support.push(finite(r)?);}
                    if support.windows(2).any(|w|w[0]>w[1]){return Err("unordered P13 leaf".into())}
                    Node::Leaf{count,impurity,support}
                }
                1 if depth<2=>{
                    let objective=finite(r)?;let feature=take::<1>(r)?[0] as usize;let threshold=finite(r)?;
                    if feature>1||objective>=impurity-1e-12||objective< -1e-10{return Err("invalid P13 split".into())}
                    let (left,l)=read_node(r,depth+1)?;let (right,n)=read_node(r,depth+1)?;
                    if l<128||n<128||l.checked_add(n)!=Some(count){return Err("invalid P13 split counts".into())}
                    Node::Split{count,impurity,objective,feature,threshold,left:Box::new(left),right:Box::new(right)}
                }
                _=>return Err("invalid P13 tag/depth".into()),
            };Ok((node,count))
        }
        let mut r=bytes.strip_prefix(b"AEVP13\x01").ok_or("P13 magic/version mismatch")?;
        let mut roots=Vec::with_capacity(Y);let mut expected=None;
        for _ in 0..Y{let(n,count)=read_node(&mut r,0)?;if expected.is_some_and(|v|v!=count){return Err("P13 unequal lead counts".into())}expected=Some(count);roots.push(n);}
        if !r.is_empty(){return Err("trailing P13 payload".into())}
        Ok(Self{tree:Tree{b:true,roots}})
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn causal_fit_and_exact_tree_reload(){
        let rows:Vec<_>=(0..512).map(|i|{
            let x=(i/256)as f64;
            CalibrationOrigin{origin:60*(i+100),fit_cutoff:60,target:[11.*x;Y],members:[[x;Y];4]}
        }).collect();
        let cutoff=60*1000;
        let m=P13::fit(&rows,cutoff).unwrap();
        let bytes=m.model_bytes().unwrap();let n=P13::from_model_bytes(&bytes).unwrap();
        for v in [0.,1.]{let p=m.support(&[[v;Y];4]).unwrap();assert_eq!(p,n.support(&[[v;Y];4]).unwrap());assert_eq!(p[0],[11.*v;SUPPORT]);}
        assert_eq!(n.model_bytes().unwrap(),bytes);
        assert!(P13::fit(&rows,rows.last().unwrap().origin+1440).is_err());
        assert!(P13::from_model_bytes(&bytes[..bytes.len()-1]).is_err());
        let mut bad=bytes;bad.push(0);assert!(P13::from_model_bytes(&bad).is_err());
    }
}
