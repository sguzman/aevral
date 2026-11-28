//! Versioned, checksummed inference bundle. Coreutils sha256sum is part of the locked Nix shell.
use super::OriginScales;
use crate::weather::{self,Station};
use std::{fs::{self,File,OpenOptions},io::{Read,Write},path::Path,process::{Command,Stdio}};
const MAGIC:&[u8]=b"AEVFORECAST\x01";
const SCHEMA:&str="weather80-utc-c015-v1";
const MAX_BYTES:usize=256*1024*1024;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Recipe{Mlp,P13}
impl Recipe{
    pub fn name(self)->&'static str{match self{Self::Mlp=>"mlp64-sgd-seed42",Self::P13=>"p13-c015-seed42"}}
    fn ids(self)->&'static [&'static str]{match self{Self::Mlp=>&["I"],Self::P13=>&["I","L","R","M","P13"]}}
}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct SourceFingerprint{pub name:String,pub sha256:[u8;32]}
pub struct Artifact{
    pub station:Station,
    pub cutoff:i64,
    pub scales:OriginScales,
    pub recipe:Recipe,
    pub sources:Vec<SourceFingerprint>,
    pub components:Vec<(String,Vec<u8>)>,
}
pub fn sha256(data:&[u8])->Result<[u8;32],String>{
    let mut child=Command::new("sha256sum").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e|format!("sha256sum: {e}"))?;
    let written=child.stdin.take().ok_or("missing checksum stdin")?.write_all(data);
    let out=child.wait_with_output().map_err(|e|e.to_string())?;
    written.map_err(|e|e.to_string())?;
    if !out.status.success(){return Err("sha256sum failed".into())}
    let hex=std::str::from_utf8(&out.stdout).map_err(|_|"checksum encoding")?.split_whitespace().next().ok_or("missing checksum")?;
    if hex.len()!=64||!hex.bytes().all(|b|b.is_ascii_hexdigit()){return Err("invalid checksum".into())}
    let mut digest=[0;32];for (i,b) in digest.iter_mut().enumerate(){*b=u8::from_str_radix(&hex[i*2..i*2+2],16).map_err(|_|"invalid checksum")?;}Ok(digest)
}
fn put(out:&mut Vec<u8>,v:&[u8])->Result<(),String>{
    let n=u32::try_from(v.len()).map_err(|_|"artifact field too long")?;
    if out.len().checked_add(4+v.len()).filter(|&n|n<=MAX_BYTES-32).is_none(){return Err("artifact size limit".into())}
    out.extend(n.to_le_bytes());out.extend(v);Ok(())
}
struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a>{
    fn take(&mut self,n:usize)->Result<&'a [u8],String>{let v=self.0.get(..n).ok_or("truncated artifact")?;self.0=&self.0[n..];Ok(v)}
    fn u32(&mut self)->Result<u32,String>{Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))}
    fn field(&mut self,max:usize)->Result<&'a [u8],String>{let n=self.u32()? as usize;if n>max{return Err("artifact field exceeds limit".into())}self.take(n)}
    fn string(&mut self,max:usize)->Result<String,String>{String::from_utf8(self.field(max)?.to_vec()).map_err(|_|"invalid artifact text".into())}
}
impl Artifact{
    fn validate(&self)->Result<(),String>{
        if self.cutoff<=0||self.cutoff>=weather::minute_limit()||self.cutoff%60!=0{return Err("invalid artifact cutoff".into())}
        OriginScales::from_values(self.scales.values())?;
        if self.sources.is_empty()||self.sources.len()>4096||self.sources.windows(2).any(|w|w[0].name>=w[1].name){return Err("source fingerprints must be unique and sorted".into())}
        if self.sources.iter().any(|s|s.name.is_empty()||s.name.len()>256||!s.name.bytes().all(|b|b.is_ascii_alphanumeric()||b"._-".contains(&b))){return Err("invalid source fingerprint name".into())}
        if self.components.len()!=self.recipe.ids().len()||self.components.iter().zip(self.recipe.ids()).any(|((id,payload),expected)|id!=expected||payload.is_empty()){return Err("missing, unordered, or empty model component".into())}
        Ok(())
    }
    pub fn encode(&self)->Result<Vec<u8>,String>{
        self.validate()?;
        let mut out=MAGIC.to_vec();put(&mut out,SCHEMA.as_bytes())?;put(&mut out,self.recipe.name().as_bytes())?;
        put(&mut out,self.station.as_str().as_bytes())?;out.extend(self.cutoff.to_le_bytes());
        for v in self.scales.values(){out.extend(v.to_le_bytes());}
        out.extend((self.sources.len() as u32).to_le_bytes());
        for s in &self.sources{put(&mut out,s.name.as_bytes())?;out.extend(s.sha256);}
        out.extend((self.components.len() as u32).to_le_bytes());
        for (id,payload) in &self.components{put(&mut out,id.as_bytes())?;put(&mut out,payload)?;}
        let digest=sha256(&out)?;out.extend(digest);Ok(out)
    }
    pub fn decode(bytes:&[u8])->Result<Self,String>{
        if bytes.len()>MAX_BYTES||bytes.len()<MAGIC.len()+32{return Err("artifact size mismatch".into())}
        let (body,digest)=bytes.split_at(bytes.len()-32);
        if sha256(body)?.as_slice()!=digest{return Err("artifact checksum mismatch".into())}
        let mut r=Reader(body.strip_prefix(MAGIC).ok_or("artifact magic/version mismatch")?);
        if r.string(64)?!=SCHEMA{return Err("artifact feature schema mismatch".into())}
        let recipe=match r.string(64)?.as_str(){"mlp64-sgd-seed42"=>Recipe::Mlp,"p13-c015-seed42"=>Recipe::P13,_=>return Err("unknown artifact recipe".into())};
        let station=Station::new(&r.string(8)?)?;
        let cutoff=i64::from_le_bytes(r.take(8)?.try_into().unwrap());
        let mut scales=[0.;10];for v in &mut scales{*v=f64::from_le_bytes(r.take(8)?.try_into().unwrap());}
        let scales=OriginScales::from_values(scales)?;
        let n=r.u32()? as usize;if !(1..=4096).contains(&n){return Err("invalid source count".into())}
        let mut sources=Vec::with_capacity(n);
        for _ in 0..n{sources.push(SourceFingerprint{name:r.string(256)?,sha256:r.take(32)?.try_into().unwrap()});}
        if r.u32()? as usize!=recipe.ids().len(){return Err("invalid component count".into())}
        let mut components=Vec::new();
        for &expected in recipe.ids(){let id=r.string(8)?;if id!=expected{return Err("unexpected component identity".into())}components.push((id,r.field(MAX_BYTES)?.to_vec()));}
        if !r.0.is_empty(){return Err("trailing artifact data".into())}
        let a=Self{station,cutoff,scales,recipe,sources,components};a.validate()?;Ok(a)
    }
    /// Publish only after a complete, synced write. Refuse replacing an existing artifact.
    pub fn save(&self,path:&Path)->Result<(),String>{
        let bytes=self.encode()?;
        let parent=path.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        let temp=parent.join(format!(".aevral-artifact-{}-{}",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)));
        let mut f=OpenOptions::new().write(true).create_new(true).open(&temp).map_err(|e|e.to_string())?;
        let result=(||{f.write_all(&bytes)?;f.sync_all()?;drop(f);fs::hard_link(&temp,path)?;File::open(parent)?.sync_all()})().map_err(|e|e.to_string());
        let cleanup=fs::remove_file(&temp).map_err(|e|e.to_string());result.and(cleanup)
    }
    pub fn load(path:&Path)->Result<Self,String>{
        let f=File::open(path).map_err(|e|e.to_string())?;let mut bytes=Vec::new();
        f.take(MAX_BYTES as u64+1).read_to_end(&mut bytes).map_err(|e|e.to_string())?;
        Self::decode(&bytes)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn envelope_roundtrip_and_rejections(){
        let a=Artifact{station:Station::new("LAX").unwrap(),cutoff:weather::minute("2025-01-01 00:00").unwrap(),scales:OriginScales::from_values([1.;10]).unwrap(),recipe:Recipe::Mlp,sources:vec![SourceFingerprint{name:"LAX-2024.csv".into(),sha256:sha256(b"abc").unwrap()}],components:vec![("I".into(),vec![1,2,3])]};
        assert_eq!(a.sources[0].sha256,[0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad]);
        let bytes=a.encode().unwrap();let b=Artifact::decode(&bytes).unwrap();
        assert_eq!(b.station,a.station);assert_eq!(b.cutoff,a.cutoff);assert_eq!(b.scales.values(),a.scales.values());assert_eq!(b.sources,a.sources);assert_eq!(b.components,a.components);assert_eq!(b.encode().unwrap(),bytes);
        let mut bad=bytes.clone();bad[15]^=1;assert!(Artifact::decode(&bad).is_err());assert!(Artifact::decode(&bytes[..bytes.len()-1]).is_err());
        let mut bad=a;bad.components[0].0="M".into();assert!(bad.encode().is_err());
    }
}
