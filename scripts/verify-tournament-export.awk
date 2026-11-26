# Inputs: metrics.txt then decompressed prediction CSV. GNU awk on Ubuntu CI.
BEGIN{FS=",";tol=0.00000006}
FNR==NR{if($1=="score"){n[$3]=$4;mae[$3]=$5;rmse[$3]=$6;fold=$2}next}
FNR==1{if($0!="fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c"){print "bad header" > "/dev/stderr";exit 1}next}
{
 if(NF!=6||$1!=fold||!($2 in n)||$3!~/^[0-9]+$/||$4!~/^[0-9]+$/||$5!~/^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/||$6!~/^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/){print "bad prediction row",FNR > "/dev/stderr";exit 1}
 m=$2; i=++count[m]; expected=(i-1)%24+1;
 if($4!=expected || (expected==1 && i>1 && $3<=last[m]) || (expected!=1 && $3!=last[m])){print "bad origin/lead sequence",FNR > "/dev/stderr";exit 1}
 last[m]=$3; key=$3 "," $4 "," $5;
 if(m=="daily"){ref[i]=key}else if(!(i in ref)||ref[i]!=key){print "unmatched target row",FNR > "/dev/stderr";exit 1}
 e=$6-$5; abs_sum[m]+=(e<0?-e:e); sq_sum[m]+=e*e
}
END{
 if(length(n)!=4){print "missing metrics" > "/dev/stderr";exit 1}
 for(m in n){
  if(count[m]!=24*n[m]||!count[m]){print "wrong row count",m > "/dev/stderr";exit 1}
  a=abs_sum[m]/count[m];r=sqrt(sq_sum[m]/count[m]);
  if((a-mae[m])^2>tol^2||(r-rmse[m])^2>tol^2){print "metric mismatch",m,a,r > "/dev/stderr";exit 1}
  printf "export-verified,%s,%s,%d,%.8f,%.8f\n",fold,m,count[m],a,r
 }
}
