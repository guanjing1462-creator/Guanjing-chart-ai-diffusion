/* Scores from 整数列表均衡度算法说明 v1.1. Sorted unique positions;
   accepts finite real positions too, avoiding rounding after segment concatenation. */
(()=>{
 const methods=['dice','jaccard','wasserstein','ks','match'];
 function sortedScore(a,b,method){
  const m=a.length,n=b.length;if(!m||!n)return +(m===n);
  let i=0,j=0,k=0;while(i<m&&j<n){if(a[i]===b[j]){k++;i++;j++;}else if(a[i]<b[j])i++;else j++;}
  if(method==='dice')return 2*k/(m+n);if(method==='jaccard')return k/(m+n-k);
  if(k===m&&m===n)return 1;
  const R=Math.max(a[m-1],b[n-1])-Math.min(a[0],b[0]),q=Math.min(m,n);let value;
  if(method==='match'){
   if(m>n)[a,b]=[b,a];const delta=b.length-a.length;let cost=0;
   if(!delta){for(i=0;i<a.length;i++)cost+=Math.abs(a[i]-b[i]);}
   else{let prev=new Float64Array(delta+1),curr=new Float64Array(delta+1);for(i=1;i<=a.length;i++){curr[0]=prev[0]+Math.abs(a[i-1]-b[i-1]);for(let t=1;t<=delta;t++)curr[t]=Math.min(curr[t-1],prev[t]+Math.abs(a[i-1]-b[i+t-1]));[prev,curr]=[curr,prev];}cost=prev[delta];}
   value=2*(q-cost/R)/(m+n);
  }else{
   i=j=0;let last=Math.min(a[0],b[0]),area=0,peak=0;
   while(i<m||j<n){const x=i===m?b[j]:j===n?a[i]:Math.min(a[i],b[j]);area+=Math.abs(i*n-j*m)*(x-last);while(i<m&&a[i]===x)i++;while(j<n&&b[j]===x)j++;peak=Math.max(peak,Math.abs(i*n-j*m));last=x;}
   value=2*q/(m+n)*(1-(method==='ks'?peak/(m*n):area/(m*n*R)));
  }
  return Math.max(0,Math.min(1,value));
 }
 function score(a,b,method){if(!methods.includes(method))throw Error('未知均衡度算法');for(const v of [a,b])if(v.some(x=>!Number.isFinite(x)))throw Error('无效的时间位置');return sortedScore([...new Set(a)].sort((x,y)=>x-y),[...new Set(b)].sort((x,y)=>x-y),method);}
 globalThis.ListBalance={methods,score,sortedScore};
})();
