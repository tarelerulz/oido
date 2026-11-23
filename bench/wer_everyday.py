import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import *   # CLI, MODEL, LM, KIT, OIDO_TYPE, SILERO, CLIPS_DIR, TRUTH_FILE, TESTSET, RESULTS_WER, JFK (all from environment variables, see bench/README.md)
import ast, os, sys, time, wave, threading, subprocess, statistics as st, glob
os.nice(19); os.sched_setaffinity(0,{1,2,3})
os.environ['GPUPORT_ASM']='1'; os.environ['GPUPORT_SPIN']='0'
sys.path.insert(0,KIT)
import numpy as np
from pathlib import Path
from gpuport.nemo8 import Nemo8Encoder
src=Path(OIDO_TYPE).read_text()
def log(*a): pass
_body=ast.parse(src).body
_lo=next(n.lineno for n in _body if isinstance(n,ast.ClassDef) and n.name=='HybridTail')
_hi=next(n.lineno for n in _body if isinstance(n,ast.ClassDef) and n.name=='OidoCLI')
# HybridTail + the CUT_RULES block that follows it (2026-11-22 refactor); the live rule is used unless the bench calls rules itself
exec(compile(ast.Module(body=[n for n in _body if _lo<=n.lineno<_hi],type_ignores=[]),'live','exec'))
h=HybridTail.__new__(HybridTail); h.dead=False
class Alive:
    def poll(self): return None
h.p=Alive()
R=RATE=16000
def load(p):
    with wave.open(p) as w: return np.frombuffer(w.readframes(w.getnframes()),'<i2')
cands=[]
W=TESTSET+'/'
import json
refs=[e['text'] for e in json.load(open(W+'sentences.json'))]
clp=[load(W+'clips16/s%02d.wav'%i) for i in range(len(refs))]
items=[]
for i,x in enumerate(clp):
    if len(x)/R>=4.5: items.append((len(x)/R,refs[i],x))
gap=np.zeros(int(.4*R),np.int16)
for k in (2,3):
    for i in range(len(clp)-k+1):
        x=clp[i]
        for j in range(1,k): x=np.concatenate([x,gap,clp[i+j]])
        if 4.5<=len(x)/R<=18.5: items.append((len(x)/R,' '.join(refs[i:i+k]),x))
print('items',len(items),flush=True)
sel=[(d,'',x,0) for d,_,x in items]

cands=[(d,'',x,0) for d,_,x in items]
CLI=CLI; MODEL=MODEL
def wr(p,x):
    with wave.open(p,'wb') as f: f.setnchannels(1);f.setsampwidth(2);f.setframerate(R);f.writeframes(x.tobytes())
cpu=subprocess.Popen(['taskset','-c','1,2,3',CLI,MODEL,'--serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,env={**os.environ,'TASR_THREADS':'3'})
def cpu_t(x):
    wr('/tmp/b4.wav',x); t=time.perf_counter(); cpu.stdin.write('/tmp/b4.wav\n'); cpu.stdin.flush(); s=cpu.stdout.readline()
    return (time.perf_counter()-t)*1000, s.strip().split('\t')[-1]
time.sleep(.5)
enc=Nemo8Encoder().start()
for t in Path(f'/proc/{enc.srv.proc.pid}/task').iterdir(): os.sched_setaffinity(int(t.name),{3})

import re, difflib
cpu_t(sel[0][2]); enc.transcribe(sel[0][2])
long=max(cands,key=lambda c:c[0])[2]
def med(f,k=3):
    v=[]
    for _ in range(k): time.sleep(.15); v.append(f())
    return st.median(v)
def tg(x):
    t=time.perf_counter(); enc.transcribe(x); return (time.perf_counter()-t)*1000
secs=[0.5*i for i in range(2,37)]
gp=[(s,med(lambda:tg(long[:int(s*R)]),2)) for s in secs if int(s*R)<=len(long)]
cp=[(s,med(lambda:cpu_t(long[:int(s*R)])[0],2)) for s in secs if int(s*R)<=len(long)]
print('GPU curve',[(a,round(b)) for a,b in gp[::3]]); print('CPU curve',[(a,round(b)) for a,b in cp[::3]],flush=True)
gx=[a for a,_ in gp]; gy=[b for _,b in gp]; cx=[a for a,_ in cp]; cy=[b for _,b in cp]
G=lambda s: float(np.interp(s,gx,gy)); C=lambda s: float(np.interp(s,cx,cy))
a_=np.abs
def rule_balance(x,quiet):
    n=len(x); ab=a_(x.astype(np.float32)); c=np.concatenate([[0.0],np.cumsum(ab)]); w=R//10
    st_=np.arange(0,n-w,160); e=(c[st_+w]-c[st_])/w; thr=max(30.0,0.05*float(np.percentile(e,70)))
    best=None
    for s0 in range(int(.15*n),int(.85*n),160):
        if quiet and (c[s0+w]-c[s0])/w>thr: continue
        cut=s0+w//2; t=max(G(cut/R),C((n-cut)/R))
        if best is None or t<best[0]: best=(t,cut)
    return None if best is None else int(best[1])
def rule_snap(x,rad):
    cb=rule_balance(x,False)
    if cb is None: return None
    n=len(x); ab=a_(x.astype(np.float32)); c=np.concatenate([[0.0],np.cumsum(ab)]); w=R//10
    lo=max(int(.15*n),cb-int(rad*R)); hi=min(int(.85*n),cb+int(rad*R)); best=None
    for s0 in range(lo,hi,160):
        if s0+w>=n: break
        e=(c[s0+w]-c[s0])/w
        if best is None or e<best[0]: best=(e,s0+w//2)
    return best[1]
def norm(t): return re.sub(r"[^a-z' ]"," ",t.lower()).split()
def wd(a,b):
    sm=difflib.SequenceMatcher(None,a,b); return len(a)+len(b)-2*sum(m.size for m in sm.get_matching_blocks())
def hyb(x,cut):
    res={}; th=threading.Thread(target=lambda:res.setdefault('c',cpu_t(x[cut:])))
    t=time.perf_counter(); th.start(); ht=enc.transcribe(x[:cut]); th.join(); return (time.perf_counter()-t)*1000, ht.strip()+' '+res['c'][1]

import collections
tot=collections.defaultdict(lambda:[0,0.0,0]); nref=0
per=[]
for (d,ref,x) in items:
    rw=norm(ref); nref+=len(rw)
    cuts={'A live':h.split(x),'C balance any':rule_balance(x,False),'D snap .5':rule_snap(x,0.5)}
    row={}
    g=time.perf_counter(); gt=enc.transcribe(x); gms=(time.perf_counter()-g)*1000
    ct=cpu_t(x); row['GPU only']=(wd(rw,norm(gt)),gms); row['CPU only']=(wd(rw,norm(ct[1])),ct[0])
    for k,cut in cuts.items():
        if cut is None: row[k]=row['GPU only']; continue
        time.sleep(.2); ms,txt=hyb(x,cut); row[k]=(wd(rw,norm(txt)),ms)
    per.append((d,len(rw),row))
    for k,(e,ms) in row.items(): tot[k][0]+=e; tot[k][1]+=ms; tot[k][2]+=1
    print(f"{d:5.1f}s {len(rw):3d}w "+'  '.join(f"{k}:{e}err/{ms:.0f}ms" for k,(e,ms) in row.items()),flush=True)
print('\nWER (word errors / %d reference words) and mean ms:'%nref)
for k,(e,ms,n) in tot.items(): print(f"{k:14s} errors {e:4d}  WER {100*e/nref:5.2f}%  mean {ms/n:5.0f} ms")
for lo,hi in ((4.5,9),(9,19)):
    print(f'-- clips {lo}-{hi} s:')
    for k in tot:
        sel_=[r for r in per if lo<=r[0]<hi]; e=sum(r[2][k][0] for r in sel_); w=sum(r[1] for r in sel_); ms=np.mean([r[2][k][1] for r in sel_])
        print(f"   {k:14s} n={len(sel_)} WER {100*e/w:5.2f}%  mean {ms:4.0f} ms")
cpu.stdin.close(); cpu.wait(5); enc.close()
