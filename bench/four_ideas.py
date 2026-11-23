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


import collections, difflib
base={}   # per item baseline from results.txt (same item order): {engine:(err,ms)}
for ln in open(RESULTS_WER):
    m=re.match(r"\s*([\d.]+)s\s+(\d+)w (.*)",ln)
    if m and 'err/' in ln:
        d={}
        for k,e,t in re.findall(r"(GPU only|CPU only|A live|C balance any|D snap \.5):(\d+)err/(\d+)ms",m.group(3)): d[k]=(int(e),int(t))
        base[len(base)]=(float(m.group(1)),int(m.group(2)),d)
assert len(base)==len(items),(len(base),len(items))
OV=int(float(os.environ.get('OV_S','1.0'))*R)
def stitch(ht,tt):
    a=norm(ht); b=norm(tt)
    if not a or not b: return a+b
    A=a[-8:]; B=b[:8]; sm=difflib.SequenceMatcher(None,A,B,autojunk=False); m=sm.find_longest_match(0,len(A),0,len(B))
    if m.size>=1: return a[:len(a)-len(A)+m.a+m.size]+b[m.b+m.size:]
    return a+b
def hyb_ov(x,cut,ov):
    n=len(x); res={}
    th=threading.Thread(target=lambda:res.setdefault('c',cpu_t(x[max(0,cut-ov):])))
    t=time.perf_counter(); th.start(); ht=enc.transcribe(x[:min(n,cut+ov)]); th.join(); ms=(time.perf_counter()-t)*1000
    return ms, stitch(ht,res['c'][1])
tot=collections.defaultdict(lambda:[0,0.0]); lat=[];nref=0
per=[]
for idx,(d,ref,x) in enumerate(items):
    rw=norm(ref); nref+=len(rw); b=base[idx][2]; row=dict(b)
    cutA=h.split(x); cutB=rule_balance(x,True); cutD=rule_snap(x,0.5)
    # idea 3: fresh-curve balance, quiet moments only (B)
    if cutB is None: row['B quiet-balance']=b['GPU only']
    else: time.sleep(.2); ms,txt=hyb(x,cutB); row['B quiet-balance']=(wd(rw,norm(txt)),ms)
    # idea 1: overlap + stitch on the live cut and on the snap cut
    for k,cut in (('A+overlap',cutA),('D+overlap',cutD)):
        if cut is None: row[k]=b['GPU only']; continue
        time.sleep(.2); ms,words=hyb_ov(x,cut,OV); row[k]=(wd(rw,words),ms)
    # idea 2: streaming simulation with the live cut (accuracy = A live, timing from separate head/tail runs)
    if cutA is not None:
        time.sleep(.2); g=time.perf_counter(); enc.transcribe(x[:cutA]); gh=(time.perf_counter()-g)*1000
        time.sleep(.2); ct_=cpu_t(x[cutA:])[0]
        end_ms=len(x)/R*1000; head_done=cutA/R*1000+300+gh          # cut known 0.3 s after the pause; head starts then
        lat_ms=max(ct_, head_done-end_ms)                           # after the speaker stops: CPU tail (GPU head usually done already)
        row['stream A (latency)']=(b['A live'][0],lat_ms)
    else: row['stream A (latency)']=b['GPU only']
    # idea 4: phrases under 9 s stay GPU-only (live default MIN) or go CPU-only; hybrid A only >= 9 s
    row['MIN9 short=GPU']=b['GPU only'] if d<9 else b['A live']
    row['MIN9 short=CPU']=b['CPU only'] if d<9 else b['A live']
    per.append((d,len(rw),row))
    for k,(e,ms) in row.items(): tot[k][0]+=e; tot[k][1]+=ms
    print(f"{d:5.1f}s "+'  '.join(f"{k}:{e}/{ms:.0f}" for k,(e,ms) in row.items()),flush=True)
print('\nWER (errors / %d words) and mean ms (stream row = latency after speech ends):'%nref)
for k,(e,ms) in tot.items(): print(f"{k:22s} errors {e:4d}  WER {100*e/nref:5.2f}%  mean {ms/len(items):5.0f} ms")
for lo,hi in ((4.5,9),(9,19)):
    print(f'-- clips {lo}-{hi} s:')
    for k in tot:
        sel_=[r for r in per if lo<=r[0]<hi]; e=sum(r[2][k][0] for r in sel_); w=sum(r[1] for r in sel_); ms=np.mean([r[2][k][1] for r in sel_])
        print(f"   {k:22s} n={len(sel_)} WER {100*e/w:5.2f}%  mean {ms:4.0f} ms")
cpu.stdin.close(); cpu.wait(5); enc.close()
