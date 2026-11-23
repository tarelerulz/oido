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

import re, difflib, queue, collections
cpu_t(sel[0][2]); enc.transcribe(sel[0][2])
def norm(t): return re.sub(r"[^a-z' ]"," ",t.lower()).split()
def wd(a,b):
    sm=difflib.SequenceMatcher(None,a,b); return len(a)+len(b)-2*sum(m.size for m in sm.get_matching_blocks())

MINCHUNK=float(os.environ.get('STREAM_MINCHUNK','3.0'))   # shortest chunk sent to the GPU, seconds
QUIET_RUN=int(os.environ.get('STREAM_QUIET_RUN','3'))     # consecutive quiet 100 ms windows = a real pause
STEP=1600                                                  # 100 ms of audio per tick
base={}
for ln in open(RESULTS_WER):
    m=re.match(r"\s*([\d.]+)s\s+(\d+)w (.*)",ln)
    if m and 'err/' in ln:
        d={}
        for k,e,t in re.findall(r"(GPU only|CPU only|A live):(\d+)err/(\d+)ms",m.group(3)): d[k]=(int(e),int(t))
        base[len(base)]=d

def stream_item(x):
    """Feed x at real-time speed. Chunks ending at a real pause go to a GPU worker while the speaker is still 'talking';
    when the audio ends the rest goes to the CPU engine. Returns (latency_ms after the audio ends, words, n_chunks)."""
    n=len(x); jobs=queue.Queue(); outs={}; 
    def worker():
        while True:
            it=jobs.get()
            if it is None: return
            i,a,b=it; outs[i]=enc.transcribe(x[a:b])
    gt=threading.Thread(target=worker,daemon=True); gt.start()
    e=[]; last=0; nchunk=0; quiet=0
    t0=time.perf_counter(); k=0
    while (k+1)*STEP<=n:
        k+=1
        delay=t0+k*STEP/R-time.perf_counter()
        if delay>0: time.sleep(delay)
        seg=np.abs(x[(k-1)*STEP:k*STEP].astype(np.float32)); e.append(float(seg.mean()))
        thr=max(30.0,0.05*float(np.percentile(e,70)))
        quiet=quiet+1 if e[-1]<=thr else 0
        end_s=k*STEP; mid=end_s-int(QUIET_RUN*STEP/2)
        if quiet>=QUIET_RUN and mid-last>=int(MINCHUNK*R) and n-end_s>=int(1.0*R):   # leave >= 1 s for the final piece
            jobs.put((nchunk,last,mid)); nchunk+=1; last=mid; quiet=0
    # the audio has ended (remaining samples < one tick are part of the tail)
    t_end=t0+n/R
    delay=t_end-time.perf_counter()
    if delay>0: time.sleep(delay)
    res={}
    ct=threading.Thread(target=lambda:res.setdefault('c',cpu_t(x[last:])))
    ct.start(); jobs.put(None); gt.join(); ct.join()
    lat=(time.perf_counter()-t_end)*1000
    words=[]
    for i in range(nchunk): words+=norm(outs[i])
    words+=norm(res['c'][1])
    return lat,words,nchunk

sub=[(i,it) for i,it in enumerate(items) if i%2==0]
print('subset items',len(sub),'est. minutes',round(sum(it[0] for _,it in sub)/60+len(sub)*1.0/60),flush=True)
tot=collections.defaultdict(lambda:[0,0.0]); nref=0; per=[]
for i,(d,ref,x) in sub:
    rw=norm(ref); nref+=len(rw)
    lat,words,nc=stream_item(x)
    row={'stream':(wd(rw,words),lat),'CPU only':base[i]['CPU only'],'A live':base[i]['A live'],'GPU only':base[i]['GPU only']}
    per.append((d,len(rw),row,nc))
    for k,(er,ms) in row.items(): tot[k][0]+=er; tot[k][1]+=ms
    print(f"{d:5.1f}s chunks {nc}  "+'  '.join(f"{k}:{er}/{ms:.0f}" for k,(er,ms) in row.items()),flush=True)
print('\nWER (errors / %d words) and mean ms after the audio ends:'%nref)
for k,(er,ms) in tot.items(): print(f"{k:10s} errors {er:4d}  WER {100*er/nref:5.2f}%  mean {ms/len(sub):5.0f} ms")
for lo,hi in ((4.5,9),(9,19)):
    print(f'-- clips {lo}-{hi} s:')
    for k in tot:
        s_=[r for r in per if lo<=r[0]<hi]; er=sum(r[2][k][0] for r in s_); w=sum(r[1] for r in s_); ms=np.mean([r[2][k][1] for r in s_])
        print(f"   {k:10s} n={len(s_)} WER {100*er/w:5.2f}%  mean {ms:4.0f} ms")
print('chunks per clip:',collections.Counter(r[3] for r in per))
cpu.stdin.close(); cpu.wait(5); enc.close()
