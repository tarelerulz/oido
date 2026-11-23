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


MINCHUNK=float(os.environ.get('STREAM_MINCHUNK','3.0'))
QUIET_RUN=int(os.environ.get('STREAM_QUIET_RUN','3'))
STEP=1600
import glob as _g
TRUTH={}; MARK={}
for ln in open(TRUTH_FILE,encoding='utf-8'):
    p=ln.rstrip('\n').split('\t')
    if len(p)>2 and p[1].strip(): TRUTH[p[0]]=p[1]; MARK[p[0]]=p[2]
rec=[]
for rel in TRUTH:
    with wave.open(CLIPS_DIR+rel) as w: x=np.frombuffer(w.readframes(w.getnframes()),'<i2')
    rec.append((len(x)/R,x,rel))
print('recorded clips used',len(rec),'audio minutes',round(sum(d for d,_,_ in rec)/60,1),flush=True)
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


rows=[]
def gpu_full(x):
    t=time.perf_counter(); txt=enc.transcribe(x); return (time.perf_counter()-t)*1000,txt
for d,x,rel in rec:
    c_ms,c_txt=cpu_t(x); time.sleep(.2); g_ms,g_txt=gpu_full(x)
    cut=h.split(x)
    if cut is not None:
        time.sleep(.2); res={}; th=threading.Thread(target=lambda:res.setdefault('c',cpu_t(x[cut:])))
        t=time.perf_counter(); th.start(); ht=enc.transcribe(x[:cut]); th.join(); a_ms=(time.perf_counter()-t)*1000
        a_words=norm(ht)+norm(res['c'][1])
    else: a_ms=g_ms; a_words=norm(g_txt)
    time.sleep(.3); s_ms,s_words,nc=stream_item(x)
    tw=norm(TRUTH[rel]); n=len(tw)
    r=dict(rel=rel,mark=MARK[rel],d=d,n=n,ms=dict(cpu=c_ms,gpu=g_ms,live=a_ms,stream=s_ms),chunks=nc,cut=cut is not None,
           err=dict(cpu=wd(tw,norm(c_txt)),gpu=wd(tw,norm(g_txt)),live=wd(tw,a_words),stream=wd(tw,s_words)))
    rows.append(r); print(f"{d:5.1f}s {n:3d}w {r['mark']:5s} chunks {nc} hybridcut {int(r['cut'])} | errors cpu {r['err']['cpu']} gpu {r['err']['gpu']} live {r['err']['live']} stream {r['err']['stream']} | ms cpu {c_ms:.0f} gpu {g_ms:.0f} live {a_ms:.0f} stream {s_ms:.0f}",flush=True)
def rep(sel,title):
    W=sum(r['n'] for r in sel)
    print(f'\n{title}: {len(sel)} clips, {W} words')
    for k in ('cpu','gpu','live','stream'):
        e=sum(r['err'][k] for r in sel); print(f"   {k:7s} word errors {e:3d}  WER {100*e/W:5.1f}%   mean wait {np.mean([r['ms'][k] for r in sel]):4.0f} ms")
print('\nSCORES vs the speaker\'s own text (case/punctuation ignored)')
rep(rows,'ALL')
rep([r for r in rows if r['mark']=='ok'],'clips you accepted as shown (CPU text = truth, so CPU is 0 by construction)')
rep([r for r in rows if r['mark']=='typed'],'clips you corrected')
print('\nhybrid found a cut in',sum(r['cut'] for r in rows),'of',len(rows),'clips; streaming head chunks:',dict(collections.Counter(r['chunks'] for r in rows)))
cpu.stdin.close(); cpu.wait(5); enc.close()
