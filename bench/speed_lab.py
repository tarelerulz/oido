import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import *   # CLI, MODEL, LM, KIT, OIDO_TYPE, SILERO, CLIPS_DIR, TRUTH_FILE, TESTSET, RESULTS_WER, JFK (all from environment variables, see bench/README.md)
"""Speed lab on the 25 labelled clips: each variant = (threads, cores, nice, audio speed-up). Measures median wait per clip (3 passes)
and word errors vs the speaker's own text. Numbers only. CPU engine = resident tasr_cli_a72 --serve."""
import os, sys, time, wave, subprocess, re, difflib, statistics as st, tempfile
import numpy as np
R=16000; ROOT=CLIPS_DIR
CLI=CLI; MODEL=MODEL
TRUTH={}
for ln in open(TRUTH_FILE,encoding='utf-8'):
    p=ln.rstrip('\n').split('\t')
    if len(p)>2 and p[1].strip(): TRUTH[p[0]]=p[1]
norm=lambda t: re.sub(r"[^a-z' ]"," ",t.lower()).split()
def wd(a,b):
    sm=difflib.SequenceMatcher(None,a,b,autojunk=False); return len(a)+len(b)-2*sum(m.size for m in sm.get_matching_blocks())
clips={}
for rel in TRUTH:
    with wave.open(ROOT+rel) as w: clips[rel]=np.frombuffer(w.readframes(w.getnframes()),'<i2')
tmp=tempfile.mkdtemp(dir='/dev/shm')
def wr(p,x):
    with wave.open(p,'wb') as f: f.setnchannels(1);f.setsampwidth(2);f.setframerate(R);f.writeframes(x.astype(np.int16).tobytes())
def speedup(x,f,how):
    if f==1.0: return x
    if how=='resample':     # cheap: play faster (pitch goes up too), linear interpolation
        n=int(len(x)/f); return np.interp(np.arange(n)*f,np.arange(len(x)),x.astype(np.float32))
    wr(tmp+'/in.wav',x); subprocess.run(['ffmpeg','-loglevel','error','-y','-i',tmp+'/in.wav','-filter:a',f'atempo={f}',tmp+'/out.wav'],check=True)
    with wave.open(tmp+'/out.wav') as w: return np.frombuffer(w.readframes(w.getnframes()),'<i2')
def run(name,threads,cores,nice,f,how='atempo'):
    audio={rel:speedup(x,f,how) for rel,x in clips.items()}
    cmd=['taskset','-c',cores]+(['nice','-n',str(nice)] if nice is not None else [])+[CLI,MODEL,'--serve']
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,env={**os.environ,'TASR_THREADS':str(threads)})
    def go(path):
        t=time.perf_counter(); p.stdin.write(path+'\n'); p.stdin.flush(); s=p.stdout.readline(); return (time.perf_counter()-t)*1000,s.rstrip('\n').split('\t')[-1]
    wr(tmp+'/w.wav',next(iter(audio.values()))); go(tmp+'/w.wav'); go(tmp+'/w.wav')
    err=0; words=0; ms=[]
    for rel,x in audio.items():
        wr(tmp+'/a.wav',x); v=[]
        for _ in range(3): time.sleep(.05); m,txt=go(tmp+'/a.wav'); v.append(m)
        ms.append(st.median(v)); tw=norm(TRUTH[rel]); err+=wd(tw,norm(txt)); words+=len(tw)
    p.stdin.close(); p.wait(5)
    print(f"{name:34s} errors {err:3d}/{words}  WER {100*err/words:5.1f}%   mean wait {np.mean(ms):4.0f} ms  median {np.median(ms):4.0f}",flush=True)
os.nice(0)
run('base: 3 thr, cores 1-3, nice 19',3,'1,2,3',19,1.0)
run('3 thr, cores 1-3, normal priority',3,'1,2,3',None,1.0)
run('2 thr, cores 1-2, normal priority',2,'1,2',None,1.0)
run('4 thr, cores 0-3, normal priority',4,'0,1,2,3',None,1.0)
for f in (1.1,1.25,1.5): run(f'audio {f}x faster (atempo), 3 thr',3,'1,2,3',None,f)
for f in (1.1,1.25): run(f'audio {f}x faster (resample), 3 thr',3,'1,2,3',None,f,'resample')
