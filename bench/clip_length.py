import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import *   # CLI, MODEL, LM, KIT, OIDO_TYPE, SILERO, CLIPS_DIR, TRUTH_FILE, TESTSET, RESULTS_WER, JFK (all from environment variables, see bench/README.md)
import os, sys, time, wave, threading, subprocess, statistics as st
os.nice(19); os.sched_setaffinity(0,{1,2,3})
os.environ['GPUPORT_ASM']='1'; os.environ['GPUPORT_SPIN']='0'
sys.path.insert(0,KIT)
import numpy as np
from pathlib import Path
from gpuport.nemo8 import Nemo8Encoder
w=wave.open(JFK); jfk=np.frombuffer(w.readframes(w.getnframes()),'<i2'); R=16000
clips={'2.5s':jfk[:int(2.5*R)],'5s':jfk[:5*R],'10s':jfk[:10*R],'15s':np.concatenate([jfk,jfk])[:15*R]}
CLI=CLI; MODEL=MODEL
def wr(p,x):
    with wave.open(p,'wb') as f: f.setnchannels(1);f.setsampwidth(2);f.setframerate(R);f.writeframes(x.tobytes())
cpu=subprocess.Popen(['taskset','-c','1,2,3',CLI,MODEL,'--serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,env={**os.environ,'TASR_THREADS':'3'})
def cpu_t(x):
    wr('/tmp/b3.wav',x); t=time.perf_counter(); cpu.stdin.write('/tmp/b3.wav\n'); cpu.stdin.flush(); s=cpu.stdout.readline()
    return (time.perf_counter()-t)*1000, s.strip().split('\t')[-1]
time.sleep(.5)
enc=Nemo8Encoder().start()
for t in enc.srv.proc and Path(f'/proc/{enc.srv.proc.pid}/task').iterdir(): os.sched_setaffinity(int(t.name),{3})
cpu_t(clips['2.5s']); enc.transcribe(clips['2.5s'])
print(f"{'clip':6} {'CPU(3thr)':>10} {'GPU':>8} {'hybrid':>8}   (median of 5, ms)")
for name,x in clips.items():
    c=[];g=[];h=[]
    cut=int(len(x)*0.6)
    for _ in range(5):
        time.sleep(.3); c.append(cpu_t(x)[0])
        time.sleep(.3); t=time.perf_counter(); enc.transcribe(x); g.append((time.perf_counter()-t)*1000)
        time.sleep(.3)
        res={}; th=threading.Thread(target=lambda:res.setdefault('c',cpu_t(x[cut:])))
        t=time.perf_counter(); th.start(); enc.transcribe(x[:cut]); th.join(); h.append((time.perf_counter()-t)*1000)
    print(f"{name:6} {st.median(c):10.0f} {st.median(g):8.0f} {st.median(h):8.0f}",flush=True)
cpu.stdin.close(); cpu.wait(5); enc.close()
