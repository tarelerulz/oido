import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import *   # CLI, MODEL, LM, KIT, OIDO_TYPE, SILERO, CLIPS_DIR, TRUTH_FILE, TESTSET, RESULTS_WER, JFK (all from environment variables, see bench/README.md)
"""Time from the END OF SPEECH until the text is READY and until it would be TYPED, for phrase-end rules 600/500/400 ms.
Replays recorded clips (+2000 ms of silence) through the live voice-detector settings in 80 ms chunks (live CHUNK=1280),
early VAD 250 ms + normal VAD S ms, real CPU engine times (menu setup: LM, 4 threads), punctuation 20 ms + typing 10 ms (measured parts).
Mirrors oido-type.py: early text is used when the early phrase matches the normal phrase, else a full pass runs at the phrase end."""
import os, glob, wave, subprocess, statistics as st, collections
import numpy as np, sherpa_onnx, ast, sys, threading, time
os.nice(0)
R=16000; CH=int(os.environ.get('CH','1280')); ROOT=CLIPS_DIR
SILERO=SILERO; PB,PA=int(0.4*R),int(0.2*R); PUNCT,TYPE=20,10
def mk(sil):
    c=sherpa_onnx.VadModelConfig(); c.silero_vad.model=SILERO; c.silero_vad.threshold=0.5; c.silero_vad.min_speech_duration=0.2
    c.silero_vad.min_silence_duration=sil; c.silero_vad.max_speech_duration=25; c.sample_rate=R
    return sherpa_onnx.VoiceActivityDetector(c,buffer_size_in_seconds=60)
def load(p):
    with wave.open(p) as w: return np.frombuffer(w.readframes(w.getnframes()),'<i2')
def last_speech(x):
    e=np.array([np.abs(x[i:i+1600]).mean() for i in range(0,len(x)-1600,1600)]); thr=max(30.0,0.05*np.percentile(e,70))
    idx=np.where(e>thr)[0]; return (idx[-1]+1)*1600 if len(idx) else len(x)

os.environ['GPUPORT_ASM']='1'; os.environ['GPUPORT_SPIN']='0'
sys.path.insert(0,KIT)
LM=LM; CLI=CLI; MODEL=MODEL
def wr(p,x):
    with wave.open(p,'wb') as f: f.setnchannels(1);f.setsampwidth(2);f.setframerate(R);f.writeframes(x.astype(np.int16).tobytes())
def spawn(cores,thr,lm):
    return subprocess.Popen(['taskset','-c',cores,CLI,MODEL]+(['--lm',LM] if lm else [])+['--serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,env={**os.environ,'TASR_THREADS':str(thr)})
def call(proc,x,wp):
    wr(wp,x); t=time.perf_counter(); proc.stdin.write(wp+'\n'); proc.stdin.flush(); proc.stdout.readline(); return (time.perf_counter()-t)*1000
KIND=sys.argv[1]
if KIND=='cpu':
    cpuA=spawn('0,1,2,3',4,True)             # menu "Oido CPU": LM, 4 threads
    eng=lambda seg: st.median(call(cpuA,seg,'/dev/shm/e18a.wav') for _ in range(3))
else:
    from gpuport.nemo8 import Nemo8Encoder
    enc=Nemo8Encoder().start()
    for t in os.listdir(f'/proc/{enc.srv.proc.pid}/task'): os.sched_setaffinity(int(t),{3})
    os.sched_setaffinity(0,{1,2,3})
    def gpu_ms(seg): t=time.perf_counter(); enc.transcribe(seg); return (time.perf_counter()-t)*1000
    if KIND=='gpu': eng=lambda seg: st.median(gpu_ms(seg) for _ in range(3))
    else:
        tail=spawn('1,2,3',3,False)
        src=open(OIDO_TYPE).read(); body=ast.parse(src).body
        lo=next(n.lineno for n in body if isinstance(n,ast.ClassDef) and n.name=='HybridTail'); hi=next(n.lineno for n in body if isinstance(n,ast.ClassDef) and n.name=='OidoCLI')
        ns=dict(os=os,np=np,RATE=R,log=lambda *a:None,threading=threading,Path=__import__('pathlib').Path)
        exec(compile(ast.Module(body=[n for n in body if lo<=n.lineno<hi],type_ignores=[]),'live','exec'),ns)
        h=ns['HybridTail'].__new__(ns['HybridTail']); h.dead=False
        class Alive:
            def poll(self): return None
        h.p=Alive()
        def hyb_ms(seg):
            cut=h.split(seg)
            if cut is None: return gpu_ms(seg)
            res={}; th=threading.Thread(target=lambda:res.setdefault('c',call(tail,seg[cut:],'/dev/shm/e18b.wav')))
            t=time.perf_counter(); th.start(); enc.transcribe(seg[:cut]); th.join(); return (time.perf_counter()-t)*1000
        call(tail,np.zeros(8000,np.int16),'/dev/shm/e18b.wav'); eng=lambda seg: st.median(hyb_ms(seg) for _ in range(3))
eng(np.zeros(16000,np.int16)) if KIND!='cpu' else None
def cut(x,a0,b0): return x[max(0,a0-PB):min(len(x),b0+PA)]
def replay(x,S):
    sil=np.zeros(2*R,np.int16); audio=np.concatenate([x,sil]); quick=mk(0.25); norm=mk(S/1000)
    qs=[]; ns=[]
    for i in range(0,len(audio),CH):
        c=audio[i:i+CH].astype(np.float32)/32768.0; seen=min(len(audio),i+CH)
        quick.accept_waveform(c); norm.accept_waveform(c)
        while not quick.empty(): qs.append((seen,quick.front.start,len(quick.front.samples))); quick.pop()
        while not norm.empty(): ns.append((seen,norm.front.start,len(norm.front.samples))); norm.pop()
    return audio,qs,ns

files=[os.path.join(ROOT,l.split('\t')[0]) for l in open(TRUTH_FILE,encoding='utf-8')]
res={s:[] for s in (600,500,400)}
for fp in files:
    x=load(fp)
    for S in res:
        audio,qs,ns=replay(x,S)
        if not ns or not qs: continue
        z=qs[-1][1]+qs[-1][2]
        seen_n,sn,nn=ns[-1]; en=sn+nn
        prev_end=(ns[-2][1]+ns[-2][2]) if len(ns)>1 else 0
        pieces=[q for q in qs if q[1]+q[2]>prev_end and q[1]>=prev_end-int(0.25*R)]
        used=False
        if pieces:
            p_start=pieces[0][1]; seen_q,sq,nq=pieces[-1]; eq=sq+nq; used=abs(eq-en)<R//4 and abs(p_start-sn)<R//4
        if used:
            e=eng(cut(audio,p_start,eq)); ready=(seen_q-z)/R*1000+e; typed=max((seen_n-z)/R*1000,ready)+PUNCT+TYPE
        else:
            e=eng(cut(audio,sn,en)); ready=(seen_n-z)/R*1000+e; typed=ready+PUNCT+TYPE
        res[S].append((ready,typed,used,e))
for S,r in res.items():
    print(f"RESULT {KIND} {S} n={len(r)} engine_mean={np.mean([x[3] for x in r]):.0f} ready_mean={np.mean([x[0] for x in r]):.0f} typed_mean={np.mean([x[1] for x in r]):.0f} typed_median={np.median([x[1] for x in r]):.0f} typed_p90={np.percentile([x[1] for x in r],90):.0f} early_used={sum(x[2] for x in r)}",flush=True)
