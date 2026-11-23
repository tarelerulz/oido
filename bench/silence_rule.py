import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import *   # CLI, MODEL, LM, KIT, OIDO_TYPE, SILERO, CLIPS_DIR, TRUTH_FILE, TESTSET, RESULTS_WER, JFK (all from environment variables, see bench/README.md)
"""How would a shorter phrase-end silence rule (500 / 400 ms instead of 600 ms) have cut the speaker's recorded phrases?
Part 1: run the live Silero VAD settings on every recorded clip, count clips that would be split into 2+ phrases.
Part 2: on the 25 labelled clips, transcribe the pieces separately (CPU engine, no LM) and score word errors vs his own text."""
import os, sys, glob, wave, re, subprocess, difflib, collections
import numpy as np, sherpa_onnx
R=16000; ROOT=CLIPS_DIR
SILERO=SILERO
PAD_B,PAD_A=int(0.4*R),int(0.2*R)
def vad(sil):
    c=sherpa_onnx.VadModelConfig(); c.silero_vad.model=SILERO; c.silero_vad.threshold=0.5
    c.silero_vad.min_speech_duration=0.2; c.silero_vad.min_silence_duration=sil; c.silero_vad.max_speech_duration=25
    c.sample_rate=R; return sherpa_onnx.VoiceActivityDetector(c,buffer_size_in_seconds=60)
def segments(x,sil):
    v=vad(sil); xf=x.astype(np.float32)/32768.0
    for i in range(0,len(xf),512): v.accept_waveform(xf[i:i+512])
    v.flush(); out=[]
    while not v.empty(): out.append((v.front.start,len(v.front.samples))); v.pop()
    return out
def load(p):
    with wave.open(p) as w: return np.frombuffer(w.readframes(w.getnframes()),'<i2')
clips=sorted(glob.glob(ROOT+'*/*.wav'))
SIL=(600,500,400)
cnt={s:collections.Counter() for s in SIL}; n_used=0
for p in clips:
    x=load(p)
    if len(x)<1.0*R: continue
    n_used+=1
    for s in SIL: cnt[s][min(len(segments(x,s/1000)),3)]+=1
print(f'PART 1: {n_used} recorded clips of 1000 ms or longer (each one phrase under the 600 ms rule)')
for s in SIL:
    c=cnt[s]; split=c[2]+c[3]
    print(f'  silence rule {s} ms: kept as one phrase {c[1]:4d} ({100*c[1]/n_used:.1f}%), split in 2 or more {split:4d} ({100*split/n_used:.1f}%), no speech found {c[0]}')
# part 2
TRUTH={}
for ln in open(TRUTH_FILE,encoding='utf-8'):
    q=ln.rstrip('\n').split('\t')
    if len(q)>2 and q[1].strip(): TRUTH[q[0]]=q[1]
norm=lambda t: re.sub(r"[^a-z' ]"," ",t.lower()).split()
def wd(a,b):
    sm=difflib.SequenceMatcher(None,a,b,autojunk=False); return len(a)+len(b)-2*sum(m.size for m in sm.get_matching_blocks())
proc=subprocess.Popen(['taskset','-c','1,2,3',CLI,MODEL,'--serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,env={**os.environ,'TASR_THREADS':'3'})
def tx(x):
    wp='/dev/shm/silence-test.wav'
    with wave.open(wp,'wb') as f: f.setnchannels(1);f.setsampwidth(2);f.setframerate(R);f.writeframes(x.astype(np.int16).tobytes())
    proc.stdin.write(wp+'\n'); proc.stdin.flush(); return proc.stdout.readline().rstrip('\n').split('\t')[-1]
err={s:0 for s in SIL}; words=0; split_clips={s:0 for s in SIL}; pieces={s:0 for s in SIL}
for rel,truth in TRUTH.items():
    x=load(ROOT+rel); tw=norm(truth); words+=len(tw)
    for s in SIL:
        segs=segments(x,s/1000); pieces[s]+=len(segs)
        if len(segs)>1: split_clips[s]+=1
        txt=[]
        for st,n in (segs or [(0,len(x))]):
            a=max(0,st-PAD_B); b=min(len(x),st+n+PAD_A); txt+=norm(tx(x[a:b]))
        err[s]+=wd(tw,txt)
proc.stdin.close(); proc.wait(5)
print(f'\nPART 2: {len(TRUTH)} labelled clips, {words} words (CPU engine, no language model, word errors vs his own text)')
for s in SIL: print(f'  silence rule {s} ms: clips split {split_clips[s]:2d} of {len(TRUTH)}, word errors {err[s]:3d} ({100*err[s]/words:.1f}%)')
