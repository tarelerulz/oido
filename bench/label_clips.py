#!/usr/bin/env python3
"""Write down what recorded dictation clips really said (2026-11-23), to get ground truth for scoring the engines.
Two ways, pick one:
  label-clips          LISTEN mode (blind): plays a clip, you type what was said.
  label-clips --read   READ mode: shows what the CPU engine heard on screen, you only fix what is wrong
                       (Enter = it is correct as shown). For when listening is hard. Note: reading the engine's text
                       makes it easier to accept its mistakes, so READ answers are saved with a mark ('ok' or 'typed').
Answers are saved after every clip in ~/doc/benchmarks/clip-truth/truth.tsv  (clip<TAB>text<TAB>ok|typed). Re-run any time: it resumes.
  usage: label-clips [--read] [N]       label up to N more clips (default 25)
  at the prompt:  type the text + Enter = save     Enter alone = play again (LISTEN) / accept shown text (READ)
                  r = play it again (both modes)    + = louder (+6 dB) and replay    - = quieter    s = can't tell, skip
                  u = undo the last saved answer   q = quit
  env: CLIPS_DIR (required), LABEL_NO_PLAY=1, LABEL_OUT=/path/truth.tsv, LABEL_GAIN=dB (start boost, default 14 or last used), LABEL_AUDIO_DEVICE, ENGINE_TEXT_TSV
  READ mode needs engine-text.tsv (clip<TAB>text): make it with  label_clips.py --make-engine-text  (uses the CPU engine in --serve mode)"""
import glob, os, subprocess, sys, wave
import sys as _s; _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CLIPS_DIR, TRUTH_FILE, CLI, MODEL
ROOT = CLIPS_DIR.rstrip('/')
if not ROOT: raise SystemExit('set CLIPS_DIR to the folder with your recordings (CLIPS_DIR/<folder>/<clip>.wav)')
OUT = os.environ.get('LABEL_OUT', TRUTH_FILE)
ENG = os.environ.get('ENGINE_TEXT_TSV', os.path.join(os.path.dirname(TRUTH_FILE), 'engine-text.tsv'))
GAINF = os.path.expanduser('~/.config/label-clips-gain')   # remembered volume boost
def load_gain():
    try: return float(os.environ.get('LABEL_GAIN') or open(GAINF).read())
    except Exception: return 14.0
def clips():
    out = []
    for p in sorted(glob.glob(ROOT + '/*/*.wav')):
        with wave.open(p) as w: n = w.getnframes() / w.getframerate()
        if 4.5 <= n <= 18.5: out.append((os.path.relpath(p, ROOT), n))
    return out[::2]            # the same every-2nd subset the speed tests used
def play(path, gain):
    if os.environ.get('LABEL_NO_PLAY'): return
    af = f'lavfi=[volume={gain}dB,alimiter=limit=0.95]'     # clips are quiet (median -31 dBFS): boost, then limit so it never clips
    dev = os.environ.get('LABEL_AUDIO_DEVICE')   # e.g. alsa/plughw:CARD=Headset,DEV=0 ; unset = mpv's default output
    subprocess.run(['mpv', '--no-config', '--no-terminal', '--no-video'] + (['--audio-device=' + dev] if dev else []) + ['--af=' + af, path], timeout=60, stdin=subprocess.DEVNULL)
def make_engine_text():
    import wave as _w
    p = subprocess.Popen([CLI, MODEL, '--serve'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    rows = []
    for rel, _ in clips():
        p.stdin.write(os.path.join(ROOT, rel) + '\n'); p.stdin.flush()
        rows.append(rel + '\t' + p.stdout.readline().rstrip('\n').split('\t')[-1].strip())
    p.stdin.close(); p.wait(5)
    os.makedirs(os.path.dirname(ENG) or '.', exist_ok=True); open(ENG, 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
    print(len(rows), 'clips transcribed ->', ENG)


def main():
    if '--make-engine-text' in sys.argv[1:]: return make_engine_text()
    args = [a for a in sys.argv[1:] if a not in ('--read', '--make-engine-text')]; read = '--read' in sys.argv[1:]
    n_max = int(args[0]) if args else 25
    eng = {}
    if read:
        if not os.path.exists(ENG): sys.exit('no engine text file: ' + ENG)
        for ln in open(ENG, encoding='utf-8'):
            if '\t' in ln: k, t = ln.rstrip('\n').split('\t', 1); eng[k] = t
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = {}
    if os.path.exists(OUT):
        for ln in open(OUT, encoding='utf-8'):
            if '\t' in ln: done[ln.split('\t')[0]] = 1
    todo = [c for c in clips() if c[0] not in done]
    gain = load_gain()
    print(f'{len(done)} clips already written down, {len(todo)} left. Doing up to {n_max}. Mode: {"READ" if read else "LISTEN"}. Volume boost {gain:+.0f} dB.')
    print('Keys: Enter=' + ('accept shown text' if read else 'replay') + '  r=replay  +/- = louder/quieter  s=skip  u=undo  q=quit\n')
    for i, (rel, secs) in enumerate(todo[:n_max], 1):
        shown = eng.get(rel, '') if read else None
        first = True
        while True:
            print(f'[{i}/{min(n_max, len(todo))}] {secs:.1f} s')
            if read: print('  engine heard:  ' + shown)
            if first or not read or True: play(os.path.join(ROOT, rel), gain)
            first = False
            try: ans = input('  correct text (Enter = as shown)> ' if read else '  what was said> ').strip()
            except (EOFError, KeyboardInterrupt): print(); return
            if ans == '+' or ans == '-':
                gain += 6 if ans == '+' else -6
                os.makedirs(os.path.dirname(GAINF), exist_ok=True); open(GAINF, 'w').write(str(gain)); print(f'  volume boost now {gain:+.0f} dB'); continue
            if ans == 'r': continue          # replay (the loop plays the clip again)
            if ans == 'q': return
            if ans == 'u':
                lines = open(OUT, encoding='utf-8').read().splitlines() if os.path.exists(OUT) else []
                if lines: print('  undid:', lines[-1].split('\t')[0]); open(OUT, 'w', encoding='utf-8').write('\n'.join(lines[:-1]) + ('\n' if lines[:-1] else ''))
                continue
            if ans == '' and not read: continue
            if ans == 's': text, how = '', 'skip'
            elif ans == '' and read: text, how = shown, 'ok'
            else:
                if len(ans) < 3:      # a stray key (like r) must not be saved as the answer
                    try: ok = input(f"  save '{ans}' as the text? (y = yes, Enter = no, play again) ").strip().lower() == 'y'
                    except (EOFError, KeyboardInterrupt): print(); return
                    if not ok: continue
                text, how = ans, 'typed'
            with open(OUT, 'a', encoding='utf-8') as f: f.write(f'{rel}\t{text}\t{how}\n')
            break
    print('done for now.')
main()
