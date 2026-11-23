"""Build the everyday test set audio: one 16 kHz mono WAV per sentence in sentences.json, read by the heart voice (sanoTTS).
Needs a running heart-server (http://127.0.0.1:8097, model "heart") and ffmpeg. Writes clips16/sNN.wav next to this script.
The audio is not committed (the voice licence for redistributing generated audio was not checked); the sentences are."""
import base64, json, os, subprocess, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); SERVER = os.environ.get('HEART_SERVER', 'http://127.0.0.1:8097')
os.makedirs(HERE + '/tts24', exist_ok=True); os.makedirs(HERE + '/clips16', exist_ok=True)
for i, s in enumerate(json.load(open(HERE + '/sentences.json'))):
    body = {"model": "heart", "request": {"text": s['text'].replace("'", "’")}}     # typographic apostrophe: contractions are read whole
    req = urllib.request.Request(SERVER + '/v1/tasks/run', data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    open(f'{HERE}/tts24/s{i:02d}.wav', 'wb').write(base64.b64decode(json.loads(urllib.request.urlopen(req, timeout=120).read())["audio"]))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f'{HERE}/tts24/s{i:02d}.wav', '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', f'{HERE}/clips16/s{i:02d}.wav'], check=True)
print('done:', len(os.listdir(HERE + '/clips16')), 'clips')
