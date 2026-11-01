#!/usr/bin/env python3
import http.server, struct, subprocess, os, tempfile, threading

MODEL       = os.path.expanduser("~/tasr_bench/nemo8.tnm")
LM          = "/sdcard/oido/models/nemo_lm.tlm"
CLI         = os.path.expanduser("~/tasr_bench/tasr_cli_mt")
PUNCT_CLI   = os.path.expanduser("~/punct_cli")
PUNCT_MODEL = os.path.expanduser("~/models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx")

# persistent punct subprocess — loaded once, reused for every request
_punct_proc = None
_punct_lock = threading.Lock()

def get_punct_proc():
    global _punct_proc
    if _punct_proc is None or _punct_proc.poll() is not None:
        _punct_proc = subprocess.Popen(
            [PUNCT_CLI, PUNCT_MODEL],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1
        )
    return _punct_proc

HTML = b"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Email &mdash; Oido</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: sans-serif;
    margin: 0; padding: 12px;
    background: #111; color: #eee;
    min-height: 100vh;
  }
  h2 { color: #7cf; margin: 0 0 14px; font-size: 1.2em; }

  label { display: block; color: #aaa; font-size: 0.82em; margin: 10px 0 3px; }

  input[type=text], textarea {
    width: 100%;
    background: #1e1e1e; color: #eee;
    border: 1px solid #444; border-radius: 8px;
    padding: 10px 12px; font-size: 1em;
    outline: none;
  }
  input[type=text]:focus, textarea:focus { border-color: #7cf; }

  textarea {
    resize: none;
    overflow: hidden;
    min-height: 120px;
    line-height: 1.5;
  }

  .row { display: flex; gap: 8px; margin-top: 14px; }

  button {
    flex: 1; padding: 16px 8px; font-size: 1em;
    border: none; border-radius: 10px; cursor: pointer;
    font-weight: bold;
  }
  #btn-rec  { background: #2a6; color: #fff; }
  #btn-rec.recording { background: #c33; }
  #btn-send  { background: #257; color: #fff; }
  #btn-punct { background: #630; color: #fff; }
  #btn-clear { background: #444; color: #ccc; flex: 0 0 auto; padding: 16px 14px; }

  #status {
    margin-top: 10px; min-height: 1.3em;
    font-size: 0.88em; color: #aaa; text-align: center;
  }
</style>
</head>
<body>
<h2>&#9993; Compose Email</h2>

<label>To</label>
<input type="text" id="to" placeholder="recipient@example.com">

<label>Subject</label>
<input type="text" id="subject" placeholder="Subject">

<label>Body</label>
<textarea id="body" placeholder="Tap record to dictate, or type here..."></textarea>

<div class="row">
  <button id="btn-rec" onclick="toggleRec()">&#127908; Record</button>
  <button id="btn-punct" onclick="fixPunct()">&#10003; Punctuate</button>
  <button id="btn-send" onclick="sendEmail()">&#9993; Gmail</button>
  <button id="btn-clear" onclick="clearBody()" title="Clear body">&#10005;</button>
</div>
<div id="status">Ready</div>

<script>
let ctx, src, proc, chunks = [], recording = false;

// Auto-expand textarea as content grows
const body = document.getElementById('body');
function grow() {
  body.style.height = 'auto';
  body.style.height = body.scrollHeight + 'px';
}
body.addEventListener('input', grow);

async function toggleRec() {
  if (!recording) await startRec(); else stopRec();
}

async function startRec() {
  recording = true;
  chunks = [];
  document.getElementById('btn-rec').textContent = '\\u23F9 Stop';
  document.getElementById('btn-rec').classList.add('recording');
  document.getElementById('status').textContent = 'Listening\\u2026';

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  ctx = new AudioContext({ sampleRate: 16000 });
  src = ctx.createMediaStreamSource(stream);
  proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = e => {
    const f32 = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(f32));
  };
  src.connect(proc);
  proc.connect(ctx.destination);
}

function stopRec() {
  recording = false;
  document.getElementById('btn-rec').textContent = '\\u{1F3A4} Record';
  document.getElementById('btn-rec').classList.remove('recording');
  document.getElementById('status').textContent = 'Transcribing\\u2026';

  proc.disconnect(); src.disconnect();
  ctx.close();

  const total = chunks.reduce((n, c) => n + c.length, 0);
  const pcm = new Int16Array(total);
  let i = 0;
  for (const c of chunks)
    for (let j = 0; j < c.length; j++)
      pcm[i++] = Math.max(-32768, Math.min(32767, c[j] * 32768));

  fetch('/transcribe', {
    method: 'POST',
    body: pcm.buffer,
    headers: { 'Content-Type': 'application/octet-stream' }
  })
  .then(r => r.text())
  .then(t => {
    if (t && t !== '[no speech]') {
      const el = document.getElementById('body');
      const cur = el.value;
      // append with a space if body already has text
      el.value = cur ? cur.trimEnd() + ' ' + t : t;
      grow();
    }
    document.getElementById('status').textContent =
      t === '[no speech]' ? 'No speech detected' : 'Added: \\u201c' + t.slice(0, 60) + (t.length > 60 ? '\\u2026' : '') + '\\u201d';
  })
  .catch(e => {
    document.getElementById('status').textContent = 'Error: ' + e;
  });
}

function fixPunct() {
  const el = document.getElementById('body');
  const txt = el.value.trim();
  if (!txt) return;
  document.getElementById('btn-punct').disabled = true;
  document.getElementById('btn-punct').textContent = 'Working...';
  document.getElementById('status').textContent = 'Adding punctuation\\u2026';
  fetch('/punct', {
    method: 'POST',
    body: txt,
    headers: { 'Content-Type': 'text/plain; charset=utf-8' }
  })
  .then(r => r.text())
  .then(t => {
    el.value = t;
    grow();
    document.getElementById('status').textContent = 'Punctuation added';
  })
  .catch(e => {
    document.getElementById('status').textContent = 'Error: ' + e;
  })
  .finally(() => {
    document.getElementById('btn-punct').disabled = false;
    document.getElementById('btn-punct').textContent = '\\u2713 Punctuate';
  });
}

function clearBody() {
  document.getElementById('body').value = '';
  grow();
  document.getElementById('status').textContent = 'Body cleared';
}

function sendEmail() {
  const to      = document.getElementById('to').value.trim();
  const subject = document.getElementById('subject').value.trim();
  const bodyTxt = document.getElementById('body').value.trim();

  if (!to) {
    document.getElementById('status').textContent = 'Please fill in To field';
    return;
  }

  const url = 'mailto:' + encodeURIComponent(to)
    + '?subject=' + encodeURIComponent(subject)
    + '&body='    + encodeURIComponent(bodyTxt);

  window.location.href = url;
  document.getElementById('status').textContent = 'Opening Gmail\\u2026';
}
</script>
</body>
</html>"""

def add_punctuation(text):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    results = []
    with _punct_lock:
        proc = get_punct_proc()
        for p in paragraphs:
            line = ' '.join(p.split())
            proc.stdin.write(line + '\n')
            proc.stdin.flush()
            result = proc.stdout.readline().rstrip('\n')
            results.append(result if result.strip() else p)
    return '\n'.join(results)

def wav_wrap(pcm_bytes, rate=16000, ch=1, bits=16):
    n = len(pcm_bytes)
    hdr = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36+n, b'WAVE', b'fmt ', 16,
        1, ch, rate, rate*ch*bits//8, ch*bits//8, bits,
        b'data', n)
    return hdr + pcm_bytes

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML)

    def do_POST(self):
        if self.path == '/punct':
            text = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode()
            result = add_punctuation(text)
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(result.encode())
            return
        pcm = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_wrap(pcm)); tmp = f.name
        try:
            out = subprocess.run([CLI, MODEL, "--lm", LM, tmp], capture_output=True, text=True)
            line = out.stdout.strip().split('\n')[-1] if out.stdout.strip() else ''
            text = line.split('\t')[-1] if '\t' in line else line or '[no speech]'
        finally:
            os.unlink(tmp)
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(text.encode())

    def log_message(self, fmt, *args):
        print(fmt % args)

print("Oido email composer on http://localhost:5000")
http.server.HTTPServer(('0.0.0.0', 5000), H).serve_forever()
