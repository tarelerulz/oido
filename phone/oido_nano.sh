#!/data/data/com.termux/files/usr/bin/bash
# Called by nano Ctrl+T with the buffer saved as $1.
# Finds [oido] in the file, records audio, transcribes, replaces marker.
export PATH=/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/bin/applets:$PATH
FILE="$1"
WAV=/data/data/com.termux/files/home/tasr_bench/oido_rec.wav
MODEL=/data/data/com.termux/files/home/tasr_bench/nemo8.tnm
LM=/sdcard/oido/models/nemo_lm.tlm
CLI=/data/data/com.termux/files/home/tasr_bench/tasr_cli_1t
PUNCT_CLI=/data/data/com.termux/files/home/punct_cli
PUNCT_MODEL=/data/data/com.termux/files/home/models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx

if ! grep -q '\[oido\]' "$FILE"; then
    exit 0
fi

pulseaudio --check 2>/dev/null || pulseaudio --start 2>/dev/null

echo "" >&2
echo "  [oido] recording 5 seconds... speak now" >&2

parecord --channels=1 --rate=16000 --format=s16le --file-format=wav "$WAV" &
REC_PID=$!
sleep 5
kill $REC_PID 2>/dev/null
wait $REC_PID 2>/dev/null

TEXT=$("$CLI" "$MODEL" --lm "$LM" "$WAV" 2>/dev/null | tail -1 | cut -f2-)
if [ -z "$TEXT" ]; then
    TEXT="[no speech]"
fi

# Punctuate via sherpa-onnx punct_cli, fall back to raw text if unavailable
if [ -x "$PUNCT_CLI" ] && [ -f "$PUNCT_MODEL" ]; then
    PUNCD=$(echo "$TEXT" | "$PUNCT_CLI" "$PUNCT_MODEL" 2>/dev/null)
    [ -n "$PUNCD" ] && TEXT="$PUNCD"
fi

python3 -c "
f, t = __import__('sys').argv[1], __import__('sys').argv[2]
c = open(f).read()
open(f, 'w').write(c.replace('[oido]', t, 1))
" "$FILE" "$TEXT"
