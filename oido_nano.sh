#!/bin/bash
# Nano Ctrl+T voice input for Pi 400. Called with buffer saved as $1.
# Finds [oido] in the file, records 5s, transcribes, replaces the marker.
FILE="$1"
WAV=/tmp/oido_rec.wav
MODEL=~/oido/models/nemo8.tnm
LM=~/oido/models/nemo_lm.tlm
CLI=~/oido/host/tasr_cli_a72

if ! grep -q '\[oido\]' "$FILE"; then
    exit 0
fi

echo "" >&2
echo "  [oido] recording 5 seconds... speak now" >&2

arecord -q -D plughw:CARD=Headset,DEV=0 -c 1 -r 16000 -f S16_LE -d 5 "$WAV"

TEXT=$("$CLI" "$MODEL" --lm "$LM" "$WAV" 2>/dev/null | tail -1 | cut -f2-)
if [ -z "$TEXT" ]; then
    TEXT="[no speech]"
fi

python3 -c "
f, t = __import__('sys').argv[1], __import__('sys').argv[2]
c = open(f).read()
open(f, 'w').write(c.replace('[oido]', t, 1))
" "$FILE" "$TEXT"
