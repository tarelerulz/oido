#!/data/data/com.termux/files/usr/bin/bash
WAV=/data/data/com.termux/files/home/tasr_bench/apollo11_16k.wav
MODEL=/data/data/com.termux/files/home/tasr_bench/nemo8.tnm
LM=/sdcard/oido/models/nemo_lm.tlm
CLI=/data/data/com.termux/files/home/tasr_bench/tasr_cli_mt
OUT=/data/data/com.termux/files/home/tasr_bench/apollo11_transcript.txt
TMP=/data/data/com.termux/files/home/tasr_bench/chunk.wav
CHUNK=300  # 5 minutes per chunk

> "$OUT"

TOTAL=$(python3 -c "import wave; w=wave.open('$WAV'); print(w.getnframes()//w.getframerate())")
CHUNKS=$(( (TOTAL + CHUNK - 1) / CHUNK ))
echo "Total: ${TOTAL}s audio, ${CHUNKS} chunks of ${CHUNK}s"

for i in $(seq 0 $((CHUNKS - 1))); do
    START=$((i * CHUNK))
    END=$((START + CHUNK))
    [ $END -gt $TOTAL ] && END=$TOTAL
    LEN=$((END - START))
    echo -n "Chunk $((i+1))/$CHUNKS  [${START}s-${END}s]  "

    python3 - <<EOF
import wave
with wave.open('$WAV') as w:
    r = w.getframerate()
    w.setpos($START * r)
    frames = w.readframes($LEN * r)
    p = w.getparams()
with wave.open('$TMP', 'w') as o:
    o.setparams(p)
    o.writeframes(frames)
EOF

    T=$(TIMEFORMAT='%R'; { time TASR_THREADS=4 "$CLI" "$MODEL" --lm "$LM" "$TMP" >> "$OUT" 2>/dev/null; } 2>&1)
    echo "${T}s"
done

echo ""
echo "Done — transcript saved to $OUT"
echo "Lines: $(wc -l < "$OUT")"
