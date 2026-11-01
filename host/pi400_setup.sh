#!/bin/bash
# Run this on the Pi 400 after copying files: bash ~/oido/pi400_setup.sh
set -e

OIDO=~/oido

echo "=== Building tasr_cli for Pi 400 (Cortex-A72 NEON) ==="
cd "$OIDO/host"
make tasr_cli_a72
echo "Build OK: $OIDO/host/tasr_cli_a72"

echo ""
echo "=== Installing oido_nano.sh ==="
cp "$OIDO/oido_nano.sh" ~/oido_nano.sh
chmod +x ~/oido_nano.sh

echo ""
echo "=== Done ==="
echo "Add this to ~/.nanorc to enable Ctrl+T voice input:"
echo "  set speller \"~/oido_nano.sh\""
echo ""
echo "Test: tasr_cli_a72 is at $OIDO/host/tasr_cli_a72"
echo "Models: $OIDO/models/nemo8.tnm  $OIDO/models/nemo_lm.tlm"
