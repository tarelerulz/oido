SRC = ../components/tinyasr/tinyasr.c ../components/tinyasr/kernels.c ../components/tinyasr/tinyasr_lm.c ../components/tinyasr/tasr_nemo.c ../components/tinyasr/tasr_seg.c
CFLAGS = -O3 -mcpu=native -pthread -DTASR_NW=8 -std=c11 -D_DEFAULT_SOURCE -ffp-contract=off -I../components/tinyasr/include -I../components/tinyasr -Wall -Wno-unused-function
all: libtinyasr.dylib tasr_cli seg_test
HDR = ../components/tinyasr/include/tinyasr.h ../components/tinyasr/include/tasr_nemo.h ../components/tinyasr/kernels.h
libtinyasr.dylib: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $(SRC) -lm
clean:
	rm -f libtinyasr.dylib

tasr_cli: tasr_cli.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ tasr_cli.c $(SRC) -lm

seg_test: seg_test.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ seg_test.c $(SRC) -lm

oled_preview: oled_preview.c ../firmware/main/oled_render.c ../firmware/main/oled.h ../firmware/main/font5x7.h
	$(CC) -O2 -std=c11 -I../firmware/main oled_preview.c ../firmware/main/oled_render.c -o $@

# Pi 400 (Cortex-A72, ARMv8.0 NEON only). Build on Pi: make tasr_cli_a72
# Also usable on phone to test NEON path: forces armv8-a so DOTPROD is disabled.
PI_FLAGS = -O3 -march=armv8-a -mtune=cortex-a72 -pthread -DTASR_NW=8 -std=c11 -D_DEFAULT_SOURCE -ffp-contract=off -funroll-loops -I../components/tinyasr/include -I../components/tinyasr -Wall -Wno-unused-function
tasr_cli_a72: tasr_cli.c $(SRC) $(HDR)
	$(CC) $(PI_FLAGS) -o $@ tasr_cli.c $(SRC) -lm

# Android/phone (ARMv8.2-A + DOTPROD, any phone ~2019+). Build in Termux: make tasr_cli_phone
PHONE_FLAGS = -O3 -march=armv8.2-a+dotprod -pthread -DTASR_NW=8 -std=c11 -D_DEFAULT_SOURCE -ffp-contract=off -funroll-loops -I../components/tinyasr/include -I../components/tinyasr -Wall -Wno-unused-function
tasr_cli_phone: tasr_cli.c $(SRC) $(HDR)
	$(CC) $(PHONE_FLAGS) -o $@ tasr_cli.c $(SRC) -lm
