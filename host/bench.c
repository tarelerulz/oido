#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "tasr_nemo.h"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench model.tnm\n"); return 1; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    uint8_t *blob = malloc(sz);
    fread(blob, 1, sz, f); fclose(f);

    tasr_nemo_t *m = tasr_nemo_load(blob, sz);
    if (!m) { fprintf(stderr, "model load failed\n"); return 1; }
    /* blob kept alive — nemo retains pointers into it */

    // Synthetic white-noise audio: alternating 2s, 3s, 4s clips, 10 total
    int durs[] = {2, 3, 4, 2, 3, 4, 2, 3, 4, 2};
    int n_clips = 10;
    int16_t *clips[10];
    int lens[10];
    unsigned state = 12345;
    for (int i = 0; i < n_clips; i++) {
        lens[i] = durs[i] * 16000;
        clips[i] = malloc(lens[i] * sizeof(int16_t));
        for (int j = 0; j < lens[i]; j++) {
            state = state * 1664525u + 1013904223u;
            clips[i][j] = (int16_t)((state >> 16) % 6001 - 3000);
        }
    }

    // warm-up
    char textbuf[512];
    tasr_nemo_transcribe(m, clips[0], lens[0], NULL, textbuf, sizeof(textbuf), NULL, 0, NULL);

    double total_audio = 0, total_compute = 0;
    for (int i = 0; i < n_clips; i++) {
        double t0 = now_s();
        tasr_nemo_transcribe(m, clips[i], lens[i], NULL, textbuf, sizeof(textbuf), NULL, 0, NULL);
        total_compute += now_s() - t0;
        total_audio += (double)lens[i] / 16000.0;
    }

    printf("audio=%.1fs  compute=%.3fs  RTFx=%.1fx\n",
           total_audio, total_compute, total_audio / total_compute);

    for (int i = 0; i < n_clips; i++) free(clips[i]);
    tasr_nemo_free(m);
    return 0;
}
