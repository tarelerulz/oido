// tasr_lmdecode: beam search with the language model over CTC scores computed elsewhere (for example on the GPU).
//
// usage: tasr_lmdecode lm.tlm [--beam 4] [--lm_weight w] [--token_bonus b]
// stdin, repeated per utterance (binary, little endian):
//   int32 T, int32 V1, then T * V1 float32 raw logits (frame by frame; the blank is the LAST column, index V1-1,
//   the same layout the nemo8 head produces).  T = 0 ends the session.
// stdout, per utterance: one line with the chosen token ids, space separated (ids index the model's token table;
// the caller maps them to text exactly like tasr_cli does).  Flushed after every line.
//
// It does the same steps as tasr_nemo_transcribe(): log-softmax per frame, blank moved to column 0, then
// tasr_decoder_step() for every frame and tasr_decoder_best() at the end, with the same defaults as tasr_cli.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinyasr.h"
#include "tinyasr_lm.h"

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc(sz > 0 ? (size_t)sz : 1);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static int read_all(void *dst, size_t n) { return fread(dst, 1, n, stdin) == n; }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s lm.tlm [--beam 4] [--lm_weight w] [--token_bonus b]\n", argv[0]); return 2; }
    int beam = 4;
    float lw = -1.f, tb = -1.f;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--beam") && i + 1 < argc) beam = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lm_weight") && i + 1 < argc) lw = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--token_bonus") && i + 1 < argc) tb = (float)atof(argv[++i]);
    }
    size_t lsz;
    uint8_t *lb = read_file(argv[1], &lsz);
    tasr_lm_t *lm = lb ? tasr_lm_load(lb, lsz) : NULL;
    if (!lm) { fprintf(stderr, "cannot load LM %s\n", argv[1]); return 1; }
    tasr_decoder_t *dec = NULL;
    int decV1 = 0;
    float *lg = NULL, *lp = NULL;
    int cap = 0;
    for (;;) {
        int32_t hdr[2];
        if (!read_all(hdr, sizeof hdr)) break;
        const int T = hdr[0], V1 = hdr[1];
        if (T <= 0) break;
        if (V1 < 2 || V1 > 1000000 || T > 100000) { fprintf(stderr, "bad header T=%d V1=%d\n", T, V1); return 1; }
        if (V1 != decV1) {       // same defaults as tasr_cli for the nemo model
            if (dec) tasr_decoder_free(dec);
            dec = tasr_decoder_create(lm, V1, beam, 6, lw < 0 ? tasr_lm_weight(lm, 0.3f) : lw, tb < 0 ? tasr_lm_bonus(lm, 0.5f) : tb);
            if (!dec) { fprintf(stderr, "decoder create failed\n"); return 1; }
            decV1 = V1;
            free(lp); lp = (float *)malloc(sizeof(float) * V1);
        }
        if (T * V1 > cap) { free(lg); cap = T * V1; lg = (float *)malloc(sizeof(float) * cap); }
        if (!lg || !lp || !read_all(lg, sizeof(float) * (size_t)T * V1)) { fprintf(stderr, "short read\n"); return 1; }
        const int V = V1 - 1;
        tasr_decoder_reset(dec);
        for (int t = 0; t < T; t++) {
            const float *l = lg + (size_t)t * V1;
            float bv = l[0];
            for (int v = 1; v < V1; v++) if (l[v] > bv) bv = l[v];
            float z = 0.f;
            for (int v = 0; v < V1; v++) z += expf(l[v] - bv);
            const float lz = bv + logf(z);
            lp[0] = l[V] - lz;
            for (int v = 0; v < V; v++) lp[v + 1] = l[v] - lz;
            tasr_decoder_step(dec, lp);
        }
        int toks[4096];
        const int nt = tasr_decoder_best(dec, toks, 4096);
        for (int i = 0; i < nt; i++) printf(i ? " %d" : "%d", toks[i]);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
