// ============================================================
// winograd_plonka_dct.h
// ============================================================
#ifndef WINOGRAD_PLONKA_DCT_H
#define WINOGRAD_PLONKA_DCT_H
#include <stdint.h>
// 12-bit fixed-point precision
#define SHIFT 12
#define HALF  (1 << (SHIFT - 1))
// Correct fixed-point (round-to-nearest) cos/sin * 4096
#define COS_PI4    2896   // cos(pi/4)
#define SIN_PI4    2896   // sin(pi/4)
#define COS_PI8    3784   // cos(pi/8)
#define SIN_PI8    1567   // sin(pi/8)
#define COS_PI16   4017   // cos(pi/16)
#define SIN_PI16    799   // sin(pi/16)
#define COS_3PI16  3406   // cos(3pi/16)
#define SIN_3PI16  2276   // sin(3pi/16)
#define COS_PI32   4076   // cos(pi/32)
#define SIN_PI32    401   // sin(pi/32)
#define COS_3PI32  3920   // cos(3pi/32)
#define SIN_3PI32  1189   // sin(3pi/32)
#define COS_5PI32  3612   // cos(5pi/32)
#define SIN_5PI32  1931   // sin(5pi/32)
#define COS_7PI32  3166   // cos(7pi/32)
#define SIN_7PI32  2598   // sin(7pi/32)
#define COS_PI64   4091   // cos(pi/64)
#define SIN_PI64    201   // sin(pi/64)
#define COS_3PI64  4052   // cos(3pi/64)
#define SIN_3PI64   601   // sin(3pi/64)
#define COS_5PI64  3973   // cos(5pi/64)
#define SIN_5PI64   995   // sin(5pi/64)
#define COS_7PI64  3857   // cos(7pi/64)
#define SIN_7PI64  1380   // sin(7pi/64)
#define COS_9PI64  3703   // cos(9pi/64)
#define SIN_9PI64  1751   // sin(9pi/64)
#define COS_11PI64 3513   // cos(11pi/64)
#define SIN_11PI64 2106   // sin(11pi/64)
#define COS_13PI64 3290   // cos(13pi/64)
#define SIN_13PI64 2440   // sin(13pi/64)
#define COS_15PI64 3035   // cos(15pi/64)
#define SIN_15PI64 2751   // sin(15pi/64)
// Plonka-style 3-step lifting (Givens rotation approximation)
// Exact inverse when using identical alpha + reverse order + same rounding
static inline void plonka_lift_fwd(int32_t *x, int32_t *y, int32_t cos_val, int32_t sin_val)
{
    if (sin_val == 0) return;
    int32_t alpha = (int32_t)(((int64_t)(cos_val - (1 << SHIFT)) << SHIFT) / sin_val);
    *x += (int32_t)(((int64_t)*y * alpha + HALF) >> SHIFT);
    *y += (int32_t)(((int64_t)*x * sin_val + HALF) >> SHIFT);
    *x += (int32_t)(((int64_t)*y * alpha + HALF) >> SHIFT);
}
static inline void plonka_lift_inv(int32_t *x, int32_t *y, int32_t cos_val, int32_t sin_val)
{
    if (sin_val == 0) return;
    int32_t alpha = (int32_t)(((int64_t)(cos_val - (1 << SHIFT)) << SHIFT) / sin_val);
    *x -= (int32_t)(((int64_t)*y * alpha + HALF) >> SHIFT);
    *y -= (int32_t)(((int64_t)*x * sin_val + HALF) >> SHIFT);
    *x -= (int32_t)(((int64_t)*y * alpha + HALF) >> SHIFT);
}
void wp_dct2 (const int32_t in[2],  int32_t out[2]);
void wp_idct2(const int32_t in[2],  int32_t out[2]);
void wp_dct4 (const int32_t in[4],  int32_t out[4]);
void wp_idct4(const int32_t in[4],  int32_t out[4]);
void wp_dct8 (const int32_t in[8],  int32_t out[8]);
void wp_idct8(const int32_t in[8],  int32_t out[8]);
void wp_dct16(const int32_t in[16], int32_t out[16]);
void wp_idct16(const int32_t in[16], int32_t out[16]);
void wp_dct32(const int32_t in[32], int32_t out[32]);
void wp_idct32(const int32_t in[32], int32_t out[32]);
#endif
// ============================================================
// winograd_plonka_dct.c
// ============================================================
///// #include "winograd_plonka_dct.h"
// ==========================================
// 2-point (multiplication-free core)
// ==========================================
void wp_dct2(const int32_t in[2], int32_t out[2])
{
    out[0] = in[0] + in[1];
    out[1] = in[0] - in[1];
}
void wp_idct2(const int32_t in[2], int32_t out[2])
{
    out[0] = (in[0] + in[1]) >> 1;
    out[1] = (in[0] - in[1]) >> 1;
}
// ==========================================
// 4-point
// ==========================================
void wp_dct4(const int32_t in[4], int32_t out[4])
{
    int32_t e0 = in[0] + in[3];
    int32_t o0 = in[0] - in[3];
    int32_t e1 = in[1] + in[2];
    int32_t o1 = in[1] - in[2];
    out[0] = e0 + e1;
    out[2] = e0 - e1;
    plonka_lift_fwd(&o0, &o1, COS_PI4, SIN_PI4);
    out[1] = o0 + o1;
    out[3] = o0 - o1;
}
void wp_idct4(const int32_t in[4], int32_t out[4])
{
    int32_t o0 = (in[1] + in[3]) >> 1;
    int32_t o1 = (in[1] - in[3]) >> 1;
    plonka_lift_inv(&o0, &o1, COS_PI4, SIN_PI4);
    int32_t e0 = (in[0] + in[2]) >> 1;
    int32_t e1 = (in[0] - in[2]) >> 1;
    out[0] = (e0 + o0) >> 1;
    out[3] = (e0 - o0) >> 1;
    out[1] = (e1 + o1) >> 1;
    out[2] = (e1 - o1) >> 1;
}
// ==========================================
// 8-point
// ==========================================
void wp_dct8(const int32_t in[8], int32_t out[8])
{
    int32_t e[4], o[4];
    for (int i = 0; i < 4; i++) {
        e[i] = in[i] + in[7 - i];
        o[i] = in[i] - in[7 - i];
    }
    int32_t even_out[4];
    wp_dct4(e, even_out);
    out[0] = even_out[0];
    out[2] = even_out[1];
    out[4] = even_out[2];
    out[6] = even_out[3];
    plonka_lift_fwd(&o[0], &o[3], COS_PI16,  SIN_PI16);
    plonka_lift_fwd(&o[1], &o[2], COS_3PI16, SIN_3PI16);
    out[1] = o[0] + o[1];
    out[7] = o[0] - o[1];
    out[3] = o[3] + o[2];
    out[5] = o[3] - o[2];
}
void wp_idct8(const int32_t in[8], int32_t out[8])
{
    int32_t o[4];
    o[0] = (in[1] + in[7]) >> 1;
    o[1] = (in[1] - in[7]) >> 1;
    o[3] = (in[3] + in[5]) >> 1;
    o[2] = (in[3] - in[5]) >> 1;
    plonka_lift_inv(&o[1], &o[2], COS_3PI16, SIN_3PI16);
    plonka_lift_inv(&o[0], &o[3], COS_PI16,  SIN_PI16);
    int32_t even_in[4] = { in[0], in[2], in[4], in[6] };
    int32_t e[4];
    wp_idct4(even_in, e);
    for (int i = 0; i < 4; i++) {
        out[i]     = (e[i] + o[i]) >> 1;
        out[7 - i] = (e[i] - o[i]) >> 1;
    }
}
// ==========================================
// 16-point
// ==========================================
void wp_dct16(const int32_t in[16], int32_t out[16])
{
    int32_t e[8], o[8];
    for (int i = 0; i < 8; i++) {
        e[i] = in[i] + in[15 - i];
        o[i] = in[i] - in[15 - i];
    }
    int32_t even_out[8];
    wp_dct8(e, even_out);
    for (int i = 0; i < 8; i++)
        out[2 * i] = even_out[i];
    plonka_lift_fwd(&o[0], &o[7], COS_PI32,  SIN_PI32);
    plonka_lift_fwd(&o[1], &o[6], COS_3PI32, SIN_3PI32);
    plonka_lift_fwd(&o[2], &o[5], COS_5PI32, SIN_5PI32);
    plonka_lift_fwd(&o[3], &o[4], COS_7PI32, SIN_7PI32);
    for (int i = 0; i < 4; i++) {
        out[2 * i + 1]  = o[i] + o[7 - i];
        out[15 - 2 * i] = o[i] - o[7 - i];
    }
}
void wp_idct16(const int32_t in[16], int32_t out[16])
{
    int32_t o[8];
    for (int i = 0; i < 4; i++) {
        o[i]     = (in[2 * i + 1] + in[15 - 2 * i]) >> 1;
        o[7 - i] = (in[2 * i + 1] - in[15 - 2 * i]) >> 1;
    }
    plonka_lift_inv(&o[3], &o[4], COS_7PI32, SIN_7PI32);
    plonka_lift_inv(&o[2], &o[5], COS_5PI32, SIN_5PI32);
    plonka_lift_inv(&o[1], &o[6], COS_3PI32, SIN_3PI32);
    plonka_lift_inv(&o[0], &o[7], COS_PI32,  SIN_PI32);
    int32_t even_in[8];
    for (int i = 0; i < 8; i++)
        even_in[i] = in[2 * i];
    int32_t e[8];
    wp_idct8(even_in, e);
    for (int i = 0; i < 8; i++) {
        out[i]      = (e[i] + o[i]) >> 1;
        out[15 - i] = (e[i] - o[i]) >> 1;
    }
}
// ==========================================
// 32-point
// ==========================================
void wp_dct32(const int32_t in[32], int32_t out[32])
{
    int32_t e[16], o[16];
    for (int i = 0; i < 16; i++) {
        e[i] = in[i] + in[31 - i];
        o[i] = in[i] - in[31 - i];
    }
    int32_t even_out[16];
    wp_dct16(e, even_out);
    for (int i = 0; i < 16; i++)
        out[2 * i] = even_out[i];
    plonka_lift_fwd(&o[0],  &o[15], COS_PI64,   SIN_PI64);
    plonka_lift_fwd(&o[1],  &o[14], COS_3PI64,  SIN_3PI64);
    plonka_lift_fwd(&o[2],  &o[13], COS_5PI64,  SIN_5PI64);
    plonka_lift_fwd(&o[3],  &o[12], COS_7PI64,  SIN_7PI64);
    plonka_lift_fwd(&o[4],  &o[11], COS_9PI64,  SIN_9PI64);
    plonka_lift_fwd(&o[5],  &o[10], COS_11PI64, SIN_11PI64);
    plonka_lift_fwd(&o[6],  &o[9],  COS_13PI64, SIN_13PI64);
    plonka_lift_fwd(&o[7],  &o[8],  COS_15PI64, SIN_15PI64);
    for (int i = 0; i < 8; i++) {
        out[2 * i + 1]  = o[i] + o[15 - i];
        out[31 - 2 * i] = o[i] - o[15 - i];
    }
}
void wp_idct32(const int32_t in[32], int32_t out[32])
{
    int32_t o[16];
    for (int i = 0; i < 8; i++) {
        o[i]      = (in[2 * i + 1] + in[31 - 2 * i]) >> 1;
        o[15 - i] = (in[2 * i + 1] - in[31 - 2 * i]) >> 1;
    }
    plonka_lift_inv(&o[7],  &o[8],  COS_15PI64, SIN_15PI64);
    plonka_lift_inv(&o[6],  &o[9],  COS_13PI64, SIN_13PI64);
    plonka_lift_inv(&o[5],  &o[10], COS_11PI64, SIN_11PI64);
    plonka_lift_inv(&o[4],  &o[11], COS_9PI64,  SIN_9PI64);
    plonka_lift_inv(&o[3],  &o[12], COS_7PI64,  SIN_7PI64);
    plonka_lift_inv(&o[2],  &o[13], COS_5PI64,  SIN_5PI64);
    plonka_lift_inv(&o[1],  &o[14], COS_3PI64,  SIN_3PI64);
    plonka_lift_inv(&o[0],  &o[15], COS_PI64,   SIN_PI64);
    int32_t even_in[16];
    for (int i = 0; i < 16; i++)
        even_in[i] = in[2 * i];
    int32_t e[16];
    wp_idct16(even_in, e);
    for (int i = 0; i < 16; i++) {
        out[i]      = (e[i] + o[i]) >> 1;
        out[31 - i] = (e[i] - o[i]) >> 1;
    }
}
// ============================================================
// main.c  (bit-exact teszt)
// ============================================================
#include <stdio.h>
#include <stdlib.h>
///// #include "winograd_plonka_dct.h"
void run_test(int size,
              void (*dct)(const int32_t*, int32_t*),
              void (*idct)(const int32_t*, int32_t*))
{
    int32_t *orig  = (int32_t*)malloc(size * sizeof(int32_t));
    int32_t *freq  = (int32_t*)malloc(size * sizeof(int32_t));
    int32_t *recon = (int32_t*)malloc(size * sizeof(int32_t));
    for (int i = 0; i < size; i++)
        orig[i] = (rand() % 256) - 128;
    dct(orig, freq);
    idct(freq, recon);
    int success = 1;
    for (int i = 0; i < size; i++) {
        if (orig[i] != recon[i]) {
            success = 0;
            break;
        }
    }
    printf("[%2d-pontos] %s\n", size,
           success ? "SIKERES (Bit-Exact)" : "HIBA");
    free(orig);
    free(freq);
    free(recon);
}
int main(void)
{
    srand(42);
    printf("=== Winograd-Plonka Integer DCT 2..32 bit-exact teszt ===\n");
    run_test(2,  wp_dct2,  wp_idct2);
    run_test(4,  wp_dct4,  wp_idct4);
    run_test(8,  wp_dct8,  wp_idct8);
    run_test(16, wp_dct16, wp_idct16);
    run_test(32, wp_dct32, wp_idct32);
    return 0;
}  