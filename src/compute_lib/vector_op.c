
#include "compute_lib.h"


// x[i] = 0
void vm_clear_f32(float *x, int num)
{
    uint32_t *p = (void*)x;
#ifdef __AVX2__
    const __m256i Z = _mm256_setzero_si256();

    while (num >= 32) {
        _mm256_storeu_si256((void*)(p + 0), Z);
        _mm256_storeu_si256((void*)(p + 8), Z);
        _mm256_storeu_si256((void*)(p + 16), Z);
        _mm256_storeu_si256((void*)(p + 24), Z);
        p += 32; num -= 32;
    }
#endif
    for (int i=0; i<num; i++) {
        p[i] = 0;
    }
}


// y[i] = x[i]
void vm_copy_f32(float *x, float *y, int num)
{
    if (x == y) return;

#ifdef __AVX2__
    while (num >= 32) {
        __m256 X0 = _mm256_loadu_ps(x + 0);
        __m256 X1 = _mm256_loadu_ps(x + 8);
        __m256 X2 = _mm256_loadu_ps(x + 16);
        __m256 X3 = _mm256_loadu_ps(x + 24);
        _mm256_storeu_ps(y + 0, X0);
        _mm256_storeu_ps(y + 8, X1);
        _mm256_storeu_ps(y + 16, X2);
        _mm256_storeu_ps(y + 24, X3);
        x += 32; y += 32; num -= 32;
    }
#endif
    for (int i=0; i<num; i++) {
        y[i] = x[i];
    }
}


// z[i] = x[i] + y[i]
void vm_add_f32(float *x, float *y, float *z, int num)
{
#ifdef __AVX2__
    while (num >= 16) {
        __m256 X0 = _mm256_loadu_ps(x + 0);
        __m256 X1 = _mm256_loadu_ps(x + 8);
        __m256 Y0 = _mm256_loadu_ps(y + 0);
        __m256 Y1 = _mm256_loadu_ps(y + 8);
        _mm256_storeu_ps(z + 0, X0 + Y0);
        _mm256_storeu_ps(z + 8, X1 + Y1);
        x += 16; y += 16; z += 16; num -= 16;
    }
#endif
    for (int i=0; i<num; i++) {
        z[i] = x[i] + y[i];
    }
}


// z[i] = x[i] * y[i]
void vm_mul_f32(float *x, float *y, float *z, int num)
{
#ifdef __AVX2__
    while (num >= 16) {
        __m256 X0 = _mm256_loadu_ps(x + 0);
        __m256 X1 = _mm256_loadu_ps(x + 8);
        __m256 Y0 = _mm256_loadu_ps(y + 0);
        __m256 Y1 = _mm256_loadu_ps(y + 8);
        _mm256_storeu_ps(z + 0, X0 * Y0);
        _mm256_storeu_ps(z + 8, X1 * Y1);
        x += 16; y += 16; z += 16; num -= 16;
    }
#endif
    for (int i=0; i<num; i++) {
        z[i] = x[i] * y[i];
    }
}


// y[i] = max(x[i], 0)
void vm_relu_f32(float *x, float *y, int num)
{
#ifdef __AVX2__
    __m256 Z = _mm256_setzero_ps();
    while (num >= 8) {
        __m256 X0 = _mm256_loadu_ps(x);
        __m256 Y0 = _mm256_max_ps(X0, Z);
        _mm256_storeu_ps(y, Y0);
        x += 8; y += 8; num -= 8;
    }
#endif
    for (int i=0; i<num; i++) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}



