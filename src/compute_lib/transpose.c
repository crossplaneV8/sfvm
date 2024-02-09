
#include "compute_lib.h"


// y[i] = x[i]
static inline void _cp_f32(float *x, float *y, int num)
{
#ifdef __AVX2__
    while (num >= 8) {
        _mm256_storeu_ps(y, _mm256_loadu_ps(x));
        x += 8; y += 8; num -= 8;
    }
#endif
    for (int i=0; i<num; i++) {
        y[i] = x[i];
    }
}


// transpose 2d tensor
void vm_transpose_2d_f32(float *x, float *y, int hi, int wi, int a0, int a1)
{
    if (a0 == 1 && a1 == 0) {
        vm_transpose_mat_f32(hi, wi, x, wi, y, hi);
    } else {
        _cp_f32(x, y, hi*wi);
    }
}


static void _transpose_021_f32(float *x, float *y, int hi, int wi, int ci)
{
    const int step = wi * ci;
    for (int i=0; i<hi; i++) {
        vm_transpose_mat_f32(wi, ci, x+i*step, ci, y+i*step, wi);
    }
}


static void _transpose_102_f32(float *x, float *y, int hi, int wi, int ci)
{
    for (int i=0; i<wi; i++) {
        float *px = x + i*ci;
        for (int j=0; j<hi; j++) {
            _cp_f32(px + j*wi*ci, y, ci);
            y += ci;
        }
    }
}


static void _transpose_120_f32(float *x, float *y, int hi, int wi, int ci)
{
    vm_transpose_mat_f32(hi, wi*ci, x, wi*ci, y, hi);
}


static void _transpose_201_f32(float *x, float *y, int hi, int wi, int ci)
{
    vm_transpose_mat_f32(hi*wi, ci, x, ci, y, hi*wi);
}


static void _transpose_210_f32(float *x, float *y, int hi, int wi, int ci)
{
    for (int i=0; i<ci; i++) {
        for (int j=0; j<wi; j++) {
            for (int k=0; k<hi; k++) {
                *y++ = x[k*wi*ci + j*ci + i];
            }
        }
    }
}


// transpose 3d tensor
void vm_transpose_3d_f32(float *x, float *y, int hi, int wi, int ci, int a0, int a1, int a2)
{
    if (a0 == 0 && a1 == 2 && a2 == 1) {
        _transpose_021_f32(x, y, hi, wi, ci);
    }
    else if (a0 == 1 && a1 == 0 && a2 == 2) {
        _transpose_102_f32(x, y, hi, wi, ci);
    }
    else if (a0 == 1 && a1 == 2 && a2 == 0) {
        _transpose_120_f32(x, y, hi, wi, ci);
    }
    else if (a0 == 2 && a1 == 0 && a2 == 1) {
        _transpose_201_f32(x, y, hi, wi, ci);
    }
    else if (a0 == 2 && a1 == 1 && a2 == 0) {
        _transpose_210_f32(x, y, hi, wi, ci);
    }
    else {
        _cp_f32(x, y, hi*wi*ci);
    }
}


// transpose 4d tensor
void vm_transpose_4d_f32(float *x, float *y, int ni, int hi, int wi, int ci, int a0, int a1, int a2, int a3)
{
    int shape[4] = {ni, hi, wi, ci}, steps[4] = {0};
    for (int n=1, i=3; i>=0; i--) {
        steps[i] = n; n *= shape[i];
    }
    int no = shape[a0], ho = shape[a1], wo = shape[a2], co = shape[a3];
    int sn = steps[a0], sh = steps[a1], sw = steps[a2], sc = steps[a3];

    if (sc == 1) {
        for (int n=0; n<no; n++) {
            for (int h=0; h<ho; h++) {
                for (int w=0; w<wo; w++) {
                    _cp_f32(x + n*sn + h*sh + w*sw, y, co);
                    y += co;
                }
            }
        }
    } else {
        for (int n=0; n<no; n++) {
            for (int h=0; h<ho; h++) {
                for (int w=0; w<wo; w++) {
                    for (int c=0; c<co; c++) {
                        *y++ = x[n*sn + h*sh + w*sw + c*sc];
                    }
                }
            }
        }
    }
}


