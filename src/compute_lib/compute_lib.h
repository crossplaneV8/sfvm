
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include "base/mem_alloc.h"


#ifdef __AVX2__
#include <x86intrin.h>
#endif


// implicit matrix
struct vm_imat
{
    int rows, cols;
    int segs, len;
    float **data;
};


// x[i] = 0
void vm_clear_f32(float *x, int num);

// y[i] = x[i]
void vm_copy_f32(float *x, float *y, int num);

// z[i] = x[i] + y[i]
void vm_add_f32(float *x, float *y, float *z, int num);

// z[i] = x[i] * y[i]
void vm_mul_f32(float *x, float *y, float *z, int num);

// y[i] = max(x[i], 0)
void vm_relu_f32(float *x, float *y, int num);

// z[i] = max(x[i] + y[i], 0)
void vm_add_relu_f32(float *x, float *y, float *z, int num);


// broadcast add 1d tensor
void vm_add_1d_f32(float *x, float *y, float *z, int nx, int ny);

// broadcast add 2d tensor
void vm_add_2d_f32(float *x, float *y, float *z,
                   int hx, int wx, int hy, int wy);

// broadcast add 3d tensor
void vm_add_3d_f32(float *x, float *y, float *z,
                   int hx, int wx, int cx,
                   int hy, int wy, int cy);

// broadcast add 4d tensor
void vm_add_4d_f32(float *x, float *y, float *z,
                   int nx, int hx, int wx, int cx,
                   int ny, int hy, int wy, int cy);


// global average pooling (NHWC layout)
void vm_gl_avgpool_nhwc_f32(float *x, float *y, int n, int h, int w, int c);

// max pooling (NHWC layout)
void vm_maxpool_nhwc_f32(float *x, float *y,
                         int ni, int hi, int wi, int ci,
                         int no, int ho, int wo, int co,
                         int ph, int pw, int sh, int sw,
                         int kh, int kw);


// transpose 2d tensor
void vm_transpose_2d_f32(float *x, float *y, int hi, int wi, int a0, int a1);

// transpose 3d tensor
void vm_transpose_3d_f32(float *x, float *y, int hi, int wi, int ci, int a0, int a1, int a2);

// transpose 4d tensor
void vm_transpose_4d_f32(float *x, float *y, int ni, int hi, int wi, int ci, int a0, int a1, int a2, int a3);

// transpose matrix
void vm_transpose_mat_f32(int rows, int cols, const float *src,
                          int src_step, float *dst, int dst_step);


// GEMM
void vm_gemm_f32(struct sf_allocator *alloc, int trans_a, int trans_b,
                 int m, int n, int k, const float *a, int lda, const float *b,
                 int ldb, float *c, int ldc, const float *bias, int relu);

// GEMM with implicit matrix A
void vm_implicit_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                          struct vm_imat *a, const float *b, int ldb,
                          float *c, int ldc, const float *bias, int relu);

// GEMM with implicit matrix A and NK16-packed matrix B
void vm_implicit_packed_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                                 struct vm_imat *a, const float *b,
                                 float *c, int ldc, const float *bias, int relu);

// convolution (data: NHWC, weight: OHWI)
void vm_conv_nhwc_ohwi_f32(struct sf_allocator *alloc, float *x, float *w, float *b, float *y,
                           int ni, int hi, int wi, int ci, int no, int ho, int wo, int co,
                           int ph, int pw, int sh, int sw, int kh, int kw, int dh, int dw, int relu);

// convolution (data: NHWC, weight: NK16-packed)
void vm_conv_nhwc_nk16_f32(struct sf_allocator *alloc, float *x, float *w, float *b, float *y,
                           int ni, int hi, int wi, int ci, int no, int ho, int wo, int co,
                           int ph, int pw, int sh, int sw, int kh, int kw, int dh, int dw, int relu);



#ifdef __cplusplus
    }
#endif

