
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


// transpose 8*8 sub matrix
static inline void _transpose_8x8(const float *src, int src_step,
                                  float *dst, int dst_step)
{
    __m256 S0 = _mm256_loadu_ps(src + 0*src_step);
    __m256 S1 = _mm256_loadu_ps(src + 1*src_step);
    __m256 S2 = _mm256_loadu_ps(src + 2*src_step);
    __m256 S3 = _mm256_loadu_ps(src + 3*src_step);
    __m256 S4 = _mm256_loadu_ps(src + 4*src_step);
    __m256 S5 = _mm256_loadu_ps(src + 5*src_step);
    __m256 S6 = _mm256_loadu_ps(src + 6*src_step);
    __m256 S7 = _mm256_loadu_ps(src + 7*src_step);

    __m256 T0 = _mm256_unpacklo_ps(S0, S1);
    __m256 T1 = _mm256_unpacklo_ps(S2, S3);
    __m256 T2 = _mm256_unpacklo_ps(S4, S5);
    __m256 T3 = _mm256_unpacklo_ps(S6, S7);
    __m256 T4 = _mm256_unpackhi_ps(S0, S1);
    __m256 T5 = _mm256_unpackhi_ps(S2, S3);
    __m256 T6 = _mm256_unpackhi_ps(S4, S5);
    __m256 T7 = _mm256_unpackhi_ps(S6, S7);

    S0 = _mm256_shuffle_ps(T0, T1, 0x44);
    S1 = _mm256_shuffle_ps(T2, T3, 0x44);
    S2 = _mm256_shuffle_ps(T4, T5, 0x44);
    S3 = _mm256_shuffle_ps(T6, T7, 0x44);
    S4 = _mm256_shuffle_ps(T0, T1, 0xee);
    S5 = _mm256_shuffle_ps(T2, T3, 0xee);
    S6 = _mm256_shuffle_ps(T4, T5, 0xee);
    S7 = _mm256_shuffle_ps(T6, T7, 0xee);

    T0 = _mm256_permute2f128_ps(S0, S1, 0x20);
    T1 = _mm256_permute2f128_ps(S4, S5, 0x20);
    T2 = _mm256_permute2f128_ps(S2, S3, 0x20);
    T3 = _mm256_permute2f128_ps(S6, S7, 0x20);
    T4 = _mm256_permute2f128_ps(S0, S1, 0x31);
    T5 = _mm256_permute2f128_ps(S4, S5, 0x31);
    T6 = _mm256_permute2f128_ps(S2, S3, 0x31);
    T7 = _mm256_permute2f128_ps(S6, S7, 0x31);

    _mm256_storeu_ps(dst + 0*dst_step, T0);
    _mm256_storeu_ps(dst + 1*dst_step, T1);
    _mm256_storeu_ps(dst + 2*dst_step, T2);
    _mm256_storeu_ps(dst + 3*dst_step, T3);
    _mm256_storeu_ps(dst + 4*dst_step, T4);
    _mm256_storeu_ps(dst + 5*dst_step, T5);
    _mm256_storeu_ps(dst + 6*dst_step, T6);
    _mm256_storeu_ps(dst + 7*dst_step, T7);
}


// transpose matrix
void vm_transpose_mat_f32(int rows, int cols, const float *src,
                          int src_step, float *dst, int dst_step)
{
    if (rows >= 8 && cols >= 8) {
        for (int i=0; i<cols; i+=8) {
            for (int j=0; j<rows; j+=8) {
                int m = i < cols - 8 ? i : cols - 8;
                int n = j < rows - 8 ? j : rows - 8;
                _transpose_8x8(src + n*src_step + m, src_step,
                               dst + m*dst_step + n, dst_step);
            }
        }
    } else {
        for (int i=0; i<cols; i++) {
            for (int j=0; j<rows; j++) {
                dst[i*dst_step + j] = src[j*src_step + i];
            }
        }
    }
}


// pack matrix src[w, h] ==> dst[w/pack, h, pack]
static void _pack_mat_T(int mat_h, int mat_w,
                        const float *src, int step,
                        float *dst, int pack)
{
    while (mat_w > 0) {
        int width = (mat_w < pack) ? mat_w : pack;
        vm_transpose_mat_f32(width, mat_h, src, step, dst, pack);
        src += pack * step;
        dst += mat_h * pack;
        mat_w -= pack;
    }
}


static void _copy_mat_f32(int mat_h, int mat_w,
                          const float *src, int src_step,
                          float *dst, int dst_step)
{
    if (mat_w >= 8) {
        for (int i=0; i<mat_h; i++) {
            for (int j=0; j<mat_w; j+=8) {
                int x = j < mat_w - 8 ? j : mat_w - 8;
                __m256 R = _mm256_loadu_ps(src + x);
                _mm256_storeu_ps(dst + x, R);
            }
            dst += dst_step;
            src += src_step;
        }
    } else {
        const int mask[16] = {-1, -1, -1, -1, -1, -1, -1, -1,};
         __m256i M = _mm256_loadu_si256((void*)(mask + 8 - mat_w));

        for (int i=0; i<mat_h; i++) {
            __m256 R = _mm256_maskload_ps(src + i*src_step, M);
            _mm256_maskstore_ps(dst + i*dst_step, M, R);
        }
    }
}


// pack matrix src[h, w] ==> dst[w/pack, h, pack]
static void _pack_mat_N(int mat_h, int mat_w,
                        const float *src, int step,
                        float *dst, int pack)
{
    while (mat_w > 0) {
        int width = (mat_w < pack) ? mat_w : pack;
        _copy_mat_f32(mat_h, width, src, step, dst, pack);
        src += pack;
        dst += mat_h * pack;
        mat_w -= pack;
    }
}


// pack matrix
void vm_pack_mat(int trans, int rows, int cols,
                 const float *src, int step,
                 float *dst, int pack)
{
    if (trans) {
        _pack_mat_T(rows, cols, src, step, dst, pack);
    } else {
        _pack_mat_N(rows, cols, src, step, dst, pack);
    }
}

