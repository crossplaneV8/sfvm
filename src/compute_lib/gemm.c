
#include "compute_lib.h"


// broadcast bias[1, 16] to matrix[16, 16]
static inline void _broadcast_bias_16x16(const float *bias, float *dst)
{
    __m256 R0 = _mm256_loadu_ps(bias + 0);
    __m256 R1 = _mm256_loadu_ps(bias + 8);

    for (int i=0; i<4; i++) {
        _mm256_storeu_ps(dst, R0); dst += 8;
        _mm256_storeu_ps(dst, R1); dst += 8;
        _mm256_storeu_ps(dst, R0); dst += 8;
        _mm256_storeu_ps(dst, R1); dst += 8;
        _mm256_storeu_ps(dst, R0); dst += 8;
        _mm256_storeu_ps(dst, R1); dst += 8;
        _mm256_storeu_ps(dst, R0); dst += 8;
        _mm256_storeu_ps(dst, R1); dst += 8;
    }
}


// copy matrix (cols <= 16)
static inline void _copy_mat_relu(int rows, int cols,
                                  const float *src, int src_step,
                                  float *dst, int dst_step, int relu)
{
    if (cols >= 8) {
        const int offset = cols - 8;
        const __m256 T = _mm256_set1_ps(relu ? 0 : -1e20);

        for (int i=0; i<rows; i++) {
            __m256 R0 = _mm256_loadu_ps(src);
            __m256 R1 = _mm256_loadu_ps(src + offset);
            R0 = _mm256_max_ps(R0, T);
            R1 = _mm256_max_ps(R1, T);
            _mm256_storeu_ps(dst, R0);
            _mm256_storeu_ps(dst + offset, R1);
            dst += dst_step;
            src += src_step;
        }
    } else {
        const float t = relu ? 0 : -1e20;

        for (int i=0; i<rows; i++) {
            switch (cols) {
                case 7: dst[6] = src[6] > t ? src[6] : t;
                case 6: dst[5] = src[5] > t ? src[5] : t;
                case 5: dst[4] = src[4] > t ? src[4] : t;
                case 4: dst[3] = src[3] > t ? src[3] : t;
                case 3: dst[2] = src[2] > t ? src[2] : t;
                case 2: dst[1] = src[1] > t ? src[1] : t;
                case 1: dst[0] = src[0] > t ? src[0] : t;
            }
            dst += dst_step;
            src += src_step;
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

    S0 = _mm256_shuffle_ps(T0, T1, 0x4444);
    S1 = _mm256_shuffle_ps(T2, T3, 0x4444);
    S2 = _mm256_shuffle_ps(T4, T5, 0x4444);
    S3 = _mm256_shuffle_ps(T6, T7, 0x4444);
    S4 = _mm256_shuffle_ps(T0, T1, 0xeeee);
    S5 = _mm256_shuffle_ps(T2, T3, 0xeeee);
    S6 = _mm256_shuffle_ps(T4, T5, 0xeeee);
    S7 = _mm256_shuffle_ps(T6, T7, 0xeeee);

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


// convert matrix to transposed block
static void _mat_to_block_T(int rows, int cols,
                            const float *src, int step, float *dst)
{
    const int n = rows / 8, r = rows % 8;

    while (cols >= 16) {
        const float *p = src;

        for (int j=n; j>0; j--) {
            _transpose_8x8(p + 0*step, step, dst + 0, 16);
            _transpose_8x8(p + 8*step, step, dst + 8, 16);
            p += 8; dst += 8*16;
        }
        if (r) {
            vm_transpose_mat_f32(16, r, p, step, dst, 16);
            dst += r*16;
        }
        src += 16*step; cols -= 16;
    }
    if (cols) {
        vm_transpose_mat_f32(cols, rows, src, step, dst, 16);
    }
}


// convert matrix to untransposed block
static void _mat_to_block_N(int rows, int cols,
                            const float *src, int step, float *dst)
{
    const int stride = rows * 16;

    while (cols >= 64) {
        const float *p = src; float *q = dst;

        for (int j=0; j<rows; j++) {
            __m256 R0 = _mm256_loadu_ps(p + 0*8);
            __m256 R1 = _mm256_loadu_ps(p + 1*8);
            __m256 R2 = _mm256_loadu_ps(p + 2*8);
            __m256 R3 = _mm256_loadu_ps(p + 3*8);
            __m256 R4 = _mm256_loadu_ps(p + 4*8);
            __m256 R5 = _mm256_loadu_ps(p + 5*8);
            __m256 R6 = _mm256_loadu_ps(p + 6*8);
            __m256 R7 = _mm256_loadu_ps(p + 7*8);
            _mm256_storeu_ps(q + 0*stride + 0, R0);
            _mm256_storeu_ps(q + 0*stride + 8, R1);
            _mm256_storeu_ps(q + 1*stride + 0, R2);
            _mm256_storeu_ps(q + 1*stride + 8, R3);
            _mm256_storeu_ps(q + 2*stride + 0, R4);
            _mm256_storeu_ps(q + 2*stride + 8, R5);
            _mm256_storeu_ps(q + 3*stride + 0, R6);
            _mm256_storeu_ps(q + 3*stride + 8, R7);
            p += step; q += 16;
        }
        src += 64; dst += 4*stride; cols -= 64;
    }
    if (cols) {
        int n = cols / 16, r = cols & 15;

        for (int i=0; i<rows; i++) {
            const float *p = src + i*step;
            float *q = dst + i*16;

            for (int j=0; j<n; j++) {
                __m256 R0 = _mm256_loadu_ps(p + 0);
                __m256 R1 = _mm256_loadu_ps(p + 8);
                _mm256_storeu_ps(q + 0, R0);
                _mm256_storeu_ps(q + 8, R1);
                p += 16; q += stride;
            }
            for (int j=0; j<r; j++) {
                q[j] = p[j];
            }
        }
    }
}


// convert matrix to block
static void _mat_to_block(int trans, int rows, int cols,
                          const float *src, int step, float *dst)
{
    if (trans) {
        _mat_to_block_T(rows, cols, src, step, dst);
    } else {
        _mat_to_block_N(rows, cols, src, step, dst);
    }
}


// convert implicit matrix to transposed block
static void _imat_transpose_16x(int len, const float **src, float *dst)
{
    __m256 S0, S1, S2, S3, S4, S5, S6, S7;
    __m256 T0, T1, T2, T3, T4, T5, T6, T7;

    int i = 0;
    while (i <= len - 8) {
        for (int j=0; j<2; j++) {
            const float **p = src + j*8;
            float *q = dst + i*16 + j*8;

            S0 = _mm256_loadu_ps(p[0] + i);
            S1 = _mm256_loadu_ps(p[1] + i);
            S2 = _mm256_loadu_ps(p[2] + i);
            S3 = _mm256_loadu_ps(p[3] + i);
            S4 = _mm256_loadu_ps(p[4] + i);
            S5 = _mm256_loadu_ps(p[5] + i);
            S6 = _mm256_loadu_ps(p[6] + i);
            S7 = _mm256_loadu_ps(p[7] + i);

            T0 = _mm256_unpacklo_ps(S0, S1);
            T1 = _mm256_unpacklo_ps(S2, S3);
            T2 = _mm256_unpacklo_ps(S4, S5);
            T3 = _mm256_unpacklo_ps(S6, S7);
            T4 = _mm256_unpackhi_ps(S0, S1);
            T5 = _mm256_unpackhi_ps(S2, S3);
            T6 = _mm256_unpackhi_ps(S4, S5);
            T7 = _mm256_unpackhi_ps(S6, S7);

            S0 = _mm256_shuffle_ps(T0, T1, 0x4444);
            S1 = _mm256_shuffle_ps(T2, T3, 0x4444);
            S2 = _mm256_shuffle_ps(T4, T5, 0x4444);
            S3 = _mm256_shuffle_ps(T6, T7, 0x4444);
            S4 = _mm256_shuffle_ps(T0, T1, 0xeeee);
            S5 = _mm256_shuffle_ps(T2, T3, 0xeeee);
            S6 = _mm256_shuffle_ps(T4, T5, 0xeeee);
            S7 = _mm256_shuffle_ps(T6, T7, 0xeeee);

            T0 = _mm256_permute2f128_ps(S0, S1, 0x20);
            T1 = _mm256_permute2f128_ps(S4, S5, 0x20);
            T2 = _mm256_permute2f128_ps(S2, S3, 0x20);
            T3 = _mm256_permute2f128_ps(S6, S7, 0x20);
            T4 = _mm256_permute2f128_ps(S0, S1, 0x31);
            T5 = _mm256_permute2f128_ps(S4, S5, 0x31);
            T6 = _mm256_permute2f128_ps(S2, S3, 0x31);
            T7 = _mm256_permute2f128_ps(S6, S7, 0x31);

            _mm256_storeu_ps(q + 0*16, T0);
            _mm256_storeu_ps(q + 1*16, T1);
            _mm256_storeu_ps(q + 2*16, T2);
            _mm256_storeu_ps(q + 3*16, T3);
            _mm256_storeu_ps(q + 4*16, T4);
            _mm256_storeu_ps(q + 5*16, T5);
            _mm256_storeu_ps(q + 6*16, T6);
            _mm256_storeu_ps(q + 7*16, T7);
        }
        i += 8;
    }
    if (i < len) {
        for (int j=0; j<16; j++) {
            for (int k=i; k<len; k++) {
                dst[k*16 + j] = src[j][k];
            }
        }
    }
}


// convert imat[y:y+h, :] to NK16-packed layout
static void _imat_to_block(struct vm_imat *mat, int y, int h, float *dst)
{
    const int rows = mat->rows, segs = mat->segs, len = mat->len;
    const float **data = (void*)(mat->data + y);

    while (h >= 16) {
        for (int s=0; s<segs; s++) {
            _imat_transpose_16x(len, data + s*rows, dst);
            dst += 16 * len;
        }
        data += 16;
        h -= 16;
    }
    if (h) {
        for (int s=0; s<segs; s++) {
            for (int j=0; j<h; j++) {
                for (int i=0; i<len; i++) {
                    dst[i*16 + j] = data[s*rows + j][i];
                }
            }
            dst += 16 * len;
        }
    }
}


// GEMM kernel (Y[16, 16] = A[k, 16] * B[k, 16])
static void _gemm_kernel_16x16(int k, const float *a,
                               const float *b, float *y)
{
    __m256 K0, K1, X0, X1, Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7;
    const int step = 8;

    while (k >= step) {
        for (int j=0; j<4; j++) {
            const float *p = a + j*4, *q = b;
            float *dst = y + j*4*16;

            Y0 = _mm256_loadu_ps(dst + 0*8);
            Y1 = _mm256_loadu_ps(dst + 1*8);
            Y2 = _mm256_loadu_ps(dst + 2*8);
            Y3 = _mm256_loadu_ps(dst + 3*8);
            Y4 = _mm256_loadu_ps(dst + 4*8);
            Y5 = _mm256_loadu_ps(dst + 5*8);
            Y6 = _mm256_loadu_ps(dst + 6*8);
            Y7 = _mm256_loadu_ps(dst + 7*8);

            for (int i=0; i<step; i++) {
                X0 = _mm256_loadu_ps(q + i*16 + 0);
                X1 = _mm256_loadu_ps(q + i*16 + 8);
                K0 = _mm256_broadcast_ss(p + i*16 + 0);
                K1 = _mm256_broadcast_ss(p + i*16 + 1);
                Y0 = _mm256_fmadd_ps(K0, X0, Y0);
                Y1 = _mm256_fmadd_ps(K0, X1, Y1);
                Y2 = _mm256_fmadd_ps(K1, X0, Y2);
                Y3 = _mm256_fmadd_ps(K1, X1, Y3);
                K0 = _mm256_broadcast_ss(p + i*16 + 2);
                K1 = _mm256_broadcast_ss(p + i*16 + 3);
                Y4 = _mm256_fmadd_ps(K0, X0, Y4);
                Y5 = _mm256_fmadd_ps(K0, X1, Y5);
                Y6 = _mm256_fmadd_ps(K1, X0, Y6);
                Y7 = _mm256_fmadd_ps(K1, X1, Y7);
            }
            _mm256_storeu_ps(dst + 0*8, Y0);
            _mm256_storeu_ps(dst + 1*8, Y1);
            _mm256_storeu_ps(dst + 2*8, Y2);
            _mm256_storeu_ps(dst + 3*8, Y3);
            _mm256_storeu_ps(dst + 4*8, Y4);
            _mm256_storeu_ps(dst + 5*8, Y5);
            _mm256_storeu_ps(dst + 6*8, Y6);
            _mm256_storeu_ps(dst + 7*8, Y7);
        }
        a += step * 16;
        b += step * 16;
        k -= step;
    }
    if (k) {
        for (int j=0; j<4; j++) {
            const float *p = a + j*4, *q = b;
            float *dst = y + j*4*16;

            Y0 = _mm256_loadu_ps(dst + 0*8);
            Y1 = _mm256_loadu_ps(dst + 1*8);
            Y2 = _mm256_loadu_ps(dst + 2*8);
            Y3 = _mm256_loadu_ps(dst + 3*8);
            Y4 = _mm256_loadu_ps(dst + 4*8);
            Y5 = _mm256_loadu_ps(dst + 5*8);
            Y6 = _mm256_loadu_ps(dst + 6*8);
            Y7 = _mm256_loadu_ps(dst + 7*8);

            for (int i=0; i<k; i++) {
                X0 = _mm256_loadu_ps(q + i*16 + 0);
                X1 = _mm256_loadu_ps(q + i*16 + 8);
                K0 = _mm256_broadcast_ss(p + i*16 + 0);
                K1 = _mm256_broadcast_ss(p + i*16 + 1);
                Y0 = _mm256_fmadd_ps(K0, X0, Y0);
                Y1 = _mm256_fmadd_ps(K0, X1, Y1);
                Y2 = _mm256_fmadd_ps(K1, X0, Y2);
                Y3 = _mm256_fmadd_ps(K1, X1, Y3);
                K0 = _mm256_broadcast_ss(p + i*16 + 2);
                K1 = _mm256_broadcast_ss(p + i*16 + 3);
                Y4 = _mm256_fmadd_ps(K0, X0, Y4);
                Y5 = _mm256_fmadd_ps(K0, X1, Y5);
                Y6 = _mm256_fmadd_ps(K1, X0, Y6);
                Y7 = _mm256_fmadd_ps(K1, X1, Y7);
            }
            _mm256_storeu_ps(dst + 0*8, Y0);
            _mm256_storeu_ps(dst + 1*8, Y1);
            _mm256_storeu_ps(dst + 2*8, Y2);
            _mm256_storeu_ps(dst + 3*8, Y3);
            _mm256_storeu_ps(dst + 4*8, Y4);
            _mm256_storeu_ps(dst + 5*8, Y5);
            _mm256_storeu_ps(dst + 6*8, Y6);
            _mm256_storeu_ps(dst + 7*8, Y7);
        }
    }
}


// GEMM
void vm_gemm_f32(struct sf_allocator *alloc, int trans_a, int trans_b,
                 int m, int n, int k, const float *a, int lda, const float *b,
                 int ldb, float *c, int ldc, const float *bias, int relu)
{
    const int step = 64;
    float zeros[16] = {0}, tmp[256] __attribute__((aligned(32)));
    float *buf_a = sf_malloc(alloc, k * step * sizeof(float));
    float *buf_b = sf_malloc(alloc, k * ((n+15)&(~15)) * sizeof(float));
    _mat_to_block(trans_b, k, n, b, ldb, buf_b);

    for (int y=0; y<m; y+=step) {
        const int h = m - y < step ? m - y : step;
        _mat_to_block(!trans_a, k, h, a, lda, buf_a);

        for (int j=0; j<n; j+=16) {
            for (int i=0; i<h; i+=16) {
                const int dw = n - j < 16 ? n - j : 16;
                const int dh = h - i < 16 ? h - i : 16;
                _broadcast_bias_16x16(bias ? bias + j : zeros, tmp);
                _gemm_kernel_16x16(k, buf_a + i*k, buf_b + j*k, tmp);
                _copy_mat_relu(dh, dw, tmp, 16, c + i*ldc + j, ldc, relu);
            }
        }
        a += trans_a ? step : step * lda;
        c += step * ldc;
    }
    sf_free(buf_a);
    sf_free(buf_b);
}


// GEMM with implicit matrix A
void vm_implicit_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                          struct vm_imat *a, const float *b, int ldb,
                          float *c, int ldc, const float *bias, int relu)
{
    const int step = 64;
    float zeros[16] = {0}, tmp[256] __attribute__((aligned(32)));
    float *buf_a = sf_malloc(alloc, k * step * sizeof(float));
    float *buf_b = sf_malloc(alloc, k * ((n+15)&(~15)) * sizeof(float));
    _mat_to_block_T(k, n, b, ldb, buf_b);

    for (int y=0; y<m; y+=step) {
        const int h = m - y < step ? m - y : step;
        _imat_to_block(a, y, h, buf_a);

        for (int j=0; j<n; j+=16) {
            for (int i=0; i<h; i+=16) {
                const int dw = n - j < 16 ? n - j : 16;
                const int dh = h - i < 16 ? h - i : 16;
                _broadcast_bias_16x16(bias ? bias + j : zeros, tmp);
                _gemm_kernel_16x16(k, buf_a + i*k, buf_b + j*k, tmp);
                _copy_mat_relu(dh, dw, tmp, 16, c + i*ldc + j, ldc, relu);
            }
        }
        c += step * ldc;
    }
    sf_free(buf_a);
    sf_free(buf_b);
}


// GEMM with implicit matrix A and NK16-packed matrix B
void vm_implicit_packed_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                                 struct vm_imat *a, const float *b,
                                 float *c, int ldc, const float *bias, int relu)
{
    const int step = 64;
    float zeros[16] = {0}, tmp[256] __attribute__((aligned(32)));
    float *buf_a = sf_malloc(alloc, k * step * sizeof(float));

    for (int y=0; y<m; y+=step) {
        const int h = m - y < step ? m - y : step;
        _imat_to_block(a, y, h, buf_a);

        for (int j=0; j<n; j+=16) {
            for (int i=0; i<h; i+=16) {
                const int dw = n - j < 16 ? n - j : 16;
                const int dh = h - i < 16 ? h - i : 16;
                _broadcast_bias_16x16(bias ? bias + j : zeros, tmp);
                _gemm_kernel_16x16(k, buf_a + i*k, b + j*k, tmp);
                _copy_mat_relu(dh, dw, tmp, 16, c + i*ldc + j, ldc, relu);
            }
        }
        c += step * ldc;
    }
    sf_free(buf_a);
}


