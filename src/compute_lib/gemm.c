
#include "compute_lib.h"


// broadcast bias to 16*16 matrix
static inline void _broadcast_bias_16x16(const float *bias, float *dst)
{
    __m256 R0 = _mm256_loadu_ps(bias);
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
        const __m256 T = relu ? _mm256_setzero_ps() :
                                _mm256_set1_ps(-1e20);

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
    __m256 S0 = _mm256_loadu_ps(src); src += src_step;
    __m256 S1 = _mm256_loadu_ps(src); src += src_step;
    __m256 S2 = _mm256_loadu_ps(src); src += src_step;
    __m256 S3 = _mm256_loadu_ps(src); src += src_step;
    __m256 S4 = _mm256_loadu_ps(src); src += src_step;
    __m256 S5 = _mm256_loadu_ps(src); src += src_step;
    __m256 S6 = _mm256_loadu_ps(src); src += src_step;
    __m256 S7 = _mm256_loadu_ps(src); src += src_step;

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

    _mm256_storeu_ps(dst, T0); dst += dst_step;
    _mm256_storeu_ps(dst, T1); dst += dst_step;
    _mm256_storeu_ps(dst, T2); dst += dst_step;
    _mm256_storeu_ps(dst, T3); dst += dst_step;
    _mm256_storeu_ps(dst, T4); dst += dst_step;
    _mm256_storeu_ps(dst, T5); dst += dst_step;
    _mm256_storeu_ps(dst, T6); dst += dst_step;
    _mm256_storeu_ps(dst, T7); dst += dst_step;
}


// transpose matrix
void vm_transpose_mat_f32(int rows, int cols, const float *src,
                          int src_step, float *dst, int dst_step)
{
    if (rows >= 8 && cols >= 8) {
        for (int i=0; i<cols; i+=8) {
            int m = i < cols-8 ? i : cols-8;

            for (int j=0; j<rows; j+=8) {
                int n = j < rows-8 ? j : rows-8;

                _transpose_8x8(src + m + n*src_step, src_step,
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
            _transpose_8x8(p, step, dst, 16);
            _transpose_8x8(p + 8*step, step, dst + 8, 16);
            p += 8; dst += 8*16;
        }
        if (r > 0) {
            vm_transpose_mat_f32(16, r, p, step, dst, 16);
            dst += r*16;
        }
        src += 16*step; cols -= 16;
    }
    if (cols > 0) {
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
            __m256 R0 = _mm256_loadu_ps(p + 0);
            __m256 R1 = _mm256_loadu_ps(p + 8);
            __m256 R2 = _mm256_loadu_ps(p + 16);
            __m256 R3 = _mm256_loadu_ps(p + 24);
            __m256 R4 = _mm256_loadu_ps(p + 32);
            __m256 R5 = _mm256_loadu_ps(p + 40);
            __m256 R6 = _mm256_loadu_ps(p + 48);
            __m256 R7 = _mm256_loadu_ps(p + 56);

            _mm256_storeu_ps(q + 0*stride, R0);
            _mm256_storeu_ps(q + 0*stride + 8, R1);
            _mm256_storeu_ps(q + 1*stride, R2);
            _mm256_storeu_ps(q + 1*stride + 8, R3);
            _mm256_storeu_ps(q + 2*stride, R4);
            _mm256_storeu_ps(q + 2*stride + 8, R5);
            _mm256_storeu_ps(q + 3*stride, R6);
            _mm256_storeu_ps(q + 3*stride + 8, R7);

            p += step; q += 16;
        }
        src += 64; dst += 4*stride; cols -= 64;
    }
    if (cols > 0) {
        int n = cols / 16, r = cols & 15;

        for (int i=0; i<rows; i++) {
            const float *p = src + i*step;
            float *q = dst + i*16;

            for (int j=0; j<n; j++) {
                __m256 R0 = _mm256_loadu_ps(p);
                __m256 R1 = _mm256_loadu_ps(p + 8);
                _mm256_storeu_ps(q, R0);
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
static void _imat_transpose_16x(int seg, int len,
                                const float **src, float *dst)
{
    __m256 S0, S1, S2, S3, S4, S5, S6, S7;
    __m256 T0, T1, T2, T3, T4, T5, T6, T7;

    int i = 0;
    while (i <= len - 8) {
        for (int j=0; j<2; j++) {
            const float **p = src + j*8*seg;
            float *q = dst + i*16 + j*8;

            S0 = _mm256_loadu_ps(p[0*seg] + i);
            S1 = _mm256_loadu_ps(p[1*seg] + i);
            S2 = _mm256_loadu_ps(p[2*seg] + i);
            S3 = _mm256_loadu_ps(p[3*seg] + i);
            S4 = _mm256_loadu_ps(p[4*seg] + i);
            S5 = _mm256_loadu_ps(p[5*seg] + i);
            S6 = _mm256_loadu_ps(p[6*seg] + i);
            S7 = _mm256_loadu_ps(p[7*seg] + i);

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

            _mm256_storeu_ps(q + 0, T0);
            _mm256_storeu_ps(q + 16, T1);
            _mm256_storeu_ps(q + 32, T2);
            _mm256_storeu_ps(q + 48, T3);
            _mm256_storeu_ps(q + 64, T4);
            _mm256_storeu_ps(q + 80, T5);
            _mm256_storeu_ps(q + 96, T6);
            _mm256_storeu_ps(q + 112, T7);
        }
        i += 8;
    }
    if (i < len) {
        for (int j=0; j<16; j++) {
            for (int k=i; k<len; k++) {
                dst[k*16 + j] = src[j*seg][k];
            }
        }
    }
}


// convert implicit matrix to transposed block
static void _imat_to_block(int rows, int cols, int seg, int len,
                           const float **src, float *dst)
{
    const int nw = cols / 16, rw = cols & 15;

    for (int i=0; i<nw; i++) {
        for (int j=0; j<seg; j++) {
            _imat_transpose_16x(seg, len, src + j, dst);
            dst += 16 * len;
        }
        src += 16 * seg;
    }
    if (rw > 0) {
        for (int k=0; k<seg; k++) {
            for (int j=0; j<rw; j++) {
                for (int i=0; i<len; i++) {
                    dst[i*16 + j] = src[j*seg + k][i];
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
            float *dst = y + j*64;

            Y0 = _mm256_loadu_ps(dst + 0);
            Y1 = _mm256_loadu_ps(dst + 8);
            Y2 = _mm256_loadu_ps(dst + 16);
            Y3 = _mm256_loadu_ps(dst + 24);
            Y4 = _mm256_loadu_ps(dst + 32);
            Y5 = _mm256_loadu_ps(dst + 40);
            Y6 = _mm256_loadu_ps(dst + 48);
            Y7 = _mm256_loadu_ps(dst + 56);

            for (int i=0; i<step; i++) {
                X0 = _mm256_loadu_ps(q + 0);
                X1 = _mm256_loadu_ps(q + 8);
                K0 = _mm256_broadcast_ss(p + 0);
                K1 = _mm256_broadcast_ss(p + 1);
                Y0 = _mm256_fmadd_ps(K0, X0, Y0);
                Y1 = _mm256_fmadd_ps(K0, X1, Y1);
                Y2 = _mm256_fmadd_ps(K1, X0, Y2);
                Y3 = _mm256_fmadd_ps(K1, X1, Y3);
                K0 = _mm256_broadcast_ss(p + 2);
                K1 = _mm256_broadcast_ss(p + 3);
                Y4 = _mm256_fmadd_ps(K0, X0, Y4);
                Y5 = _mm256_fmadd_ps(K0, X1, Y5);
                Y6 = _mm256_fmadd_ps(K1, X0, Y6);
                Y7 = _mm256_fmadd_ps(K1, X1, Y7);
                p += 16; q += 16;
            }
            _mm256_storeu_ps(dst + 0, Y0);
            _mm256_storeu_ps(dst + 8, Y1);
            _mm256_storeu_ps(dst + 16, Y2);
            _mm256_storeu_ps(dst + 24, Y3);
            _mm256_storeu_ps(dst + 32, Y4);
            _mm256_storeu_ps(dst + 40, Y5);
            _mm256_storeu_ps(dst + 48, Y6);
            _mm256_storeu_ps(dst + 56, Y7);
        }
        a += step * 16;
        b += step * 16;
        k -= step;
    }
    if (k > 0) {
        for (int j=0; j<4; j++) {
            const float *p = a + j*4, *q = b;
            float *dst = y + j*64;

            Y0 = _mm256_loadu_ps(dst + 0);
            Y1 = _mm256_loadu_ps(dst + 8);
            Y2 = _mm256_loadu_ps(dst + 16);
            Y3 = _mm256_loadu_ps(dst + 24);
            Y4 = _mm256_loadu_ps(dst + 32);
            Y5 = _mm256_loadu_ps(dst + 40);
            Y6 = _mm256_loadu_ps(dst + 48);
            Y7 = _mm256_loadu_ps(dst + 56);

            for (int i=0; i<k; i++) {
                X0 = _mm256_loadu_ps(q + 0);
                X1 = _mm256_loadu_ps(q + 8);
                K0 = _mm256_broadcast_ss(p + 0);
                K1 = _mm256_broadcast_ss(p + 1);
                Y0 = _mm256_fmadd_ps(K0, X0, Y0);
                Y1 = _mm256_fmadd_ps(K0, X1, Y1);
                Y2 = _mm256_fmadd_ps(K1, X0, Y2);
                Y3 = _mm256_fmadd_ps(K1, X1, Y3);
                K0 = _mm256_broadcast_ss(p + 2);
                K1 = _mm256_broadcast_ss(p + 3);
                Y4 = _mm256_fmadd_ps(K0, X0, Y4);
                Y5 = _mm256_fmadd_ps(K0, X1, Y5);
                Y6 = _mm256_fmadd_ps(K1, X0, Y6);
                Y7 = _mm256_fmadd_ps(K1, X1, Y7);
                p += 16; q += 16;
            }
            _mm256_storeu_ps(dst + 0, Y0);
            _mm256_storeu_ps(dst + 8, Y1);
            _mm256_storeu_ps(dst + 16, Y2);
            _mm256_storeu_ps(dst + 24, Y3);
            _mm256_storeu_ps(dst + 32, Y4);
            _mm256_storeu_ps(dst + 40, Y5);
            _mm256_storeu_ps(dst + 48, Y6);
            _mm256_storeu_ps(dst + 56, Y7);
        }
    }
}


// GEMM
void vm_gemm_f32(struct sf_allocator *alloc, int trans_a, int trans_b,
                 int m, int n, int k, const float *a, int lda, const float *b,
                 int ldb, float *c, int ldc, const float *bias, int relu)
{
    float zeros[16] = {0}, tmp[256] __attribute__((aligned(32)));
    const int np = (n + 15) & (~15), dm = 64;
    const int da = (trans_a ? dm : dm * lda), dc = dm * ldc;
    float *buf_a = sf_malloc(alloc, k * dm * sizeof(float));
    float *buf_b = sf_malloc(alloc, k * np * sizeof(float));
    _mat_to_block(trans_b, k, n, b, ldb, buf_b);

    while (m > 0) {
        const int H = m < dm ? m : dm;
        _mat_to_block(!trans_a, k, H, a, lda, buf_a);

        for (int j=0; j<n; j+=16) {
            const int w = n-j < 16 ? n-j : 16;

            for (int i=0; i<H; i+=16) {
                const int h = H-i < 16 ? H-i : 16;
                _broadcast_bias_16x16(bias ? bias + j : zeros, tmp);
                _gemm_kernel_16x16(k, buf_a + i*k, buf_b + j*k, tmp);
                _copy_mat_relu(h, w, tmp, 16, c + i*ldc + j, ldc, relu);
            }
        }
        a += da; c += dc; m -= dm;
    }
    sf_free(buf_a);
    sf_free(buf_b);
}


// GEMM with implicit matrix A
void vm_implicit_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                          const float **a, int seg, int len, const float *b,
                          int ldb, float *c, int ldc, const float *bias, int relu)
{
    float zeros[16] = {0}, tmp[256] __attribute__((aligned(32)));
    const int np = (n + 15) & (~15), dm = 64;
    const int da = dm * seg, dc = dm * ldc;
    float *buf_a = sf_malloc(alloc, k * dm * sizeof(float));
    float *buf_b = sf_malloc(alloc, k * np * sizeof(float));
    _mat_to_block_T(k, n, b, ldb, buf_b);

    while (m > 0) {
        const int H = m < dm ? m : dm;
        _imat_to_block(k, H, seg, len, a, buf_a);

        for (int j=0; j<n; j+=16) {
            const int w = n-j < 16 ? n-j : 16;

            for (int i=0; i<H; i+=16) {
                const int h = H-i < 16 ? H-i : 16;
                _broadcast_bias_16x16(bias ? bias + j : zeros, tmp);
                _gemm_kernel_16x16(k, buf_a + i*k, buf_b + j*k, tmp);
                _copy_mat_relu(h, w, tmp, 16, c + i*ldc + j, ldc, relu);
            }
        }
        a += da; c += dc; m -= dm;
    }
    sf_free(buf_a);
    sf_free(buf_b);
}


// GEMM with implicit matrix A and NK16-packed matrix B
void vm_implicit_packed_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                                 const float **a, int seg, int len, const float *b,
                                 float *c, int ldc, const float *bias, int relu)
{
    const int dm = 64, da = dm * seg, dc = dm * ldc;
    float zeros[16] = {0}, tmp[256] __attribute__((aligned(32)));
    float *buf_a = sf_malloc(alloc, k * dm * sizeof(float));

    while (m > 0) {
        const int H = m < dm ? m : dm;
        _imat_to_block(k, H, seg, len, a, buf_a);

        for (int j=0; j<n; j+=16) {
            const int w = n-j < 16 ? n-j : 16;

            for (int i=0; i<H; i+=16) {
                const int h = H-i < 16 ? H-i : 16;
                _broadcast_bias_16x16(bias ? bias + j : zeros, tmp);
                _gemm_kernel_16x16(k, buf_a + i*k, b + j*k, tmp);
                _copy_mat_relu(h, w, tmp, 16, c + i*ldc + j, ldc, relu);
            }
        }
        a += da; c += dc; m -= dm;
    }
    sf_free(buf_a);
}


