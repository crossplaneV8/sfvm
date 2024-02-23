
#include "compute_lib.h"



// broadcast bias[x : x+16] to 16x16 matrix
static inline void _broadcast_bias_16x16(const float *bias, int x, float *dst)
{
    __m256 R0, R1;

    if (bias != NULL) {
        R0 = _mm256_loadu_ps(bias + x + 0);
        R1 = _mm256_loadu_ps(bias + x + 8);
    } else {
        R0 = _mm256_setzero_ps();
        R1 = _mm256_setzero_ps();
    }
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


// GEMM kernel (Y[16, 16] = A[k, 16] * B[k, 16])
static void _gemm_kernel_16x16(int k, const float *a,
                               const float *b, float *y)
{
    __m256 K0, K1, X0, X1, Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7;
    const int step = 64;

    while (k >= step) {
        for (int j=0; j<16; j+=4) {
            const float *p = a + j;
            float *dst = y + j*16;

            Y0 = _mm256_loadu_ps(dst + 0*8);
            Y1 = _mm256_loadu_ps(dst + 1*8);
            Y2 = _mm256_loadu_ps(dst + 2*8);
            Y3 = _mm256_loadu_ps(dst + 3*8);
            Y4 = _mm256_loadu_ps(dst + 4*8);
            Y5 = _mm256_loadu_ps(dst + 5*8);
            Y6 = _mm256_loadu_ps(dst + 6*8);
            Y7 = _mm256_loadu_ps(dst + 7*8);

            for (int i=0; i<step*16; i+=16) {
                X0 = _mm256_loadu_ps(b + i + 0);
                X1 = _mm256_loadu_ps(b + i + 8);
                K0 = _mm256_broadcast_ss(p + i + 0);
                K1 = _mm256_broadcast_ss(p + i + 1);
                Y0 = _mm256_fmadd_ps(K0, X0, Y0);
                Y1 = _mm256_fmadd_ps(K0, X1, Y1);
                Y2 = _mm256_fmadd_ps(K1, X0, Y2);
                Y3 = _mm256_fmadd_ps(K1, X1, Y3);
                K0 = _mm256_broadcast_ss(p + i + 2);
                K1 = _mm256_broadcast_ss(p + i + 3);
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
        for (int j=0; j<16; j+=4) {
            const float *p = a + j;
            float *dst = y + j*16;

            Y0 = _mm256_loadu_ps(dst + 0*8);
            Y1 = _mm256_loadu_ps(dst + 1*8);
            Y2 = _mm256_loadu_ps(dst + 2*8);
            Y3 = _mm256_loadu_ps(dst + 3*8);
            Y4 = _mm256_loadu_ps(dst + 4*8);
            Y5 = _mm256_loadu_ps(dst + 5*8);
            Y6 = _mm256_loadu_ps(dst + 6*8);
            Y7 = _mm256_loadu_ps(dst + 7*8);

            for (int i=0; i<k*16; i+=16) {
                X0 = _mm256_loadu_ps(b + i + 0);
                X1 = _mm256_loadu_ps(b + i + 8);
                K0 = _mm256_broadcast_ss(p + i + 0);
                K1 = _mm256_broadcast_ss(p + i + 1);
                Y0 = _mm256_fmadd_ps(K0, X0, Y0);
                Y1 = _mm256_fmadd_ps(K0, X1, Y1);
                Y2 = _mm256_fmadd_ps(K1, X0, Y2);
                Y3 = _mm256_fmadd_ps(K1, X1, Y3);
                K0 = _mm256_broadcast_ss(p + i + 2);
                K1 = _mm256_broadcast_ss(p + i + 3);
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


// GEMM(transA=0, transB=1) for m < 16 case
static void _plain_gemm_NT_f32(int m, int n, int k,
                               const float *a, int lda,
                               const float *b, int ldb,
                               float *c, int ldc,
                               const float *bias, int relu)
{
    const int mask[16] = {-1, -1, -1, -1, -1, -1, -1, -1,};
    const int hk = k / 8, rk = k & 7;

    __m256i M = _mm256_loadu_si256((void*)(mask + 8 - rk));
    __m256 A, B0, B1, B2, B3;

    for (int i=0; i<m; i++) {
        int j = 0;
        for (; j<=n-4; j+=4) {
            const float *pa = a + i*lda;
            const float *pb = b + j*ldb;
            __m256 S0 = _mm256_setzero_ps();
            __m256 S1 = _mm256_setzero_ps();
            __m256 S2 = _mm256_setzero_ps();
            __m256 S3 = _mm256_setzero_ps();

            for (int x=0; x<hk; x++) {
                A = _mm256_loadu_ps(pa);
                B0 = _mm256_loadu_ps(pb + 0*ldb);
                B1 = _mm256_loadu_ps(pb + 1*ldb);
                B2 = _mm256_loadu_ps(pb + 2*ldb);
                B3 = _mm256_loadu_ps(pb + 3*ldb);
                S0 = _mm256_fmadd_ps(A, B0, S0);
                S1 = _mm256_fmadd_ps(A, B1, S1);
                S2 = _mm256_fmadd_ps(A, B2, S2);
                S3 = _mm256_fmadd_ps(A, B3, S3);
                pa += 8; pb += 8;
            }
            if (rk) {
                A = _mm256_maskload_ps(pa, M);
                B0 = _mm256_maskload_ps(pb + 0*ldb, M);
                B1 = _mm256_maskload_ps(pb + 1*ldb, M);
                B2 = _mm256_maskload_ps(pb + 2*ldb, M);
                B3 = _mm256_maskload_ps(pb + 3*ldb, M);
                S0 = _mm256_fmadd_ps(A, B0, S0);
                S1 = _mm256_fmadd_ps(A, B1, S1);
                S2 = _mm256_fmadd_ps(A, B2, S2);
                S3 = _mm256_fmadd_ps(A, B3, S3);
            }
            float tmp[4][8];
            _mm256_storeu_ps(tmp[0], S0);
            _mm256_storeu_ps(tmp[1], S1);
            _mm256_storeu_ps(tmp[2], S2);
            _mm256_storeu_ps(tmp[3], S3);
            for (int x=0; x<4; x++) {
                c[j + x] = tmp[x][0] + tmp[x][1] + tmp[x][2] + tmp[x][3] +
                           tmp[x][4] + tmp[x][5] + tmp[x][6] + tmp[x][7];
            }
        }
        for (; j<n; j++) {
            const float *pa = a + i*lda;
            const float *pb = b + j*ldb;
            double sum = 0;
            for (int x=0; x<k; x++) {
                sum += pa[x] * pb[x];
            }
            c[j] = sum;
        }
        if (bias) {
            if (relu) {
                vm_add_relu_f32((void*)bias, c, c, n);
            } else {
                vm_add_f32((void*)bias, c, c, n);
            }
        } else if (relu) {
            vm_relu_f32(c, c, n);
        }
        c += ldc;
    }
}


// GEMM
void vm_gemm_f32(struct sf_allocator *alloc, int trans_a, int trans_b,
                 int m, int n, int k, const float *a, int lda, const float *b,
                 int ldb, float *c, int ldc, const float *bias, int relu)
{
    if (m < 16 && trans_a == 0 && trans_b == 1) {
         _plain_gemm_NT_f32(m, n, k, a, lda, b, ldb, c, ldc, bias, relu);
    } else {
        const int n16 = (n + 15) & (~15);
        const int inc_a = trans_a ? 1 : lda;
        float *pack_b = sf_malloc(alloc, k * n16 * sizeof(float));
        _mat_to_block(trans_b, k, n, b, ldb, pack_b);

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int y=0; y<m; y+=16) {
            float pack[k * 16] __attribute__((aligned(32)));
            float tmp[16 * 16] __attribute__((aligned(32)));
            const int dy = (m - y) < 16 ? (m - y) : 16;
            _mat_to_block(!trans_a, k, dy, a + y*inc_a, lda, pack);

            for (int x=0; x<n; x+=16) {
                const int dx = (n - x) < 16 ? (n - x) : 16;
                _broadcast_bias_16x16(bias, x, tmp);
                _gemm_kernel_16x16(k, pack, pack_b + x*k, tmp);
                _copy_mat_relu(dy, dx, tmp, 16, c + y*ldc + x, ldc, relu);
            }
        }
        sf_free(pack_b);
    }
}


// GEMM with implicit matrix A
void vm_implicit_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                          struct vm_imat *a, const float *b, int ldb,
                          float *c, int ldc, const float *bias, int relu)
{
    const int n16 = (n + 15) & (~15);
    float *pack_b = sf_malloc(alloc, k * n16 * sizeof(float));
    _mat_to_block_T(k, n, b, ldb, pack_b);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int y=0; y<m; y+=16) {
        float pack[k * 16] __attribute__((aligned(32)));
        float tmp[16 * 16] __attribute__((aligned(32)));
        const int dy = (m - y) < 16 ? (m - y) : 16;
        vm_pack_imat_16x(a, y, pack);

        for (int x=0; x<n; x+=16) {
            const int dx = (n - x) < 16 ? (n - x) : 16;
            _broadcast_bias_16x16(bias, x, tmp);
            _gemm_kernel_16x16(k, pack, pack_b + x*k, tmp);
            _copy_mat_relu(dy, dx, tmp, 16, c + y*ldc + x, ldc, relu);
        }
    }
    sf_free(pack_b);
}


// GEMM with implicit matrix A and NK16-packed matrix B
void vm_implicit_packed_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                                 struct vm_imat *a, const float *b,
                                 float *c, int ldc, const float *bias, int relu)
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int y=0; y<m; y+=16) {
        float pack[k * 16] __attribute__((aligned(32)));
        float tmp[16 * 16] __attribute__((aligned(32)));
        const int dy = (m - y) < 16 ? (m - y) : 16;
        vm_pack_imat_16x(a, y, pack);

        for (int x=0; x<n; x+=16) {
            const int dx = (n - x) < 16 ? (n - x) : 16;
            _broadcast_bias_16x16(bias, x, tmp);
            _gemm_kernel_16x16(k, pack, b + x*k, tmp);
            _copy_mat_relu(dy, dx, tmp, 16, c + y*ldc + x, ldc, relu);
        }
    }
}


