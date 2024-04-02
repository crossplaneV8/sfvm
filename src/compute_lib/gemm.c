
#include "compute_lib.h"


// kernel_shape: [G, M, N, K]
#define _KERNEL_G   (3)
#define _KERNEL_M   (6)
#define _KERNEL_N   (16)
#define _KERNEL_K   (256)


// broadcast bias[x : x+16] to matrix[n, 16]
static inline void _broadcast_bias_16(const float *bias, int x, int n, float *dst)
{
    __m256 R0, R1;

    if (bias != NULL) {
        R0 = _mm256_loadu_ps(bias + x + 0);
        R1 = _mm256_loadu_ps(bias + x + 8);
    } else {
        R0 = _mm256_setzero_ps();
        R1 = _mm256_setzero_ps();
    }
    for (int i=0; i<n; i++) {
        _mm256_storeu_ps(dst + i*16 + 0, R0);
        _mm256_storeu_ps(dst + i*16 + 8, R1);
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


// GEMM kernel (Y[6, 16] = A[k, 6] * B[k, 16])
static inline void _gemm_kernel_6x16(const int k, const float *a,
                                     const float *b, float *y)
{
    __m256 A0, A1, B0, B1;
    __m256 Y00 = _mm256_loadu_ps(y+0x00), Y01 = _mm256_loadu_ps(y+0x08);
    __m256 Y10 = _mm256_loadu_ps(y+0x10), Y11 = _mm256_loadu_ps(y+0x18);
    __m256 Y20 = _mm256_loadu_ps(y+0x20), Y21 = _mm256_loadu_ps(y+0x28);
    __m256 Y30 = _mm256_loadu_ps(y+0x30), Y31 = _mm256_loadu_ps(y+0x38);
    __m256 Y40 = _mm256_loadu_ps(y+0x40), Y41 = _mm256_loadu_ps(y+0x48);
    __m256 Y50 = _mm256_loadu_ps(y+0x50), Y51 = _mm256_loadu_ps(y+0x58);

    for (int i=0; i<k; i++) {
        B0 = _mm256_loadu_ps(b + 0);
        B1 = _mm256_loadu_ps(b + 8);

        A0 = _mm256_broadcast_ss(a + 0);
        A1 = _mm256_broadcast_ss(a + 1);
        Y00 = _mm256_fmadd_ps(A0, B0, Y00);
        Y01 = _mm256_fmadd_ps(A0, B1, Y01);
        Y10 = _mm256_fmadd_ps(A1, B0, Y10);
        Y11 = _mm256_fmadd_ps(A1, B1, Y11);

        A0 = _mm256_broadcast_ss(a + 2);
        A1 = _mm256_broadcast_ss(a + 3);
        Y20 = _mm256_fmadd_ps(A0, B0, Y20);
        Y21 = _mm256_fmadd_ps(A0, B1, Y21);
        Y30 = _mm256_fmadd_ps(A1, B0, Y30);
        Y31 = _mm256_fmadd_ps(A1, B1, Y31);

        A0 = _mm256_broadcast_ss(a + 4);
        A1 = _mm256_broadcast_ss(a + 5);
        Y40 = _mm256_fmadd_ps(A0, B0, Y40);
        Y41 = _mm256_fmadd_ps(A0, B1, Y41);
        Y50 = _mm256_fmadd_ps(A1, B0, Y50);
        Y51 = _mm256_fmadd_ps(A1, B1, Y51);

        a += 6; b += 16;
    }
    _mm256_storeu_ps(y+0x00, Y00); _mm256_storeu_ps(y+0x08, Y01);
    _mm256_storeu_ps(y+0x10, Y10); _mm256_storeu_ps(y+0x18, Y11);
    _mm256_storeu_ps(y+0x20, Y20); _mm256_storeu_ps(y+0x28, Y21);
    _mm256_storeu_ps(y+0x30, Y30); _mm256_storeu_ps(y+0x38, Y31);
    _mm256_storeu_ps(y+0x40, Y40); _mm256_storeu_ps(y+0x48, Y41);
    _mm256_storeu_ps(y+0x50, Y50); _mm256_storeu_ps(y+0x58, Y51);
}


// GEMM block (Y[way*6, 16] = A[way, k, 6] * B[k, 16])
static void _gemm_block_6x16(const int way, const int k,
                             const float *a, const float *b,
                             float *y)
{
    const int step = _KERNEL_K;
    int z = k;

    while (z >= step) {
        for (int i=0; i<way; i++) {
            _gemm_kernel_6x16(step, a + i*k*6, b, y + i*6*16);
        }
        a += step * 6;
        b += step * 16;
        z -= step;
    }
    if (z) {
        for (int i=0; i<way; i++) {
            _gemm_kernel_6x16(z, a + i*k*6, b, y + i*6*16);
        }
    }
}


// GEMM(transA=0, transB=1) for m < 6 case
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
    if (m < _KERNEL_M && trans_a == 0 && trans_b == 1) {
         _plain_gemm_NT_f32(m, n, k, a, lda, b, ldb, c, ldc, bias, relu);
    } else {
        const int _g = _KERNEL_G;
        const int _m = _KERNEL_M;
        const int _n = _KERNEL_N;

        const int n16 = (n + 15) & (~15);
        const int inc_a = trans_a ? 1 : lda;
        float *pack_b = sf_malloc(alloc, k * n16 * sizeof(float));
        vm_pack_mat(trans_b, k, n, b, ldb, pack_b, _n);

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int y=0; y<m; y+=_g*_m) {
            float pack[_g * k * _m] __attribute__((aligned(32)));
            float tmp[_g * _m * _n] __attribute__((aligned(32)));
            const int dy = (m - y) < _g*_m ? (m - y) : _g*_m;
            vm_pack_mat(!trans_a, k, dy, a + y*inc_a, lda, pack, _m);

            for (int x=0; x<n; x+=_n) {
                const int dx = (n - x) < _n ? (n - x) : _n;
                _broadcast_bias_16(bias, x, _g*_m, tmp);
                _gemm_block_6x16(_g, k, pack, pack_b + x*k, tmp);
                _copy_mat_relu(dy, dx, tmp, _n, c + y*ldc + x, ldc, relu);
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
    const int _g = _KERNEL_G;
    const int _m = _KERNEL_M;
    const int _n = _KERNEL_N;

    const int n16 = (n + 15) & (~15);
    float *pack_b = sf_malloc(alloc, k * n16 * sizeof(float));
    vm_pack_mat(1, k, n, b, ldb, pack_b, _n);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int y=0; y<m; y+=_g*_m) {
        float pack[_g * k * _m] __attribute__((aligned(32)));
        float tmp[_g * _m * _n] __attribute__((aligned(32)));
        const int dy = (m - y) < _g*_m ? (m - y) : _g*_m;

        for (int i=0; i<_g; i++) {
            vm_pack_imat_6x(a, y + i*_m, pack + i*k*_m);
        }
        for (int x=0; x<n; x+=_n) {
            const int dx = (n - x) < _n ? (n - x) : _n;
            _broadcast_bias_16(bias, x, _g*_m, tmp);
            _gemm_block_6x16(_g, k, pack, pack_b + x*k, tmp);
            _copy_mat_relu(dy, dx, tmp, _n, c + y*ldc + x, ldc, relu);
        }
    }
    sf_free(pack_b);
}


// GEMM with implicit matrix A and NK16-packed matrix B
void vm_implicit_packed_gemm_f32(struct sf_allocator *alloc, int m, int n, int k,
                                 struct vm_imat *a, const float *b,
                                 float *c, int ldc, const float *bias, int relu)
{
    const int _g = _KERNEL_G;
    const int _m = _KERNEL_M;
    const int _n = _KERNEL_N;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int y=0; y<m; y+=_g*_m) {
        float pack[_g * k * _m] __attribute__((aligned(32)));
        float tmp[_g * _m * _n] __attribute__((aligned(32)));
        const int dy = (m - y) < _g*_m ? (m - y) : _g*_m;

        for (int i=0; i<_g; i++) {
            vm_pack_imat_6x(a, y + i*_m, pack + i*k*_m);
        }
        for (int x=0; x<n; x+=_n) {
            const int dx = (n - x) < _n ? (n - x) : _n;
            _broadcast_bias_16(bias, x, _g*_m, tmp);
            _gemm_block_6x16(_g, k, pack, b + x*k, tmp);
            _copy_mat_relu(dy, dx, tmp, _n, c + y*ldc + x, ldc, relu);
        }
    }
}


