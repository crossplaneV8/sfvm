
#include "compute_lib.h"



// separate zero padding from small channel convolution
static float *_zero_pad(struct sf_allocator *alloc, int n, int h0, int w0,
                        int c, int ph, int pw, int h1, int w1, float *src)
{
    float *buf = sf_malloc(alloc, n*h1*w1*c*sizeof(float));

    for (int i=0; i<n; i++) {
        float *dst = buf + i*h1*w1*c + ph*w1*c + pw*c;
        vm_clear_f32(buf + i*h1*w1*c, h1*w1*c);

        for (int j=0; j<h0; j++) {
            vm_copy_f32(src, dst, w0 * c);
            src += w0 * c;
            dst += w1 * c;
        }
    }
    return buf;
}


// generate implicit matrix
static struct vm_imat *_gen_imat(struct sf_allocator *alloc, float *data,
                                 int ni, int hi, int wi, int ci,
                                 int no, int ho, int wo, int co,
                                 int ph, int pw, int sh, int sw,
                                 int kh, int kw, int dh, int dw)
{
    struct vm_imat *mat = sf_malloc(alloc, sizeof(struct vm_imat));
    memset(mat, 0, sizeof(struct vm_imat));

    if ((ph > 0 || pw > 0) && dh == 1 && dw == 1 && ci < 8) {
        const int h1 = kh + (ho - 1) * sh;
        const int w1 = kw + (wo - 1) * sw;
        data = _zero_pad(alloc, ni, hi, wi, ci, ph, pw, h1, w1, data);
        mat->temp = data;
        hi = h1; ph = 0;
        wi = w1; pw = 0;
    }
    if (ph == 0 && pw == 0 && dh == 1 && dw == 1) {
        mat->segs = kh;
        mat->len = kw * ci;
    } else {
        mat->segs = kh * kw;
        mat->len = ci;
    }
    mat->rows = no * ho * wo;
    mat->cols = kh * kw * ci;
    mat->hi = hi; mat->wi = wi;
    mat->ho = ho; mat->wo = wo;
    mat->ci = ci; mat->co = co;
    mat->ph = ph; mat->pw = pw;
    mat->sh = sh; mat->sw = sw;
    mat->kh = kh; mat->kw = kw;
    mat->dh = dh; mat->dw = dw;

    mat->pads = sf_malloc(alloc, mat->len * sizeof(float));
    vm_clear_f32(mat->pads, mat->len);
    mat->data = data;

    return mat;
}


// free memory of implicit matrix
static void _discard_imat(struct vm_imat *mat)
{
    if (mat != NULL) {
        if (mat->temp != NULL) {
            sf_free(mat->temp);
        }
        if (mat->pads != NULL) {
            sf_free(mat->pads);
        }
        sf_free(mat);
    }
}


// get data index of rows in the implicit matrix
static void _get_imat_rows(struct vm_imat *mat, const float *idx[], int begin, int rows)
{
    const int hi = mat->hi, wi = mat->wi;
    const int ph = mat->ph, pw = mat->pw;
    const int sh = mat->sh, sw = mat->sw;
    const int kh = mat->kh, kw = mat->kw;
    const int dh = mat->dh, dw = mat->dw;
    const int ostep1 = mat->wo;
    const int ostep2 = mat->ho * ostep1;
    const int istep0 = mat->ci;
    const int istep1 = mat->wi * istep0;
    const int istep2 = mat->hi * istep1;
    const int segs = mat->segs;
    int dy[segs], dx[segs];

    for (int i=0; i<segs; i++) {
        const int s = i * (kh * kw / segs);
        const int ky = s/kw, kx = s - ky*kw;
        dy[i] = ky*dh - ph; dx[i] = kx*dw - pw;
    }
    for (int i=0; i<rows; i++) {
        const int s = begin + i;
        for (int j=0; j<segs; j++) {
            idx[j*rows + i] = mat->pads;
        }
        if ((unsigned)s < (unsigned)(mat->rows)) {
            const int n = s/ostep2, r = s - n*ostep2;
            const int h = r/ostep1, w = r - h*ostep1;
            for (int j=0; j<segs; j++) {
                const int y = h*sh + dy[j], x = w*sw + dx[j];
                if ((unsigned)y < (unsigned)hi && (unsigned)x < (unsigned)wi) {
                    idx[j*rows + i] = mat->data + n*istep2 + y*istep1 + x*istep0;
                }
            }
        }
    }
}


static void _transpose_16x(int len, const float *src[16], float *dst)
{
    if (len >= 8) {
        __m256 S0, S1, S2, S3, S4, S5, S6, S7;
        __m256 T0, T1, T2, T3, T4, T5, T6, T7;

        for (int i=0; i<len; i+=8) {
            int k = i < (len - 8) ? i : (len - 8);
            for (int j=0; j<16; j+=8) {
                float *q = dst + k*16 + j;

                S0 = _mm256_loadu_ps(src[j + 0] + k);
                S1 = _mm256_loadu_ps(src[j + 1] + k);
                S2 = _mm256_loadu_ps(src[j + 2] + k);
                S3 = _mm256_loadu_ps(src[j + 3] + k);
                S4 = _mm256_loadu_ps(src[j + 4] + k);
                S5 = _mm256_loadu_ps(src[j + 5] + k);
                S6 = _mm256_loadu_ps(src[j + 6] + k);
                S7 = _mm256_loadu_ps(src[j + 7] + k);

                T0 = _mm256_unpacklo_ps(S0, S1);
                T1 = _mm256_unpacklo_ps(S2, S3);
                T2 = _mm256_unpacklo_ps(S4, S5);
                T3 = _mm256_unpacklo_ps(S6, S7);
                T4 = _mm256_unpackhi_ps(S0, S1);
                T5 = _mm256_unpackhi_ps(S2, S3);
                T6 = _mm256_unpackhi_ps(S4, S5);
                T7 = _mm256_unpackhi_ps(S6, S7);

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

                _mm256_storeu_ps(q + 0*16, T0);
                _mm256_storeu_ps(q + 1*16, T1);
                _mm256_storeu_ps(q + 2*16, T2);
                _mm256_storeu_ps(q + 3*16, T3);
                _mm256_storeu_ps(q + 4*16, T4);
                _mm256_storeu_ps(q + 5*16, T5);
                _mm256_storeu_ps(q + 6*16, T6);
                _mm256_storeu_ps(q + 7*16, T7);
            }
        }
    } else {
        for (int i=0; i<16; i++) {
            for (int j=0; j<len; j++) {
                dst[j*16 + i] = src[i][j];
            }
        }
    }
}


// slice imat[y:y+16, 0:k], transpose to [k, 16]
void vm_pack_imat_16x(struct vm_imat *mat, int y, float *dst)
{
    const int segs = mat->segs, len = mat->len;
    const float *data[segs][16];
    _get_imat_rows(mat, (void*)data, y, 16);

    for (int s=0; s<segs; s++) {
        _transpose_16x(len, data[s], dst);
        dst += len * 16;
    }
}


// convolution (data: NHWC, weight: OHWI)
void vm_conv_nhwc_ohwi_f32(struct sf_allocator *alloc, float *x, float *w, float *b, float *y,
                           int ni, int hi, int wi, int ci, int no, int ho, int wo, int co,
                           int ph, int pw, int sh, int sw, int kh, int kw, int dh, int dw, int relu)
{
    struct vm_imat *mat = _gen_imat(alloc, x, ni, hi, wi, ci, no, ho, wo,
                                    co, ph, pw, sh, sw, kh, kw, dh, dw);
    vm_implicit_gemm_f32(alloc, mat->rows, co, mat->cols, mat, w, mat->cols, y, co, b, relu);
    _discard_imat(mat);
}


// convolution (data: NHWC, weight: NK16-packed)
void vm_conv_nhwc_nk16_f32(struct sf_allocator *alloc, float *x, float *w, float *b, float *y,
                           int ni, int hi, int wi, int ci, int no, int ho, int wo, int co,
                           int ph, int pw, int sh, int sw, int kh, int kw, int dh, int dw, int relu)
{
    struct vm_imat *mat = _gen_imat(alloc, x, ni, hi, wi, ci, no, ho, wo,
                                    co, ph, pw, sh, sw, kh, kw, dh, dw);
    vm_implicit_packed_gemm_f32(alloc, mat->rows, co, mat->cols, mat, w, y, co, b, relu);
    _discard_imat(mat);
}


