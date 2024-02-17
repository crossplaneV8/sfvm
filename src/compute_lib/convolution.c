
#include "compute_lib.h"



// generate implicit matrix (padding > 0 or dilation > 1)
static struct vm_imat *_gen_imat_0(struct sf_allocator *alloc, float *data,
                                   int ni, int hi, int wi, int ci,
                                   int no, int ho, int wo, int co,
                                   int ph, int pw, int sh, int sw,
                                   int kh, int kw, int dh, int dw)
{
    struct vm_imat *mat = sf_malloc(alloc, sizeof(struct vm_imat));
    memset(mat, 0, sizeof(struct vm_imat));
    mat->rows = ni * ho * wo;
    mat->cols = kh * kw * ci;
    mat->segs = kh * kw;
    mat->len = ci;

    size_t size = (mat->segs * mat->rows * sizeof(float*) + 31) & (~31);
    float **idx = sf_malloc(alloc, size + ci * sizeof(float));
    float *pad = (float*)((uint8_t*)idx + size);
    vm_clear_f32(pad, ci);
    mat->data = idx;

    const int wc = wi*ci, hwc = hi*wi*ci;

    for (int ii=0; ii<kh; ii++) {
        for (int jj=0; jj<kw; jj++) {
            for (int n=0; n<ni; n++) {
                for (int i=0; i<ho; i++) {
                    for (int j=0; j<wo; j++) {
                        int y = i*sh + ii*dh - ph;
                        int x = j*sw + jj*dw - pw;
                        int fy = (y >= 0) & (y < hi);
                        int fx = (x >= 0) & (x < wi);
                        *idx++ = (fy & fx) ? (data + n*hwc + y*wc + x*ci) : pad;
                    }
                }
            }
        }
    }
    return mat;
}


// generate implicit matrix (padding = 0 and dilation = 1)
static struct vm_imat *_gen_imat_1(struct sf_allocator *alloc, float *data,
                                   int ni, int hi, int wi, int ci,
                                   int no, int ho, int wo, int co,
                                   int kh, int kw, int sh, int sw)
{
    struct vm_imat *mat = sf_malloc(alloc, sizeof(struct vm_imat));
    memset(mat, 0, sizeof(struct vm_imat));
    mat->rows = ni * ho * wo;
    mat->cols = kh * kw * ci;
    mat->segs = kh;
    mat->len = kw * ci;
    mat->data = sf_malloc(alloc, mat->segs * mat->rows * sizeof(float*));

    const int wc = wi*ci, hwc = hi*wi*ci;
    float **idx = mat->data;

    for (int k=0; k<kh; k++) {
        for (int n=0; n<ni; n++) {
            for (int i=0; i<ho; i++) {
                for (int j=0; j<wo; j++) {
                    *idx++ = data + n*hwc + i*sh*wc + j*sw*ci + k*wc;
                }
            }
        }
    }
    return mat;
}


// generate implicit matrix (kernel_w = input_w and padding = 0)
static struct vm_imat *_gen_imat_2(struct sf_allocator *alloc, float *data,
                                   int ni, int hi, int wi, int ci,
                                   int no, int ho, int wo, int co,
                                   int kh, int kw, int sh, int sw)
{
    struct vm_imat *mat = sf_malloc(alloc, sizeof(struct vm_imat));
    memset(mat, 0, sizeof(struct vm_imat));
    mat->rows = ni * ho * wo;
    mat->cols = kh * kw * ci;
    mat->segs = 1;
    mat->len = kh * kw * ci;
    mat->data = sf_malloc(alloc, mat->segs * mat->rows * sizeof(float*));

    const int hwc = hi*wi*ci, step = sh*wi*ci;
    float **idx = mat->data;

    for (int n=0; n<ni; n++) {
        for (int i=0; i<ho; i++) {
            *idx++ = data + n*hwc + i*step;
        }
    }
    return mat;
}


// generate implicit matrix
static struct vm_imat *_gen_imat(struct sf_allocator *alloc, float *data,
                                 int ni, int hi, int wi, int ci,
                                 int no, int ho, int wo, int co,
                                 int ph, int pw, int sh, int sw,
                                 int kh, int kw, int dh, int dw)
{
    if (ph == 0 && pw == 0 && dh == 1 && dw == 1) {
        if (kw == wi) {
            return _gen_imat_2(alloc, data, ni, hi, wi, ci, no, ho, wo, co, kh, kw, sh, sw);
        } else {
            return _gen_imat_1(alloc, data, ni, hi, wi, ci, no, ho, wo, co, kh, kw, sh, sw);
        }
    } else {
        return _gen_imat_0(alloc, data, ni, hi, wi, ci, no, ho, wo, co, ph, pw, sh, sw, kh, kw, dh, dw);
    }
}


// free memory of implicit matrix
static void _discard_imat(struct vm_imat *mat)
{
    if (mat != NULL) {
        if (mat->data != NULL) {
            sf_free(mat->data);
        }
        sf_free(mat);
    }
}


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


// convolution (data: NHWC, weight: OHWI)
void vm_conv_nhwc_ohwi_f32(struct sf_allocator *alloc, float *x, float *w, float *b, float *y,
                           int ni, int hi, int wi, int ci, int no, int ho, int wo, int co,
                           int ph, int pw, int sh, int sw, int kh, int kw, int dh, int dw, int relu)
{
    if ((ph > 0 || pw > 0) && ci < 8 && dh == 1 && dw == 1) {
        const int h1 = kh + (ho - 1) * sh;
        const int w1 = kw + (wo - 1) * sw;
        float *x_pad = _zero_pad(alloc, ni, hi, wi, ci, ph, pw, h1, w1, x);
        struct vm_imat *mat = _gen_imat(alloc, x_pad, ni, h1, w1, ci, no, ho,
                                        wo, co, 0, 0, sh, sw, kh, kw, dh, dw);
        vm_implicit_gemm_f32(alloc, mat->rows, co, mat->cols, mat, w, mat->cols, y, co, b, relu);
        _discard_imat(mat);
        sf_free(x_pad);
    }
    else {
        struct vm_imat *mat = _gen_imat(alloc, x, ni, hi, wi, ci, no, ho, wo,
                                        co, ph, pw, sh, sw, kh, kw, dh, dw);
        vm_implicit_gemm_f32(alloc, mat->rows, co, mat->cols, mat, w, mat->cols, y, co, b, relu);
        _discard_imat(mat);
    }
}


// convolution (data: NHWC, weight: NK16-packed)
void vm_conv_nhwc_nk16_f32(struct sf_allocator *alloc, float *x, float *w, float *b, float *y,
                           int ni, int hi, int wi, int ci, int no, int ho, int wo, int co,
                           int ph, int pw, int sh, int sw, int kh, int kw, int dh, int dw, int relu)
{
    if ((ph > 0 || pw > 0) && ci < 8 && dh == 1 && dw == 1) {
        const int h1 = kh + (ho - 1) * sh;
        const int w1 = kw + (wo - 1) * sw;
        float *x_pad = _zero_pad(alloc, ni, hi, wi, ci, ph, pw, h1, w1, x);
        struct vm_imat *mat = _gen_imat(alloc, x_pad, ni, h1, w1, ci, no, ho,
                                        wo, co, 0, 0, sh, sw, kh, kw, dh, dw);
        vm_implicit_packed_gemm_f32(alloc, mat->rows, co, mat->cols, mat, w, y, co, b, relu);
        _discard_imat(mat);
        sf_free(x_pad);
    }
    else {
        struct vm_imat *mat = _gen_imat(alloc, x, ni, hi, wi, ci, no, ho, wo,
                                        co, ph, pw, sh, sw, kh, kw, dh, dw);
        vm_implicit_packed_gemm_f32(alloc, mat->rows, co, mat->cols, mat, w, y, co, b, relu);
        _discard_imat(mat);
    }
}


