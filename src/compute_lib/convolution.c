
#include "compute_lib.h"



// implicit matrix
struct vm_imat
{
    int rows, cols;
    int segs, len;
    float **data;
};


// generate implicit matrix (padding > 0 or dilation > 1)
static struct vm_imat _gen_imat_0(struct sf_allocator *alloc, float *data,
                                  int ni, int hi, int wi, int ci,
                                  int no, int ho, int wo, int co,
                                  int ph, int pw, int sh, int sw,
                                  int kh, int kw, int dh, int dw)
{
    const int wc = wi*ci, hwc = hi*wi*ci;
    struct vm_imat mat = {
        .rows = ni * ho * wo,
        .cols = kh * kw * ci,
        .segs = kh * kw,
        .len = ci,
    };
    size_t size = (mat.rows*mat.segs*sizeof(float*) + 31) & (~31);
    mat.data = sf_malloc(alloc, size + mat.len*sizeof(float));
    float *pad = (float*)((uint8_t*)mat.data + size);
    float **pm = mat.data;
    vm_clear_f32(pad, ci);

    for (int n=0; n<ni; n++) {
        float *px = data + n*hwc;

        for (int i=0; i<ho; i++) {
            for (int j=0; j<wo; j++) {
                for (int ii=0; ii<kh; ii++) {
                    int y = i*sh + ii*dh - ph;
                    int vy = (y >= 0) & (y < hi);

                    for (int jj=0; jj<kw; jj++) {
                        int x = j*sw + jj*dw - pw;
                        int vx = vy & (x >= 0) & (x < wi);
                        *pm++ = vx ? (px + y*wc + x*ci) : pad;
                    }
                }
            }
        }
    }
    return mat;
}


// generate implicit matrix (padding = 0 and dilation = 1)
static struct vm_imat _gen_imat_1(struct sf_allocator *alloc, float *data,
                                  int ni, int hi, int wi, int ci,
                                  int no, int ho, int wo, int co,
                                  int kh, int kw, int sh, int sw)
{
    const int wc = wi*ci, hwc = hi*wi*ci;
    struct vm_imat mat = {
        .rows = ni * ho * wo,
        .cols = kh * kw * ci,
        .segs = kh,
        .len = kw * ci,
    };
    mat.data = sf_malloc(alloc, mat.rows*mat.segs*sizeof(float*));
    float **pm = mat.data;

    for (int n=0; n<ni; n++) {
        for (int i=0; i<ho; i++) {
            float *px = data + n*hwc + i*sh*wc;

            for (int j=0; j<wo; j++) {
                for (int k=0; k<kh; k++) {
                    *pm++ = px + j*sw*ci + k*wc;
                }
            }
        }
    }
    return mat;
}


// generate implicit matrix (kernel_w = input_w and padding = 0)
static struct vm_imat _gen_imat_2(struct sf_allocator *alloc, float *data,
                                  int ni, int hi, int wi, int ci,
                                  int no, int ho, int wo, int co,
                                  int kh, int kw, int sh, int sw)
{
    const int hwc = hi*wi*ci, step = sh*wi*ci;
    struct vm_imat mat = {
        .rows = ni * ho * wo,
        .cols = kh * kw * ci,
        .segs = 1,
        .len = kh * kw * ci,
    };
    mat.data = sf_malloc(alloc, mat.rows*mat.segs*sizeof(float*));
    float **pm = mat.data;

    for (int n=0; n<ni; n++) {
        for (int i=0; i<ho; i++) {
            *pm++ = data + n*hwc + i*step;
        }
    }
    return mat;
}


// generate implicit matrix
static struct vm_imat _gen_imat(struct sf_allocator *alloc, float *data,
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


// convolution (NHWC OHWI layout)
void vm_conv_nhwc_ohwi_f32(struct sf_allocator *alloc, float *x, float *w, float *b, float *y,
                           int ni, int hi, int wi, int ci, int no, int ho, int wo, int co,
                           int ph, int pw, int sh, int sw, int kh, int kw, int dh, int dw, int relu)
{
    struct vm_imat mat = _gen_imat(alloc, x, ni, hi, wi, ci, no, ho,
                                   wo, co, ph, pw, sh, sw, kh, kw, dh, dw);
    vm_implicit_gemm_f32(alloc, mat.rows, co, mat.cols, (void*)mat.data,
                         mat.segs, mat.len, w, mat.cols, y, co, b, relu);
    sf_free(mat.data);
}


