
#include "compute_lib.h"


// global average pooling (NHWC layout)
void vm_gl_avgpool_nhwc_f32(float *x, float *y, int n, int h, int w, int c)
{
    const int hw = h * w, hwc = h * w * c;
    const float scale = 1.0 / hw;

    for (int i=0; i<n; i++) {
        float *px = x + i * hwc;
        float *py = y + i * c;
        int m = c;
#ifdef __AVX2__
        const __m256 K = _mm256_set1_ps(scale);

        while (m >= 32) {
            __m256 S0 = _mm256_setzero_ps();
            __m256 S1 = _mm256_setzero_ps();
            __m256 S2 = _mm256_setzero_ps();
            __m256 S3 = _mm256_setzero_ps();

            for (int j=0; j<hw; j++) {
                S0 += _mm256_loadu_ps(px + j*c + 0);
                S1 += _mm256_loadu_ps(px + j*c + 8);
                S2 += _mm256_loadu_ps(px + j*c + 16);
                S3 += _mm256_loadu_ps(px + j*c + 24);
            }
            _mm256_storeu_ps(py + 0, K*S0);
            _mm256_storeu_ps(py + 8, K*S1);
            _mm256_storeu_ps(py + 16, K*S2);
            _mm256_storeu_ps(py + 24, K*S3);
            px += 32; py += 32; m -= 32;
        }
#endif
        if (m > 0) {
            for (int j=0; j<m; j++) {py[j] = 0;}
            for (int k=0; k<hw; k++) {
                for (int j=0; j<m; j++) {
                    py[j] += px[k*c + j];
                }
            }
            for (int j=0; j<m; j++) {py[j] *= scale;}
        }
    }
}



static void _roi_maxpool_hwc_f32(float *x, float *y,
                                 int hi, int wi, int ci,
                                 int h0, int h1, int w0, int w1)
{
    int m = ci;
#ifdef __AVX2__
    while (m >= 32) {
        __m256 R0 = _mm256_set1_ps(-1e20);
        __m256 R1 = _mm256_set1_ps(-1e20);
        __m256 R2 = _mm256_set1_ps(-1e20);
        __m256 R3 = _mm256_set1_ps(-1e20);

        for (int h=h0; h<h1; h++) {
            for (int w=w0; w<w1; w++) {
                const float *px = x + h*wi*ci + w*ci;
                __m256 X0 = _mm256_loadu_ps(px + 0);
                __m256 X1 = _mm256_loadu_ps(px + 8);
                __m256 X2 = _mm256_loadu_ps(px + 16);
                __m256 X3 = _mm256_loadu_ps(px + 24);
                R0 = _mm256_max_ps(R0, X0);
                R1 = _mm256_max_ps(R1, X1);
                R2 = _mm256_max_ps(R2, X2);
                R3 = _mm256_max_ps(R3, X3);
            }
        }
        _mm256_storeu_ps(y + 0, R0);
        _mm256_storeu_ps(y + 8, R1);
        _mm256_storeu_ps(y + 16, R2);
        _mm256_storeu_ps(y + 24, R3);
        x += 32; y += 32; m -= 32;
    }
#endif
    if (m > 0) {
        for (int i=0; i<m; i++) {y[i] = -1e20;}
        for (int h=h0; h<h1; h++) {
            for (int w=w0; w<w1; w++) {
                const float *px = x + h*wi*ci + w*ci;
                for (int i=0; i<m; i++) {
                    y[i] = px[i] > y[i] ? px[i] : y[i];
                }
            }
        }
    }
}


// max pooling (NHWC layout)
void vm_maxpool_nhwc_f32(float *x, float *y,
                         int ni, int hi, int wi, int ci,
                         int no, int ho, int wo, int co,
                         int ph, int pw, int sh, int sw,
                         int kh, int kw)
{
    for (int n=0; n<no; n++) {
        float *px = x + n * hi * wi * ci;

        for (int h=0; h<ho; h++) {
            for (int w=0; w<wo; w++) {
                int h0 = h * sh - ph, h1 = h0 + kh;
                int w0 = w * sw - pw, w1 = w0 + kw;
                h0 = h0 > 0 ? h0 : 0;
                w0 = w0 > 0 ? w0 : 0;
                h1 = h1 < hi ? h1 : hi;
                w1 = w1 < wi ? w1 : wi;

                _roi_maxpool_hwc_f32(px, y, hi, wi, ci, h0, h1, w0, w1);
                y += co;
            }
        }
    }
}


