
#include "compute_lib.h"


// z[i] = x[i] + y
static inline void _scalar_add_f32(float *x, float y, float *z, int num)
{
#ifdef __AVX2__
    __m256 Y = _mm256_set1_ps(y);
    while (num >= 16) {
        __m256 X0 = _mm256_loadu_ps(x + 0);
        __m256 X1 = _mm256_loadu_ps(x + 8);
        _mm256_storeu_ps(z + 0, X0 + Y);
        _mm256_storeu_ps(z + 8, X1 + Y);
        x += 16; y += 16; z += 16; num -= 16;
    }
#endif
    for (int i=0; i<num; i++) {
        z[i] = x[i] + y;
    }
}


// z[i] = x[i] + y[i]
static inline void _vector_add_f32(float *x, float *y, float *z, int num)
{
#ifdef __AVX2__
    while (num >= 16) {
        __m256 X0 = _mm256_loadu_ps(x + 0);
        __m256 X1 = _mm256_loadu_ps(x + 8);
        __m256 Y0 = _mm256_loadu_ps(y + 0);
        __m256 Y1 = _mm256_loadu_ps(y + 8);
        _mm256_storeu_ps(z + 0, X0 + Y0);
        _mm256_storeu_ps(z + 8, X1 + Y1);
        x += 16; y += 16; z += 16; num -= 16;
    }
#endif
    for (int i=0; i<num; i++) {
        z[i] = x[i] + y[i];
    }
}


// broadcast add 1d tensor
void vm_add_1d_f32(float *x, float *y, float *z, int nx, int ny)
{
    int nz = nx > ny ? nx : ny;

    if (nx < nz) {
        _scalar_add_f32(y, x[0], z, nz);
    } else if (ny < nz) {
        _scalar_add_f32(x, y[0], z, nz);
    } else {
        _vector_add_f32(x, y, z, nz);
    }
}


// broadcast add 2d tensor
void vm_add_2d_f32(float *x, float *y, float *z,
                   int hx, int wx, int hy, int wy)
{
    int hz = (hx > hy ? hx : hy),   wz = (wx > wy ? wx : wy);
    int sxh =       wx*(hx == hz),  syh =       wy*(hy == hz);
    int sxw =          (wx == wz),  syw =          (wy == wz);

    for (int h=0; h<hz; h++) {
        float *px = x + h*sxh;
        float *py = y + h*syh;

        if (sxw == 0) {
            _scalar_add_f32(py, px[0], z, wz);
        } else if (syw == 0) {
            _scalar_add_f32(px, py[0], z, wz);
        } else {
            _vector_add_f32(px, py, z, wz);
        }
        z += wz;
    }
}


// broadcast add 3d tensor
void vm_add_3d_f32(float *x, float *y, float *z,
                   int hx, int wx, int cx,
                   int hy, int wy, int cy)
{
    int hz = (hx > hy ? hx : hy);
    int wz = (wx > wy ? wx : wy),   cz = (cx > cy ? cx : cy);
    int sxh =    wx*cx*(hx == hz),  syh =    wy*cy*(hy == hz);
    int sxw =       cx*(wx == wz),  syw =       cy*(wy == wz);
    int sxc =          (cx == cz),  syc =          (cy == cz);

    for (int h=0; h<hz; h++) {
        for (int w=0; w<wz; w++) {
            float *px = x + h*sxh + w*sxw;
            float *py = y + h*syh + w*syw;

            if (sxc == 0) {
                _scalar_add_f32(py, px[0], z, cz);
            } else if (syc == 0) {
                _scalar_add_f32(px, py[0], z, cz);
            } else {
                _vector_add_f32(px, py, z, cz);
            }
            z += cz;
        }
    }
}


// broadcast add 4d tensor
void vm_add_4d_f32(float *x, float *y, float *z,
                   int nx, int hx, int wx, int cx,
                   int ny, int hy, int wy, int cy)
{
    int nz = (nx > ny ? nx : ny),   hz = (hx > hy ? hx : hy);
    int wz = (wx > wy ? wx : wy),   cz = (cx > cy ? cx : cy);
    int sxn = hx*wx*cx*(nx == nz),  syn = hy*wy*cy*(ny == nz);
    int sxh =    wx*cx*(hx == hz),  syh =    wy*cy*(hy == hz);
    int sxw =       cx*(wx == wz),  syw =       cy*(wy == wz);
    int sxc =          (cx == cz),  syc =          (cy == cz);

    for (int n=0; n<nz; n++) {
        for (int h=0; h<hz; h++) {
            for (int w=0; w<wz; w++) {
                float *px = x + n*sxn + h*sxh + w*sxw;
                float *py = y + n*syn + h*syh + w*syw;

                if (sxc == 0) {
                    _scalar_add_f32(py, px[0], z, cz);
                } else if (syc == 0) {
                    _scalar_add_f32(px, py[0], z, cz);
                } else {
                    _vector_add_f32(px, py, z, cz);
                }
                z += cz;
            }
        }
    }
}


