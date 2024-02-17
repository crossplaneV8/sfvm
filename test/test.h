
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#include "sfvm.h"


// calculate max absolute error
static inline double _max_error(const float *x, const float *y, int num)
{
    float e_max = 0;
    for (int i=0; i<num; i++) {
        float d = (x[i] > y[i]) ? (x[i] - y[i]) : (y[i] - x[i]);
        e_max = (d > e_max) ? d : e_max;
    }
    return e_max;
}


// calculate root mean square error
static inline double _rms_error(const float *x, const float *y, int num)
{
    double sum = 0;
    for (int i=0; i<num; i++) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(sum / num);
}


void test_conv(void);
void test_resnet(void);
void test_perf(void);


#ifdef __cplusplus
    }
#endif


