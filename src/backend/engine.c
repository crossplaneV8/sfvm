
#include "backend.h"


// get input data address by name
void *sf_get_input_addr(struct sf_engine *engine, const char *name)
{
    for (int i=0; i<engine->i_cnt; i++) {
        if (strcmp(engine->i_names[i], name) == 0) {
            return engine->addr[engine->i_regs[i]];
        }
    }
    return NULL;
}


// get output data address by index
void *sf_get_output_addr(struct sf_engine *engine, int index)
{
    if (index >= 0 && index < engine->o_cnt) {
        return engine->addr[engine->o_regs[index]];
    }
    return NULL;
}


// execute codes on virtual machine
void sf_engine_run(struct sf_engine *engine)
{
    void **addr = engine->addr;
    const int *pc = engine->vm_code;

    while (1) {
        switch (*pc++) {
            case VM_STOP: return;
            case VM_ADD_1D_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                float *z = addr[*pc++];
                int nx = *pc++, ny = *pc++;
                vm_add_1d_f32(x, y, z, nx, ny);
                break;
            }
            case VM_ADD_2D_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                float *z = addr[*pc++];
                int hx = *pc++, wx = *pc++;
                int hy = *pc++, wy = *pc++;
                vm_add_2d_f32(x, y, z, hx, wx, hy, wy);
                break;
            }
            case VM_ADD_3D_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                float *z = addr[*pc++];
                int hx = *pc++, wx = *pc++, cx = *pc++;
                int hy = *pc++, wy = *pc++, cy = *pc++;
                vm_add_3d_f32(x, y, z, hx, wx, cx, hy, wy, cy);
                break;
            }
            case VM_ADD_4D_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                float *z = addr[*pc++];
                int nx = *pc++, hx = *pc++, wx = *pc++, cx = *pc++;
                int ny = *pc++, hy = *pc++, wy = *pc++, cy = *pc++;
                vm_add_4d_f32(x, y, z, nx, hx, wx, cx, ny, hy, wy, cy);
                break;
            }
            case VM_CONV_NHWC_OHWI_F32: {
                float *x = addr[*pc++];
                float *w = addr[*pc++];
                float *b = addr[*pc++];
                float *y = addr[*pc++];
                int ni = *pc++, hi = *pc++, wi = *pc++, ci = *pc++;
                int no = *pc++, ho = *pc++, wo = *pc++, co = *pc++;
                int ph = *pc++, pw = *pc++, sh = *pc++, sw = *pc++;
                int kh = *pc++, kw = *pc++, dh = *pc++, dw = *pc++;
                int relu = *pc++;
                vm_conv_nhwc_ohwi_f32(engine->alloc, x, w, b, y,
                                      ni, hi, wi, ci, no, ho, wo, co,
                                      ph, pw, sh, sw, kh, kw, dh, dw, relu);
                break;
            }
            case VM_MAX_POOL_NHWC_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                int ni = *pc++, hi = *pc++, wi = *pc++, ci = *pc++;
                int no = *pc++, ho = *pc++, wo = *pc++, co = *pc++;
                int ph = *pc++, pw = *pc++, sh = *pc++, sw = *pc++;
                int kh = *pc++, kw = *pc++;
                vm_maxpool_nhwc_f32(x, y, ni, hi, wi, ci, no, ho,
                                    wo, co, ph, pw, sh, sw, kh, kw);
                break;
            }
            case VM_GAVG_POOL_NHWC_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                int n = *pc++, h = *pc++;
                int w = *pc++, c = *pc++;
                vm_gl_avgpool_nhwc_f32(x, y, n, h, w, c);
                break;
            }
            case VM_RELU_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                int num = *pc++;
                vm_relu_f32(x, y, num);
                break;
            }
            case VM_COPY_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                int num = *pc++;
                vm_copy_f32(x, y, num);
                break;
            }
            case VM_TRANSPOSE_2D_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                int hi = *pc++, wi = *pc++;
                int a0 = *pc++, a1 = *pc++;
                vm_transpose_2d_f32(x, y, hi, wi, a0, a1);
                break;
            }
            case VM_TRANSPOSE_3D_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                int hi = *pc++, wi = *pc++, ci = *pc++;
                int a0 = *pc++, a1 = *pc++, a2 = *pc++;
                vm_transpose_3d_f32(x, y, hi, wi, ci, a0, a1, a2);
                break;
            }
            case VM_TRANSPOSE_4D_F32: {
                float *x = addr[*pc++];
                float *y = addr[*pc++];
                int ni = *pc++, hi = *pc++, wi = *pc++, ci = *pc++;
                int a0 = *pc++, a1 = *pc++, a2 = *pc++, a3 = *pc++;
                vm_transpose_4d_f32(x, y, ni, hi, wi, ci, a0, a1, a2, a3);
                break;
            }
            case VM_GEMM_F32: {
                float *a = addr[*pc++];
                float *b = addr[*pc++];
                float *c = addr[*pc++];
                float *y = addr[*pc++];
                int trans_a = *pc++, trans_b = *pc++;
                int m = *pc++, n = *pc++, k = *pc++;
                int lda = (trans_a ? m : k), ldb = (trans_b ? k : n);
                vm_gemm_f32(engine->alloc, trans_a, trans_b,
                            m, n, k, a, lda, b, ldb, y, n, c, 0);
                break;
            }
        }
    }
}


// print vm code
void sf_print_code(FILE *f, struct sf_engine *engine)
{
    const int *pc = engine->vm_code;
    const int *end = pc + engine->code_cnt;

    while (pc < end) {
        const char *name = NULL;
        int regs = 0, args = 0;
        switch (*pc++) {
            case VM_STOP:               name = "VM_STOP";               regs = 0, args = 0; break;
            case VM_ADD_1D_F32:         name = "VM_ADD_1D_F32";         regs = 3; args = 2; break;
            case VM_ADD_2D_F32:         name = "VM_ADD_2D_F32";         regs = 3; args = 4; break;
            case VM_ADD_3D_F32:         name = "VM_ADD_3D_F32";         regs = 3; args = 6; break;
            case VM_ADD_4D_F32:         name = "VM_ADD_4D_F32";         regs = 3; args = 8; break;
            case VM_CONV_NHWC_OHWI_F32: name = "VM_CONV_NHWC_OHWI_F32"; regs = 4; args = 17; break;
            case VM_MAX_POOL_NHWC_F32:  name = "VM_MAX_POOL_NHWC_F32";  regs = 2; args = 14; break;
            case VM_GAVG_POOL_NHWC_F32: name = "VM_GAVG_POOL_NHWC_F32"; regs = 2; args = 4; break;
            case VM_RELU_F32:           name = "VM_RELU_F32";           regs = 2; args = 1; break;
            case VM_COPY_F32:           name = "VM_COPY_F32";           regs = 2; args = 1; break;
            case VM_TRANSPOSE_2D_F32:   name = "VM_TRANSPOSE_2D_F32";   regs = 2; args = 4; break;
            case VM_TRANSPOSE_3D_F32:   name = "VM_TRANSPOSE_3D_F32";   regs = 2; args = 6; break;
            case VM_TRANSPOSE_4D_F32:   name = "VM_TRANSPOSE_4D_F32";   regs = 2; args = 8; break;
            case VM_GEMM_F32:           name = "VM_GEMM_F32";           regs = 4; args = 5; break;
            default: return;
        }
        if (name != NULL) {
            fprintf(f, name);
            for (int i=0; i<regs; i++) {
                fprintf(f, " R%d,", *pc++);
            }
            for (int i=0; i<args; i++) {
                fprintf(f, " %d,", *pc++);
            }
            fprintf(f, "\n");
        }
    }
    fprintf(f, "\n");
}
