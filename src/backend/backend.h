
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include "graph/graph.h"
#include "compute_lib/compute_lib.h"


// register info
struct sf_reg_info
{
    int rank;
    int size;
    int ref_cnt;
    void *data;
};


// inference engine
struct sf_engine
{
    struct sf_allocator *alloc;     // mempool allocator

    int reg_cnt, reg_len;
    struct sf_reg_info *info;       // list of register info
    void **addr;                    // list of register addr
    int *reg_map;                   // node index ==> register

    int i_cnt, o_cnt;
    char **i_names;                 // input names
    int *i_regs;                    // input registers
    int *o_regs;                    // output registers

    int code_cnt, code_len;         // code length
    int *vm_code;                   // vm code
};


// VM instructions
enum sf_vm_instr
{
    VM_STOP,
    VM_ADD_1D_F32,
    VM_ADD_2D_F32,
    VM_ADD_3D_F32,
    VM_ADD_4D_F32,
    VM_ADD_RELU_F32,
    VM_CONV_NHWC_OHWI_F32,
    VM_CONV_NHWC_NK16_F32,
    VM_MAX_POOL_NHWC_F32,
    VM_GAVG_POOL_NHWC_F32,
    VM_RELU_F32,
    VM_COPY_F32,
    VM_TRANSPOSE_2D_F32,
    VM_TRANSPOSE_3D_F32,
    VM_TRANSPOSE_4D_F32,
    VM_GEMM_F32,
};


// generate inference engine from graph
struct sf_engine *sf_engine_from_graph(struct sf_graph *graph);

// clone an existing engine (share same data)
struct sf_engine *sf_clone_engine(struct sf_engine *engine);

// discard inference engine
void sf_discard_engine(struct sf_engine *engine);

// get input data address by name
void *sf_get_input_addr(struct sf_engine *engine, const char *name);

// get output data address by index
void *sf_get_output_addr(struct sf_engine *engine, int index);

// execute code on virtual machine
void sf_engine_run(struct sf_engine *engine);

// print vm code
void sf_print_code(FILE *f, struct sf_engine *engine);


#ifdef __cplusplus
    }
#endif


