
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include "base_struct.h"
#include "mem_alloc.h"


#define SF_MAX_DIMS         (8)
#define SF_MAX_ARGS         (32)
#define SF_MAX_STR_LEN      (256)


// numerical type
enum sf_data_type
{
    SF_UNKNOWN,
    SF_FLOAT16,
    SF_FLOAT32,
    SF_FLOAT64,
    SF_INT8,
    SF_INT16,
    SF_INT32,
    SF_INT64,
};


// operator type
enum sf_op_type
{
    OP_UNKNOWN,
    OP_INPUT,
    OP_CONST,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_CONV,
    OP_AVG_POOL,
    OP_MAX_POOL,
    OP_G_AVG_POOL,
    OP_G_MAX_POOL,
    OP_BATCHNORM,
    OP_IDENTITY,
    OP_RELU,
    OP_SIGMOID,
    OP_SOFTMAX,
    OP_SLICE,
    OP_CONCAT,
    OP_FLATTEN,
    OP_SQUEEZE,
    OP_RESHAPE,
    OP_TRANSPOSE,
    OP_REDUCE_SUM,
    OP_REDUCE_AVG,
    OP_REDUCE_VAR,
    OP_REDUCE_STD,
    OP_REDUCE_MIN,
    OP_REDUCE_MAX,
    OP_CAST,
    OP_GEMM,
};


// tensor descriptor
struct sf_tensor_desc
{
    enum sf_data_type dtype;
    int num_dims;
    int shape[SF_MAX_DIMS];
};


// node of DAG (directed acyclic graph)
struct sf_node
{
    enum sf_op_type op_type;            // type of operator

    int num_args;                       // number of input arguments
    struct sf_node *args[SF_MAX_ARGS];  // list of input arguments

    struct sf_tensor_desc o_desc;       // descriptor of output tensor

    union   // attributes
    {
        struct {    // input node
            char name[SF_MAX_STR_LEN];
            struct sf_tensor_desc data_desc;
        } input_attrs;

        struct {    // constant node
            void *shared_data;  // memory shared between nodes with a ref-cnt
            struct sf_tensor_desc data_desc;
        } const_attrs;

        struct {    // convolution node
            char x_layout[SF_MAX_DIMS];
            char w_layout[SF_MAX_DIMS];
            int pad_h0, pad_h1, pad_w0, pad_w1;
            int stride_h, stride_w;
            int dilate_h, dilate_w;
            int has_bias, has_relu;
        } conv_attrs;

        struct {    // pooling node
            int pad_h0, pad_h1, pad_w0, pad_w1;
            int stride_h, stride_w;
            int kernel_h, kernel_w;
            char layout[SF_MAX_DIMS];
        } pool_attrs;

        struct {    // batch-norm node
            double epsilon;
            char layout[SF_MAX_DIMS];
        } bn_attrs;

        struct {    // softmax, concat, flatten
            int axis;   // allows negative
        } axis_attrs;

        struct {    // slice node
            int num_dims;
            int start[SF_MAX_DIMS];
            int shape[SF_MAX_DIMS];
        } slice_attrs;

        struct {    // squeeze node
            int num_axes;
            int axes[SF_MAX_DIMS];  // allows negative
        } squeeze_attrs;

        struct {    // reshape node
            int num_dims;
            int shape[SF_MAX_DIMS]; // allows -1
        } reshape_attrs;

        struct {    // transpose node
            int num_dims;
            int axes[SF_MAX_DIMS];
        } transpose_attrs;

        struct {    // reduce node
            int num_axes;
            int axes[SF_MAX_DIMS];  // allows negative
            int keep_dims;
        } reduce_attrs;

        struct {    // cast node
            enum sf_data_type dtype;
        } cast_attrs;

        struct {    // gemm node
            float alpha, beta;
            int trans_a, trans_b;
        } gemm_attrs;
    };
};


// DAG (directed acyclic graph)
struct sf_graph
{
    struct sf_allocator *alloc;     // memory allocator for all nodes and constant data
    struct sf_list *outputs;        // list of output nodes
    struct sf_list *nodes;          // list of all graph nodes
};


// get string name of data type
const char *sf_get_dtype_name(enum sf_data_type dtype);

// get string name of node op type
const char *sf_get_op_name(struct sf_node *node);

// calculate tensor size (in bytes)
size_t sf_tensor_size(struct sf_tensor_desc desc);


// clone an existing node into the graph
struct sf_node *sf_clone_node(struct sf_graph *graph, struct sf_node *node,
                              struct sf_node **new_args);

// create a new input node
struct sf_node *sf_create_input_node(struct sf_graph *graph, const char *name, struct sf_tensor_desc desc);

// create a new constant node
struct sf_node *sf_create_const_node(struct sf_graph *graph, struct sf_tensor_desc desc, void *shared_data);

// create a new add node
struct sf_node *sf_create_add_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y);

// create a new subtract node
struct sf_node *sf_create_sub_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y);

// create a new multiply node
struct sf_node *sf_create_mul_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y);

// create a new divide node
struct sf_node *sf_create_div_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y);

// create a new convolution node
struct sf_node *sf_create_conv_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *w,
                                    struct sf_node *b, const char *x_layout, const char *w_layout,
                                    int has_relu, int pad_h0, int pad_h1, int pad_w0, int pad_w1,
                                    int stride_h, int stride_w, int dilate_h, int dilate_w);

// create a new pooling node
struct sf_node *sf_create_pool_node(struct sf_graph *graph, struct sf_node *x,
                                    int pad_h0, int pad_h1, int pad_w0, int pad_w1,
                                    int stride_h, int stride_w, int kernel_h, int kernel_w,
                                    enum sf_op_type pool_type, const char *layout);

// create a new global pooling node
struct sf_node *sf_create_global_pool_node(struct sf_graph *graph, struct sf_node *x,
                                           enum sf_op_type pool_type, const char *layout);

// create a new batch-norm node
struct sf_node *sf_create_batchnorm_node(struct sf_graph *graph, struct sf_node *data,
                                         struct sf_node *scale, struct sf_node *bias,
                                         struct sf_node *mean, struct sf_node *var,
                                         double epsilon, const char *layout);

// create a new identity node
struct sf_node *sf_create_identity_node(struct sf_graph *graph, struct sf_node *x);

// create a new ReLU node
struct sf_node *sf_create_relu_node(struct sf_graph *graph, struct sf_node *x);

// create a new sigmoid node
struct sf_node *sf_create_sigmoid_node(struct sf_graph *graph, struct sf_node *x);

// create a new softmax node
struct sf_node *sf_create_softmax_node(struct sf_graph *graph, struct sf_node *x, int axis);

// create a new slice node
struct sf_node *sf_create_slice_node(struct sf_graph *graph, struct sf_node *x,
                                     int num_dims, const int *start, const int *shape);

// create a new concatenate node
struct sf_node *sf_create_concat_node(struct sf_graph *graph, int axis,
                                      int num_args, struct sf_node **args);

// create a new flatten node
struct sf_node *sf_create_flatten_node(struct sf_graph *graph, struct sf_node *x, int axis);

// create a new squeeze node
struct sf_node *sf_create_squeeze_node(struct sf_graph *graph, struct sf_node *x,
                                       int num_axes, const int *axes);

// create a new reshape node
struct sf_node *sf_create_reshape_node(struct sf_graph *graph, struct sf_node *x,
                                       int num_dims, const int *new_shape);

// create a new transpose node
struct sf_node *sf_create_transpose_node(struct sf_graph *graph, struct sf_node *x,
                                         int num_dims, const int *axes);

// create a new reduce op node
struct sf_node *sf_create_reduce_node(struct sf_graph *graph, struct sf_node *x, int num_axes,
                                      const int *axes, int keep_dims, enum sf_op_type type);

// create a new cast node
struct sf_node *sf_create_cast_node(struct sf_graph *graph, struct sf_node *x, enum sf_data_type dtype);

// create a new GEMM node
struct sf_node *sf_create_gemm_node(struct sf_graph *graph, struct sf_node *a, struct sf_node *b,
                                    struct sf_node *c, float alpha, float beta, int trans_a, int trans_b);


#ifdef __cplusplus
    }
#endif

