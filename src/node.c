
#include "node.h"


// get string name of data type
const char *sf_get_dtype_name(enum sf_data_type dtype)
{
    switch (dtype) {
        case SF_UNKNOWN:        return "unknown";
        case SF_FLOAT16:        return "f16";
        case SF_FLOAT32:        return "f32";
        case SF_FLOAT64:        return "f64";
        case SF_INT8:           return "i8";
        case SF_INT16:          return "i16";
        case SF_INT32:          return "i32";
        case SF_INT64:          return "i64";
    }
    return "unknown";
}


// get string name of node op type
const char *sf_get_op_name(struct sf_node *node)
{
    switch (node->op_type) {
        case OP_UNKNOWN:        return "unknown";
        case OP_INPUT:          return "input";
        case OP_CONST:          return "const";
        case OP_ADD:            return "add";
        case OP_SUB:            return "sub";
        case OP_MUL:            return "mul";
        case OP_DIV:            return "div";
        case OP_CONV:           return "conv";
        case OP_AVG_POOL:       return "avgpool";
        case OP_MAX_POOL:       return "maxpool";
        case OP_G_AVG_POOL:     return "global_avgpool";
        case OP_G_MAX_POOL:     return "global_maxpool";
        case OP_BATCHNORM:      return "batchnorm";
        case OP_IDENTITY:       return "identity";
        case OP_RELU:           return "relu";
        case OP_SIGMOID:        return "sigmoid";
        case OP_SOFTMAX:        return "softmax";
        case OP_SLICE:          return "slice";
        case OP_CONCAT:         return "concat";
        case OP_FLATTEN:        return "flatten";
        case OP_SQUEEZE:        return "squeeze";
        case OP_RESHAPE:        return "reshape";
        case OP_TRANSPOSE:      return "transpose";
        case OP_REDUCE_SUM:     return "reduce_sum";
        case OP_REDUCE_AVG:     return "reduce_avg";
        case OP_REDUCE_VAR:     return "reduce_var";
        case OP_REDUCE_STD:     return "reduce_std";
        case OP_REDUCE_MIN:     return "reduce_min";
        case OP_REDUCE_MAX:     return "reduce_max";
        case OP_CAST:           return "cast";
        case OP_GEMM:           return "gemm";
    }
    return "unknown";
}


// calculate number of tensor elements
size_t sf_tensor_prod(struct sf_tensor_desc desc)
{
    size_t size = 1;
    for (int i=0; i<desc.num_dims; i++) {
        size *= (size_t)desc.shape[i];
    }
    return size;
}


// calculate tensor size (in bytes)
size_t sf_tensor_size(struct sf_tensor_desc desc)
{
    size_t size = 0;
    switch (desc.dtype) {
        case SF_UNKNOWN:    size = 0; break;
        case SF_FLOAT16:    size = 2; break;
        case SF_FLOAT32:    size = 4; break;
        case SF_FLOAT64:    size = 8; break;
        case SF_INT8:       size = 1; break;
        case SF_INT16:      size = 2; break;
        case SF_INT32:      size = 4; break;
        case SF_INT64:      size = 8; break;
    }
    return sf_tensor_prod(desc) * size;
}


// create a new node
struct sf_node *sf_create_node(struct sf_graph *graph, enum sf_op_type type, void *attrs)
{
    struct sf_node *node = sf_malloc(graph->alloc, sizeof(struct sf_node));
    memset(node, 0, sizeof(struct sf_node));

    node->op_type = type;
    if (attrs != NULL) {
        sf_shared_memory_inc(attrs);
        node->attrs = attrs;
    }
    sf_list_append(graph->nodes, node);
    return node;
}


// clone an existing node into the graph
struct sf_node *sf_clone_node(struct sf_graph *graph, struct sf_node *node, struct sf_node **new_args)
{
    struct sf_node *new_node = sf_create_node(graph, node->op_type, node->attrs);

    if (new_args != NULL) {
        new_node->num_args = node->num_args;
        for (int i=0; i<node->num_args; i++) {
            new_node->args[i] = new_args[i];
        }
    }
    return new_node;
}


// create a new input node
struct sf_node *sf_create_input_node(struct sf_graph *graph, const char *name, struct sf_tensor_desc desc)
{
    assert(strlen(name) < SF_MAX_STR_LEN);
    struct sf_input_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_input_attrs));
    strcpy(attrs->name, name);
    attrs->data_desc = desc;
    return sf_create_node(graph, OP_INPUT, attrs);
}


// create a new constant node
struct sf_node *sf_create_const_node(struct sf_graph *graph, struct sf_tensor_desc desc, const void *data)
{
    const size_t data_size = sf_tensor_size(desc);
    struct sf_const_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_const_attrs) + data_size);
    attrs->data_desc = desc;
    if (data != NULL) {
        memcpy(attrs->data, data, data_size);
    }
    return sf_create_node(graph, OP_CONST, attrs);
}


// create a new add node
struct sf_node *sf_create_add_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y)
{
    struct sf_node *node = sf_create_node(graph, OP_ADD, NULL);
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = y;
    return node;
}


// create a new subtract node
struct sf_node *sf_create_sub_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y)
{
    struct sf_node *node = sf_create_node(graph, OP_SUB, NULL);
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = y;
    return node;
}


// create a new multiply node
struct sf_node *sf_create_mul_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y)
{
    struct sf_node *node = sf_create_node(graph, OP_MUL, NULL);
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = y;
    return node;
}


// create a new divide node
struct sf_node *sf_create_div_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y)
{
    struct sf_node *node = sf_create_node(graph, OP_DIV, NULL);
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = y;
    return node;
}


// create a new convolution node
struct sf_node *sf_create_conv_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *w,
                                    struct sf_node *b, const char *x_layout, const char *w_layout,
                                    int pad_h0, int pad_h1, int pad_w0, int pad_w1, int stride_h,
                                    int stride_w, int dilate_h, int dilate_w, int has_relu)
{
    struct sf_conv_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_conv_attrs));
    strcpy(attrs->x_layout, x_layout);
    strcpy(attrs->w_layout, w_layout);
    attrs->pad_h0 = pad_h0;
    attrs->pad_h1 = pad_h1;
    attrs->pad_w0 = pad_w0;
    attrs->pad_w1 = pad_w1;
    attrs->stride_h = stride_h;
    attrs->stride_w = stride_w;
    attrs->dilate_h = dilate_h;
    attrs->dilate_w = dilate_w;
    attrs->has_relu = has_relu;

    struct sf_node *node = sf_create_node(graph, OP_CONV, attrs);
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = w;
    if (b != NULL) {
        node->args[node->num_args++] = b;
    }
    return node;
}


// create a new pooling node
struct sf_node *sf_create_pool_node(struct sf_graph *graph, struct sf_node *x,
                                    int pad_h0, int pad_h1, int pad_w0, int pad_w1,
                                    int stride_h, int stride_w, int kernel_h, int kernel_w,
                                    enum sf_op_type pool_type, const char *layout)
{
    struct sf_pool_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_pool_attrs));
    attrs->pad_h0 = pad_h0;
    attrs->pad_h1 = pad_h1;
    attrs->pad_w0 = pad_w0;
    attrs->pad_w1 = pad_w1;
    attrs->stride_h = stride_h;
    attrs->stride_w = stride_w;
    attrs->kernel_h = kernel_h;
    attrs->kernel_w = kernel_w;
    strcpy(attrs->layout, layout);

    struct sf_node *node = sf_create_node(graph, pool_type, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new global pooling node
struct sf_node *sf_create_global_pool_node(struct sf_graph *graph, struct sf_node *x,
                                           enum sf_op_type pool_type, const char *layout)
{
    struct sf_pool_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_pool_attrs));
    memset(attrs, 0, sizeof(struct sf_pool_attrs));
    strcpy(attrs->layout, layout);

    struct sf_node *node = sf_create_node(graph, pool_type, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new batch-norm node
struct sf_node *sf_create_batchnorm_node(struct sf_graph *graph, struct sf_node *data,
                                         struct sf_node *scale, struct sf_node *bias,
                                         struct sf_node *mean, struct sf_node *var,
                                         double epsilon, const char *layout)
{
    struct sf_bn_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_bn_attrs));
    attrs->epsilon = epsilon;
    strcpy(attrs->layout, layout);

    struct sf_node *node = sf_create_node(graph, OP_BATCHNORM, attrs);
    node->args[node->num_args++] = data;
    node->args[node->num_args++] = scale;
    node->args[node->num_args++] = bias;
    node->args[node->num_args++] = mean;
    node->args[node->num_args++] = var;
    return node;
}


// create a new identity node
struct sf_node *sf_create_identity_node(struct sf_graph *graph, struct sf_node *x)
{
    struct sf_node *node = sf_create_node(graph, OP_IDENTITY, NULL);
    node->args[node->num_args++] = x;
    return node;
}


// create a new ReLU node
struct sf_node *sf_create_relu_node(struct sf_graph *graph, struct sf_node *x)
{
    struct sf_node *node = sf_create_node(graph, OP_RELU, NULL);
    node->args[node->num_args++] = x;
    return node;
}


// create a new sigmoid node
struct sf_node *sf_create_sigmoid_node(struct sf_graph *graph, struct sf_node *x)
{
    struct sf_node *node = sf_create_node(graph, OP_SIGMOID, NULL);
    node->args[node->num_args++] = x;
    return node;
}


// create a new softmax node
struct sf_node *sf_create_softmax_node(struct sf_graph *graph, struct sf_node *x, int axis)
{
    struct sf_axis_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_axis_attrs));
    attrs->axis = axis;

    struct sf_node *node = sf_create_node(graph, OP_SOFTMAX, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new slice node
struct sf_node *sf_create_slice_node(struct sf_graph *graph, struct sf_node *x,
                                     int num_dims, const int *start, const int *shape)
{
    struct sf_slice_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_slice_attrs));
    attrs->num_dims = num_dims;
    for (int i=0; i<num_dims; i++) {
        attrs->start[i] = start[i];
        attrs->shape[i] = shape[i];
    }

    struct sf_node *node = sf_create_node(graph, OP_SLICE, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new concatenate node
struct sf_node *sf_create_concat_node(struct sf_graph *graph, int axis,
                                      int num_args, struct sf_node **args)
{
    struct sf_axis_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_axis_attrs));
    attrs->axis = axis;

    struct sf_node *node = sf_create_node(graph, OP_CONCAT, attrs);
    assert(num_args <= SF_MAX_ARGS);
    node->num_args = num_args;
    for (int i=0; i<num_args; i++) {
        node->args[i] = args[i];
    }
    return node;
}


// create a new flatten node
struct sf_node *sf_create_flatten_node(struct sf_graph *graph, struct sf_node *x, int axis)
{
    struct sf_axis_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_axis_attrs));
    attrs->axis = axis;

    struct sf_node *node = sf_create_node(graph, OP_FLATTEN, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new squeeze node
struct sf_node *sf_create_squeeze_node(struct sf_graph *graph, struct sf_node *x,
                                       int num_axes, const int *axes)
{
    struct sf_squeeze_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_squeeze_attrs));
    attrs->num_axes = num_axes;
    for (int i=0; i<num_axes; i++) {
        attrs->axes[i] = axes[i];
    }

    struct sf_node *node = sf_create_node(graph, OP_SQUEEZE, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new reshape node
struct sf_node *sf_create_reshape_node(struct sf_graph *graph, struct sf_node *x,
                                       int num_dims, const int *new_shape)
{
    struct sf_reshape_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_reshape_attrs));
    attrs->num_dims = num_dims;
    for (int i=0; i<num_dims; i++) {
        attrs->shape[i] = new_shape[i];
    }

    struct sf_node *node = sf_create_node(graph, OP_RESHAPE, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new transpose node
struct sf_node *sf_create_transpose_node(struct sf_graph *graph, struct sf_node *x,
                                         int num_dims, const int *axes)
{
    struct sf_transpose_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_transpose_attrs));
    attrs->num_dims = num_dims;
    for (int i=0; i<num_dims; i++) {
        attrs->axes[i] = axes[i];
    }

    struct sf_node *node = sf_create_node(graph, OP_TRANSPOSE, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new layout transpose node
struct sf_node *sf_create_layout_trans_node(struct sf_graph *graph, struct sf_node *x,
                                            const char *src_layout, const char *dst_layout)
{
    const int num = strlen(src_layout);
    int axes[SF_MAX_DIMS] = {0};

    for (int i=0; i<num; i++) {
        for (int j=0; j<num; j++) {
            if (src_layout[j] == dst_layout[i]) {
                axes[i] = j; break;
            }
        }
    }
    return sf_create_transpose_node(graph, x, num, axes);
}


// create a new reduce op node
struct sf_node *sf_create_reduce_node(struct sf_graph *graph, struct sf_node *x, int num_axes,
                                      const int *axes, int keep_dims, enum sf_op_type type)
{
    struct sf_reduce_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_reduce_attrs));
    attrs->num_axes = num_axes;
    for (int i=0; i<num_axes; i++) {
        attrs->axes[i] = axes[i];
    }
    attrs->keep_dims = keep_dims;

    struct sf_node *node = sf_create_node(graph, type, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new cast node
struct sf_node *sf_create_cast_node(struct sf_graph *graph, struct sf_node *x, enum sf_data_type dtype)
{
    struct sf_cast_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_cast_attrs));
    attrs->dtype = dtype;

    struct sf_node *node = sf_create_node(graph, OP_CAST, attrs);
    node->args[node->num_args++] = x;
    return node;
}


// create a new GEMM node
struct sf_node *sf_create_gemm_node(struct sf_graph *graph, struct sf_node *a, struct sf_node *b,
                                    struct sf_node *c, float alpha, float beta, int trans_a, int trans_b)
{
    struct sf_gemm_attrs *attrs = sf_malloc(graph->alloc, sizeof(struct sf_gemm_attrs));
    attrs->alpha = alpha;
    attrs->beta = beta;
    attrs->trans_a = trans_a;
    attrs->trans_b = trans_b;

    struct sf_node *node = sf_create_node(graph, OP_GEMM, attrs);
    node->args[node->num_args++] = a;
    node->args[node->num_args++] = b;
    if (c != NULL) {
        node->args[node->num_args++] = c;
    }
    return node;
}



