
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
static struct sf_node *_sf_create_node(struct sf_graph *graph)
{
    struct sf_node *node = sf_malloc(graph->alloc, sizeof(struct sf_node));
    memset(node, 0, sizeof(struct sf_node));
    sf_list_append(graph->nodes, node);
    return node;
}


// clone an existing node into the graph
struct sf_node *sf_clone_node(struct sf_graph *graph, struct sf_node *node,
                              struct sf_node **new_args)
{
    struct sf_node *new_node = _sf_create_node(graph);
    memcpy(new_node, node, sizeof(struct sf_node));
    new_node->o_desc = (struct sf_tensor_desc){SF_UNKNOWN};

    if (new_args != NULL) {
        for (int i=0; i<node->num_args; i++) {
            new_node->args[i] = new_args[i];
        }
        new_node->num_args = node->num_args;
    } else {
        new_node->num_args = 0;
    }
    if (node->op_type == OP_CONST) {
        sf_shared_memory_inc(node->const_attrs.shared_data);
    }
    return new_node;
}


// create a new input node
struct sf_node *sf_create_input_node(struct sf_graph *graph, const char *name, struct sf_tensor_desc desc)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_INPUT;
    strcpy(node->input_attrs.name, name);
    node->input_attrs.data_desc = desc;
    return node;
}


// create a new constant node
struct sf_node *sf_create_const_node(struct sf_graph *graph, struct sf_tensor_desc desc, void *shared_data)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_CONST;
    node->const_attrs.shared_data = shared_data;
    node->const_attrs.data_desc = desc;
    sf_shared_memory_inc(shared_data);
    return node;
}


// create a new add node
struct sf_node *sf_create_add_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_ADD;
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = y;
    return node;
}


// create a new subtract node
struct sf_node *sf_create_sub_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_SUB;
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = y;
    return node;
}


// create a new multiply node
struct sf_node *sf_create_mul_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_MUL;
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = y;
    return node;
}


// create a new divide node
struct sf_node *sf_create_div_node(struct sf_graph *graph, struct sf_node *x, struct sf_node *y)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_DIV;
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
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_CONV;
    node->args[node->num_args++] = x;
    node->args[node->num_args++] = w;
    if (b != NULL) {
        node->args[node->num_args++] = b;
    }
    strcpy(node->conv_attrs.x_layout, x_layout);
    strcpy(node->conv_attrs.w_layout, w_layout);
    node->conv_attrs.has_relu = has_relu;
    node->conv_attrs.pad_h0 = pad_h0;
    node->conv_attrs.pad_h1 = pad_h1;
    node->conv_attrs.pad_w0 = pad_w0;
    node->conv_attrs.pad_w1 = pad_w1;
    node->conv_attrs.stride_h = stride_h;
    node->conv_attrs.stride_w = stride_w;
    node->conv_attrs.dilate_h = dilate_h;
    node->conv_attrs.dilate_w = dilate_w;
    return node;
}


// create a new pooling node
struct sf_node *sf_create_pool_node(struct sf_graph *graph, struct sf_node *x,
                                    int pad_h0, int pad_h1, int pad_w0, int pad_w1,
                                    int stride_h, int stride_w, int kernel_h, int kernel_w,
                                    enum sf_op_type pool_type, const char *layout)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = pool_type;
    node->args[node->num_args++] = x;
    node->pool_attrs.pad_h0 = pad_h0;
    node->pool_attrs.pad_h1 = pad_h1;
    node->pool_attrs.pad_w0 = pad_w0;
    node->pool_attrs.pad_w1 = pad_w1;
    node->pool_attrs.stride_h = stride_h;
    node->pool_attrs.stride_w = stride_w;
    node->pool_attrs.kernel_h = kernel_h;
    node->pool_attrs.kernel_w = kernel_w;
    strcpy(node->pool_attrs.layout, layout);
    return node;
}


// create a new global pooling node
struct sf_node *sf_create_global_pool_node(struct sf_graph *graph, struct sf_node *x,
                                           enum sf_op_type pool_type, const char *layout)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = pool_type;
    node->args[node->num_args++] = x;
    strcpy(node->pool_attrs.layout, layout);
    return node;
}


// create a new batch-norm node
struct sf_node *sf_create_batchnorm_node(struct sf_graph *graph, struct sf_node *data,
                                         struct sf_node *scale, struct sf_node *bias,
                                         struct sf_node *mean, struct sf_node *var,
                                         double epsilon, const char *layout)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_BATCHNORM;
    node->args[node->num_args++] = data;
    node->args[node->num_args++] = scale;
    node->args[node->num_args++] = bias;
    node->args[node->num_args++] = mean;
    node->args[node->num_args++] = var;
    node->bn_attrs.epsilon = epsilon;
    strcpy(node->bn_attrs.layout, layout);
    return node;
}


// create a new identity node
struct sf_node *sf_create_identity_node(struct sf_graph *graph, struct sf_node *x)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_IDENTITY;
    node->args[node->num_args++] = x;
    return node;
}


// create a new ReLU node
struct sf_node *sf_create_relu_node(struct sf_graph *graph, struct sf_node *x)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_RELU;
    node->args[node->num_args++] = x;
    return node;
}


// create a new sigmoid node
struct sf_node *sf_create_sigmoid_node(struct sf_graph *graph, struct sf_node *x)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_SIGMOID;
    node->args[node->num_args++] = x;
    return node;
}


// create a new softmax node
struct sf_node *sf_create_softmax_node(struct sf_graph *graph, struct sf_node *x, int axis)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_SOFTMAX;
    node->args[node->num_args++] = x;
    node->axis_attrs.axis = axis;
    return node;
}


// create a new slice node
struct sf_node *sf_create_slice_node(struct sf_graph *graph, struct sf_node *x,
                                     int num_dims, const int *start, const int *shape)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_SLICE;
    node->args[node->num_args++] = x;
    node->slice_attrs.num_dims = num_dims;
    for (int i=0; i<num_dims; i++) {
        node->slice_attrs.start[i] = start[i];
        node->slice_attrs.shape[i] = shape[i];
    }
    return node;
}


// create a new concatenate node
struct sf_node *sf_create_concat_node(struct sf_graph *graph, int axis,
                                      int num_args, struct sf_node **args)
{
    assert(num_args <= SF_MAX_ARGS);
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_CONCAT;
    node->num_args = num_args;
    for (int i=0; i<num_args; i++) {
        node->args[i] = args[i];
    }
    node->axis_attrs.axis = axis;
    return node;
}


// create a new flatten node
struct sf_node *sf_create_flatten_node(struct sf_graph *graph, struct sf_node *x, int axis)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_FLATTEN;
    node->args[node->num_args++] = x;
    node->axis_attrs.axis = axis;
    return node;
}


// create a new squeeze node
struct sf_node *sf_create_squeeze_node(struct sf_graph *graph, struct sf_node *x,
                                       int num_axes, const int *axes)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_SQUEEZE;
    node->args[node->num_args++] = x;
    node->squeeze_attrs.num_axes = num_axes;
    for (int i=0; i<num_axes; i++) {
        node->squeeze_attrs.axes[i] = axes[i];
    }
    return node;
}


// create a new reshape node
struct sf_node *sf_create_reshape_node(struct sf_graph *graph, struct sf_node *x,
                                       int num_dims, const int *new_shape)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_RESHAPE;
    node->args[node->num_args++] = x;
    node->reshape_attrs.num_dims = num_dims;
    for (int i=0; i<num_dims; i++) {
        node->reshape_attrs.shape[i] = new_shape[i];
    }
    return node;
}


// create a new transpose node
struct sf_node *sf_create_transpose_node(struct sf_graph *graph, struct sf_node *x,
                                         int num_dims, const int *axes)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_TRANSPOSE;
    node->args[node->num_args++] = x;
    node->transpose_attrs.num_dims = num_dims;
    for (int i=0; i<num_dims; i++) {
        node->transpose_attrs.axes[i] = axes[i];
    }
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
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = type;
    node->args[node->num_args++] = x;
    node->reduce_attrs.num_axes = num_axes;
    for (int i=0; i<num_axes; i++) {
        node->reduce_attrs.axes[i] = axes[i];
    }
    node->reduce_attrs.keep_dims = keep_dims;
    return node;
}


// create a new cast node
struct sf_node *sf_create_cast_node(struct sf_graph *graph, struct sf_node *x, enum sf_data_type dtype)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_CAST;
    node->args[node->num_args++] = x;
    node->cast_attrs.dtype = dtype;
    return node;
}


// create a new GEMM node
struct sf_node *sf_create_gemm_node(struct sf_graph *graph, struct sf_node *a, struct sf_node *b,
                                    struct sf_node *c, float alpha, float beta, int trans_a, int trans_b)
{
    struct sf_node *node = _sf_create_node(graph);
    node->op_type = OP_GEMM;
    node->args[node->num_args++] = a;
    node->args[node->num_args++] = b;
    if (c != NULL) {
        node->args[node->num_args++] = c;
    }
    node->gemm_attrs.alpha = alpha;
    node->gemm_attrs.beta = beta;
    node->gemm_attrs.trans_a = trans_a;
    node->gemm_attrs.trans_b = trans_b;
    return node;
}



