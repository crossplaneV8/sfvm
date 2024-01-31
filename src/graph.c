
#include "graph.h"


// create an empty graph
struct sf_graph *sf_create_graph(void)
{
    struct sf_graph *graph = malloc(sizeof(struct sf_graph));
    graph->alloc = sf_create_allocator();
    graph->outputs = sf_create_list();
    graph->nodes = sf_create_list();
    return graph;
}


// discard an existing graph
void sf_discard_graph(struct sf_graph *graph)
{
    if (graph != NULL) {
        sf_discard_allocator(graph->alloc);
        sf_discard_list(graph->outputs);
        sf_discard_list(graph->nodes);
        free(graph);
    }
}


// set data type and shape of input node
void sf_set_in_desc(struct sf_graph *graph, const char *name, struct sf_tensor_desc desc)
{
    for (int i=0; i<graph->nodes->cnt; i++) {
        struct sf_node *node = graph->nodes->buf[i];
        if (node->op_type == OP_INPUT) {
            if (strcmp(node->input_attrs.name, name) == 0) {
                node->input_attrs.data_desc = desc; break;
            }
        }
    }
}


static void _infer_input(struct sf_node *node)
{
    node->o_desc = node->input_attrs.data_desc;
}


static void _infer_const(struct sf_node *node)
{
    node->o_desc = node->const_attrs.data_desc;
}


static void _infer_broadcast_op(struct sf_node *node)
{
    assert(node->num_args == 2);
    struct sf_tensor_desc a_desc = node->args[0]->o_desc;
    struct sf_tensor_desc b_desc = node->args[1]->o_desc;
    assert (a_desc.num_dims == b_desc.num_dims);

    node->o_desc.dtype = a_desc.dtype;
    node->o_desc.num_dims = a_desc.num_dims;

    for (int i=0; i<a_desc.num_dims; i++) {
        int a = a_desc.shape[i], b = b_desc.shape[i];
        node->o_desc.shape[i] = (a > b) ? a : b;
    }
}


static int _axis_idx(const char *layout, char axis)
{
    for (int i=0; i<SF_MAX_DIMS; i++) {
        if (layout[i] == axis) return i;
    }
    return -1;
}


static void _infer_conv(struct sf_node *node)
{
    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    struct sf_tensor_desc w_desc = node->args[1]->o_desc;
    assert(x_desc.num_dims == 4);
    assert(w_desc.num_dims == 4);

    const char *x_layout = node->conv_attrs.x_layout;
    const char *w_layout = node->conv_attrs.w_layout;
    int pad_h0 = node->conv_attrs.pad_h0;
    int pad_h1 = node->conv_attrs.pad_h1;
    int pad_w0 = node->conv_attrs.pad_w0;
    int pad_w1 = node->conv_attrs.pad_w1;
    int stride_h = node->conv_attrs.stride_h;
    int stride_w = node->conv_attrs.stride_w;
    int dilate_h = node->conv_attrs.dilate_h;
    int dilate_w = node->conv_attrs.dilate_w;

    int _n = _axis_idx(x_layout, 'N');
    int _c = _axis_idx(x_layout, 'C');
    int _h = _axis_idx(x_layout, 'H');
    int _w = _axis_idx(x_layout, 'W');

    int kernel_h = w_desc.shape[_axis_idx(w_layout, 'H')];
    int kernel_w = w_desc.shape[_axis_idx(w_layout, 'W')];
    int in_h = pad_h0 + x_desc.shape[_h] + pad_h1;
    int in_w = pad_w0 + x_desc.shape[_w] + pad_w1;
    kernel_h = (kernel_h - 1) * dilate_h + 1;
    kernel_w = (kernel_w - 1) * dilate_w + 1;

    node->o_desc.dtype = x_desc.dtype;
    node->o_desc.num_dims = 4;
    node->o_desc.shape[_n] = x_desc.shape[_n];
    node->o_desc.shape[_c] = w_desc.shape[_axis_idx(w_layout, 'O')];
    node->o_desc.shape[_h] = (in_h - kernel_h) / stride_h + 1;
    node->o_desc.shape[_w] = (in_w - kernel_w) / stride_w + 1;
}


static void _infer_pool(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    const char *layout = node->pool_attrs.layout;
    int pad_h0 = node->pool_attrs.pad_h0;
    int pad_h1 = node->pool_attrs.pad_h1;
    int pad_w0 = node->pool_attrs.pad_w0;
    int pad_w1 = node->pool_attrs.pad_w1;
    int stride_h = node->pool_attrs.stride_h;
    int stride_w = node->pool_attrs.stride_w;
    int kernel_h = node->pool_attrs.kernel_h;
    int kernel_w = node->pool_attrs.kernel_w;

    int _h = _axis_idx(layout, 'H');
    int _w = _axis_idx(layout, 'W');

    int in_h = pad_h0 + desc.shape[_h] + pad_h1;
    int in_w = pad_w0 + desc.shape[_w] + pad_w1;

    desc.shape[_h] = (in_h - kernel_h) / stride_h + 1;
    desc.shape[_w] = (in_w - kernel_w) / stride_w + 1;
    node->o_desc = desc;
}


static void _infer_global_pool(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    const char *layout = node->pool_attrs.layout;
    desc.shape[_axis_idx(layout, 'H')] = 1;
    desc.shape[_axis_idx(layout, 'W')] = 1;
    node->o_desc = desc;
}


static void _infer_activation(struct sf_node *node)
{
    node->o_desc = node->args[0]->o_desc;
}


static void _infer_slice(struct sf_node *node)
{
    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    node->o_desc.dtype = x_desc.dtype;
    node->o_desc.num_dims = x_desc.num_dims;
    for (int i=0; i<x_desc.num_dims; i++) {
        node->o_desc.shape[i] = node->slice_attrs.shape[i];
    }
}


static void _infer_concat(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    int axis = (node->axis_attrs.axis + desc.num_dims) % desc.num_dims;

    for (int i=1; i<node->num_args; i++) {
        desc.shape[axis] += node->args[i]->o_desc.shape[axis];
    }
    node->o_desc = desc;
}


static void _infer_squeeze(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    for (int i=0; i<node->squeeze_attrs.num_axes; i++) {
        int axis = (node->squeeze_attrs.axes[i] + desc.num_dims) % desc.num_dims;
        desc.shape[axis] = 0;
    }
    int cnt = 0;
    for (int i=0; i<desc.num_dims; i++) {
        if (desc.shape[i] > 0) {
            desc.shape[cnt++] = desc.shape[i];
        }
    }
    desc.num_dims = cnt;
    node->o_desc = desc;
}


static void _infer_flatten(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    int axis = (node->axis_attrs.axis + desc.num_dims) % desc.num_dims;
    int n = 1, c = 1;
    for (int i=0; i<axis; i++) {
        n *= desc.shape[i];
    }
    for (int i=axis; i<desc.num_dims; i++) {
        c *= desc.shape[i];
    }
    node->o_desc.dtype = desc.dtype;
    node->o_desc.num_dims = 2;
    node->o_desc.shape[0] = n;
    node->o_desc.shape[1] = c;
}


static void _infer_reshape(struct sf_node *node)
{
    int64_t prod = 1, axis = -1;
    for (int i=0; i<node->reshape_attrs.num_dims; i++) {
        if (node->reshape_attrs.shape[i] > 0) {
            prod *= node->reshape_attrs.shape[i];
        } else {
            axis = i;
        }
    }

    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    node->o_desc.dtype = x_desc.dtype;
    node->o_desc.num_dims = node->reshape_attrs.num_dims;

    for (int i=0; i<node->reshape_attrs.num_dims; i++) {
        node->o_desc.shape[i] = node->reshape_attrs.shape[i];
    }
    if (axis >= 0) {
        int64_t size = 1;
        for (int i=0; i<x_desc.num_dims; i++) {
            size *= x_desc.shape[i];
        }
        node->o_desc.shape[axis] = (int)(size / prod);
    }
}


static void _infer_transpose(struct sf_node *node)
{
    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    node->o_desc.dtype = x_desc.dtype;
    node->o_desc.num_dims = x_desc.num_dims;
    for (int i=0; i<node->transpose_attrs.num_dims; i++) {
        node->o_desc.shape[i] = x_desc.shape[node->transpose_attrs.axes[i]];
    }
}


static void _infer_reduce(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    for (int i=0; i<node->reduce_attrs.num_axes; i++) {
        int axis = (node->reduce_attrs.axes[i] + desc.num_dims) % desc.num_dims;
        desc.shape[axis] = node->reduce_attrs.keep_dims ? 1 : 0;
    }
    node->o_desc.dtype = desc.dtype;
    node->o_desc.num_dims = 0;

    for (int i=0; i<desc.num_dims; i++) {
        if (desc.shape[i] > 0) {
            node->o_desc.shape[node->o_desc.num_dims++] = desc.shape[i];
        }
    }
}


static void _infer_cast(struct sf_node *node)
{
    node->o_desc = node->args[0]->o_desc;
    node->o_desc.dtype = node->cast_attrs.dtype;
}


static void _infer_gemm(struct sf_node *node)
{
    struct sf_tensor_desc a_desc = node->args[0]->o_desc;
    struct sf_tensor_desc b_desc = node->args[1]->o_desc;
    assert(a_desc.num_dims == 2);
    assert(b_desc.num_dims == 2);

    int trans_a = node->gemm_attrs.trans_a;
    int trans_b = node->gemm_attrs.trans_b;

    node->o_desc.dtype = a_desc.dtype;
    node->o_desc.num_dims = 2;
    node->o_desc.shape[0] = trans_a ? a_desc.shape[1] : a_desc.shape[0];
    node->o_desc.shape[1] = trans_b ? b_desc.shape[0] : b_desc.shape[1];
}


// infer data type and shape of a node
static void _infer_tensor_desc(struct sf_node *node)
{
    for (int i=0; i<node->num_args; i++) {
        if (node->args[i]->o_desc.dtype == SF_UNKNOWN) return;
    }
    switch (node->op_type) {
        case OP_UNKNOWN:    break;
        case OP_INPUT:      _infer_input(node); break;
        case OP_CONST:      _infer_const(node); break;
        case OP_ADD:        _infer_broadcast_op(node); break;
        case OP_SUB:        _infer_broadcast_op(node); break;
        case OP_MUL:        _infer_broadcast_op(node); break;
        case OP_DIV:        _infer_broadcast_op(node); break;
        case OP_CONV:       _infer_conv(node); break;
        case OP_AVG_POOL:   _infer_pool(node); break;
        case OP_MAX_POOL:   _infer_pool(node); break;
        case OP_G_AVG_POOL: _infer_global_pool(node); break;
        case OP_G_MAX_POOL: _infer_global_pool(node); break;
        case OP_BATCHNORM:  _infer_activation(node); break;
        case OP_IDENTITY:   _infer_activation(node); break;
        case OP_RELU:       _infer_activation(node); break;
        case OP_SIGMOID:    _infer_activation(node); break;
        case OP_SOFTMAX:    _infer_activation(node); break;
        case OP_SLICE:      _infer_slice(node); break;
        case OP_CONCAT:     _infer_concat(node); break;
        case OP_FLATTEN:    _infer_flatten(node); break;
        case OP_SQUEEZE:    _infer_squeeze(node); break;
        case OP_RESHAPE:    _infer_reshape(node); break;
        case OP_TRANSPOSE:  _infer_transpose(node); break;
        case OP_REDUCE_SUM: _infer_reduce(node); break;
        case OP_REDUCE_AVG: _infer_reduce(node); break;
        case OP_REDUCE_VAR: _infer_reduce(node); break;
        case OP_REDUCE_STD: _infer_reduce(node); break;
        case OP_REDUCE_MIN: _infer_reduce(node); break;
        case OP_REDUCE_MAX: _infer_reduce(node); break;
        case OP_CAST:       _infer_cast(node); break;
        case OP_GEMM:       _infer_gemm(node); break;
    }
}


// infer tensor descriptor recursively with a memo map
static void _infer_tensor_desc_dfs(struct sf_node *node, struct sf_dict *memo)
{
    if (sf_read_dict(memo, node) == NULL) {
        for (int i=0; i<node->num_args; i++) {
            struct sf_node *arg = node->args[i];
            if (arg->o_desc.dtype == SF_UNKNOWN) {
                _infer_tensor_desc_dfs(arg, memo);
            }
        }
        _infer_tensor_desc(node);
        sf_write_dict(memo, node, node);
    }
}


// infer tensor descriptor of a node recursively
void sf_infer_tensor_desc_dfs(struct sf_node *node)
{
    struct sf_dict *memo = sf_create_dict();
    _infer_tensor_desc_dfs(node, memo);
    sf_discard_dict(memo);
}


// infer data type and shape of all nodes in the graph
void sf_graph_infer_tensor_desc(struct sf_graph *graph)
{
    for (int i=0; i<graph->nodes->cnt; i++) {
        _infer_tensor_desc(graph->nodes->buf[i]);
    }
}


static void _print_vec_i32(FILE *f, int num_dims, const int *shape)
{
    fprintf(f, "[");
    for (int i=0; i<num_dims; i++) {
        fprintf(f, (i ? ", %d" : "%d"), shape[i]);
    }
    fprintf(f, "]");
}


// print attributes of a node
static void _print_node_attr(FILE *f, struct sf_node *node)
{
    switch (node->op_type) {
        case OP_UNKNOWN: {
            break;
        }
        case OP_INPUT: {
            fprintf(f, "name: \"%s\"", node->input_attrs.name);
            break;
        }
        case OP_CONST: {
            struct sf_tensor_desc desc = node->const_attrs.data_desc;
            fprintf(f, "shape: ");
            _print_vec_i32(f, desc.num_dims, desc.shape);
            fprintf(f, ", dtype: %s", sf_get_dtype_name(node->o_desc.dtype));
            break;
        }
        case OP_ADD:        break;
        case OP_SUB:        break;
        case OP_MUL:        break;
        case OP_DIV:        break;
        case OP_CONV: {
            fprintf(f, "layout: %s %s, pads: [%d,%d,%d,%d], stride: [%d,%d], dilate: [%d,%d], relu: %d",
                    node->conv_attrs.x_layout, node->conv_attrs.w_layout, node->conv_attrs.pad_h0, node->conv_attrs.pad_h1,
                    node->conv_attrs.pad_w0, node->conv_attrs.pad_w1, node->conv_attrs.stride_h, node->conv_attrs.stride_w,
                    node->conv_attrs.dilate_h, node->conv_attrs.dilate_w, node->conv_attrs.has_relu);
            break;
        }
        case OP_AVG_POOL: case OP_MAX_POOL: {
            fprintf(f, "layout: %s, pads: [%d,%d,%d,%d], stride: [%d,%d], kernel: [%d,%d]",
                    node->pool_attrs.layout, node->pool_attrs.pad_h0, node->pool_attrs.pad_h1,
                    node->pool_attrs.pad_w0, node->pool_attrs.pad_w1, node->pool_attrs.stride_h,
                    node->pool_attrs.stride_w, node->pool_attrs.kernel_h, node->pool_attrs.kernel_w);
            break;
        }
        case OP_G_AVG_POOL: case OP_G_MAX_POOL: {
            fprintf(f, "layout: %s", node->pool_attrs.layout);
            break;
        }
        case OP_BATCHNORM: {
            fprintf(f, "layout: %s, epsilon: %g", node->bn_attrs.layout, node->bn_attrs.epsilon);
            break;
        }
        case OP_IDENTITY:   break;
        case OP_RELU:       break;
        case OP_SIGMOID:    break;
        case OP_SOFTMAX: case OP_CONCAT: case OP_FLATTEN: {
            fprintf(f, "axis: %d", node->axis_attrs.axis);
            break;
        }
        case OP_SLICE: {
            fprintf(f, "start: ");
            _print_vec_i32(f, node->slice_attrs.num_dims, node->slice_attrs.start);
            fprintf(f, ", shape: ");
            _print_vec_i32(f, node->slice_attrs.num_dims, node->slice_attrs.shape);
            break;
        }
        case OP_SQUEEZE: {
            fprintf(f, "axes: ");
            _print_vec_i32(f, node->squeeze_attrs.num_axes, node->squeeze_attrs.axes);
            break;
        }
        case OP_RESHAPE: {
            fprintf(f, "new_shape: ");
            _print_vec_i32(f, node->reshape_attrs.num_dims, node->reshape_attrs.shape);
            break;
        }
        case OP_TRANSPOSE: {
            fprintf(f, "axes: ");
            _print_vec_i32(f, node->transpose_attrs.num_dims, node->transpose_attrs.axes);
            break;
        }
        case OP_REDUCE_SUM: case OP_REDUCE_AVG: case OP_REDUCE_VAR:
        case OP_REDUCE_STD: case OP_REDUCE_MIN: case OP_REDUCE_MAX:{
            fprintf(f, "axes: ");
            _print_vec_i32(f, node->reduce_attrs.num_axes, node->transpose_attrs.axes);
            fprintf(f, ", keep_dims: %d", node->reduce_attrs.keep_dims);
            break;
        }
        case OP_CAST: {
            fprintf(f, "dtype: %s", sf_get_dtype_name(node->cast_attrs.dtype));
            break;
        }
        case OP_GEMM: {
            fprintf(f, "alpha: %f, beta: %f, trans_a: %d, trans_b: %d",
                    node->gemm_attrs.alpha, node->gemm_attrs.beta,
                    node->gemm_attrs.trans_a, node->gemm_attrs.trans_b);
            break;
        }
    }
}


// print graph to file in SSA format
void sf_print_graph(FILE *f, struct sf_graph *graph, int with_desc, int with_attrs)
{
    for (int i=0; i<graph->nodes->cnt; i++) {
        struct sf_node *node = graph->nodes->buf[i];
        if (with_desc) {
            fprintf(f, "%s", sf_get_dtype_name(node->o_desc.dtype));
            _print_vec_i32(f, node->o_desc.num_dims, node->o_desc.shape);
            fprintf(f, " ");
        }
        fprintf(f, "%%%d = %s(", i, sf_get_op_name(node));
        for (int j=0; j<node->num_args; j++) {
            int k = sf_list_find(graph->nodes, node->args[j]);
            fprintf(f, j ? ", %%%d" : "%%%d" , k);
        }
        if (with_attrs) {
            fprintf(f, ") {");
            _print_node_attr(f, node);
            fprintf(f, "}\n");
        } else {
            fprintf(f, ")\n");
        }
    }
    for (int i=0; i<graph->outputs->cnt; i++) {
        int k = sf_list_find(graph->nodes, graph->outputs->buf[i]);
        fprintf(f, i ? ", %%%d" : "(%%%d", k);
    }
    fprintf(f, ")\n");
}



