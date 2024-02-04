
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
            struct sf_input_attrs *attrs = node->attrs;
            if (strcmp(attrs->name, name) == 0) {
                attrs->data_desc = desc; break;
            }
        }
    }
}


static void _infer_input(struct sf_node *node)
{
    struct sf_input_attrs *attrs = node->attrs;
    node->o_desc = attrs->data_desc;
}


static void _infer_const(struct sf_node *node)
{
    struct sf_const_attrs *attrs = node->attrs;
    node->o_desc = attrs->data_desc;
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


static void _infer_conv(struct sf_node *node)
{
    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    struct sf_tensor_desc w_desc = node->args[1]->o_desc;
    assert(x_desc.num_dims == 4);
    assert(w_desc.num_dims == 4);

    struct sf_conv_attrs *attrs = node->attrs;

    int _n = find_axis(attrs->x_layout, 'N');
    int _c = find_axis(attrs->x_layout, 'C');
    int _h = find_axis(attrs->x_layout, 'H');
    int _w = find_axis(attrs->x_layout, 'W');

    int kernel_h = w_desc.shape[find_axis(attrs->w_layout, 'H')];
    int kernel_w = w_desc.shape[find_axis(attrs->w_layout, 'W')];
    int in_h = attrs->pad_h0 + x_desc.shape[_h] + attrs->pad_h1;
    int in_w = attrs->pad_w0 + x_desc.shape[_w] + attrs->pad_w1;
    kernel_h = (kernel_h - 1) * attrs->dilate_h + 1;
    kernel_w = (kernel_w - 1) * attrs->dilate_w + 1;

    node->o_desc.dtype = x_desc.dtype;
    node->o_desc.num_dims = 4;
    node->o_desc.shape[_n] = x_desc.shape[_n];
    node->o_desc.shape[_c] = w_desc.shape[find_axis(attrs->w_layout, 'O')];
    node->o_desc.shape[_h] = (in_h - kernel_h) / attrs->stride_h + 1;
    node->o_desc.shape[_w] = (in_w - kernel_w) / attrs->stride_w + 1;
}


static void _infer_pool(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    struct sf_pool_attrs *attrs = node->attrs;

    int _h = find_axis(attrs->layout, 'H');
    int _w = find_axis(attrs->layout, 'W');

    int in_h = attrs->pad_h0 + desc.shape[_h] + attrs->pad_h1;
    int in_w = attrs->pad_w0 + desc.shape[_w] + attrs->pad_w1;

    desc.shape[_h] = (in_h - attrs->kernel_h) / attrs->stride_h + 1;
    desc.shape[_w] = (in_w - attrs->kernel_w) / attrs->stride_w + 1;
    node->o_desc = desc;
}


static void _infer_global_pool(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    struct sf_pool_attrs *attrs = node->attrs;
    desc.shape[find_axis(attrs->layout, 'H')] = 1;
    desc.shape[find_axis(attrs->layout, 'W')] = 1;
    node->o_desc = desc;
}


static void _infer_activation(struct sf_node *node)
{
    node->o_desc = node->args[0]->o_desc;
}


static void _infer_slice(struct sf_node *node)
{
    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    struct sf_slice_attrs *attrs = node->attrs;
    node->o_desc.dtype = x_desc.dtype;
    node->o_desc.num_dims = x_desc.num_dims;
    for (int i=0; i<x_desc.num_dims; i++) {
        node->o_desc.shape[i] = attrs->shape[i];
    }
}


static void _infer_concat(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    struct sf_axis_attrs *attrs = node->attrs;
    int axis = (attrs->axis + desc.num_dims) % desc.num_dims;

    for (int i=1; i<node->num_args; i++) {
        desc.shape[axis] += node->args[i]->o_desc.shape[axis];
    }
    node->o_desc = desc;
}


static void _infer_squeeze(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    struct sf_squeeze_attrs *attrs = node->attrs;
    for (int i=0; i<attrs->num_axes; i++) {
        int axis = (attrs->axes[i] + desc.num_dims) % desc.num_dims;
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
    struct sf_axis_attrs *attrs = node->attrs;
    int axis = (attrs->axis + desc.num_dims) % desc.num_dims;
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
    struct sf_reshape_attrs *attrs = node->attrs;
    int64_t prod = 1, axis = -1;
    for (int i=0; i<attrs->num_dims; i++) {
        if (attrs->shape[i] > 0) {
            prod *= attrs->shape[i];
        } else {
            axis = i;
        }
    }

    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    node->o_desc.dtype = x_desc.dtype;
    node->o_desc.num_dims = attrs->num_dims;

    for (int i=0; i<attrs->num_dims; i++) {
        node->o_desc.shape[i] = attrs->shape[i];
    }
    if (axis >= 0) {
        node->o_desc.shape[axis] = (int)(sf_tensor_prod(x_desc) / prod);
    }
}


static void _infer_transpose(struct sf_node *node)
{
    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    struct sf_transpose_attrs *attrs = node->attrs;
    node->o_desc.dtype = x_desc.dtype;
    node->o_desc.num_dims = x_desc.num_dims;
    for (int i=0; i<attrs->num_dims; i++) {
        node->o_desc.shape[i] = x_desc.shape[attrs->axes[i]];
    }
}


static void _infer_reduce(struct sf_node *node)
{
    struct sf_tensor_desc desc = node->args[0]->o_desc;
    struct sf_reduce_attrs *attrs = node->attrs;
    for (int i=0; i<attrs->num_axes; i++) {
        int axis = (attrs->axes[i] + desc.num_dims) % desc.num_dims;
        desc.shape[axis] = attrs->keep_dims ? 1 : 0;
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
    struct sf_cast_attrs *attrs = node->attrs;
    node->o_desc = node->args[0]->o_desc;
    node->o_desc.dtype = attrs->dtype;
}


static void _infer_gemm(struct sf_node *node)
{
    struct sf_gemm_attrs *attrs = node->attrs;
    struct sf_tensor_desc a_desc = node->args[0]->o_desc;
    struct sf_tensor_desc b_desc = node->args[1]->o_desc;
    assert(a_desc.num_dims == 2);
    assert(b_desc.num_dims == 2);

    node->o_desc.dtype = a_desc.dtype;
    node->o_desc.num_dims = 2;
    node->o_desc.shape[0] = attrs->trans_a ? a_desc.shape[1] : a_desc.shape[0];
    node->o_desc.shape[1] = attrs->trans_b ? b_desc.shape[0] : b_desc.shape[1];
}


// infer data type and shape of a node
void sf_infer_tensor_desc(struct sf_node *node)
{
    for (int i=0; i<node->num_args; i++) {
        if (node->args[i]->o_desc.dtype == SF_UNKNOWN) {
            sf_infer_tensor_desc(node->args[i]);
        }
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
    assert(node->o_desc.dtype != SF_UNKNOWN);
}


// infer data type and shape of all nodes in the graph
void sf_graph_infer_tensor_desc(struct sf_graph *graph)
{
    for (int i=0; i<graph->nodes->cnt; i++) {
        sf_infer_tensor_desc(graph->nodes->buf[i]);
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
            struct sf_input_attrs *attrs = node->attrs;
            fprintf(f, "name: \"%s\"", attrs->name);
            break;
        }
        case OP_CONST: {
            struct sf_const_attrs *attrs = node->attrs;
            fprintf(f, "shape: ");
            _print_vec_i32(f, attrs->data_desc.num_dims, attrs->data_desc.shape);
            fprintf(f, ", dtype: %s", sf_get_dtype_name(node->o_desc.dtype));
            break;
        }
        case OP_ADD:        break;
        case OP_SUB:        break;
        case OP_MUL:        break;
        case OP_DIV:        break;
        case OP_CONV: {
            struct sf_conv_attrs *attrs = node->attrs;
            fprintf(f, "layout: %s %s, pads: [%d,%d,%d,%d], stride: [%d,%d], dilate: [%d,%d], relu: %d",
                    attrs->x_layout, attrs->w_layout, attrs->pad_h0, attrs->pad_h1, attrs->pad_w0, attrs->pad_w1,
                    attrs->stride_h, attrs->stride_w, attrs->dilate_h, attrs->dilate_w, attrs->has_relu);
            break;
        }
        case OP_AVG_POOL: case OP_MAX_POOL: {
            struct sf_pool_attrs *attrs = node->attrs;
            fprintf(f, "layout: %s, pads: [%d,%d,%d,%d], stride: [%d,%d], kernel: [%d,%d]",
                    attrs->layout, attrs->pad_h0, attrs->pad_h1, attrs->pad_w0, attrs->pad_w1,
                    attrs->stride_h, attrs->stride_w, attrs->kernel_h, attrs->kernel_w);
            break;
        }
        case OP_G_AVG_POOL: case OP_G_MAX_POOL: {
            struct sf_pool_attrs *attrs = node->attrs;
            fprintf(f, "layout: %s", attrs->layout);
            break;
        }
        case OP_BATCHNORM: {
            struct sf_bn_attrs *attrs = node->attrs;
            fprintf(f, "layout: %s, epsilon: %g", attrs->layout, attrs->epsilon);
            break;
        }
        case OP_IDENTITY:   break;
        case OP_RELU:       break;
        case OP_SIGMOID:    break;
        case OP_SOFTMAX: case OP_CONCAT: case OP_FLATTEN: {
            struct sf_axis_attrs *attrs = node->attrs;
            fprintf(f, "axis: %d", attrs->axis);
            break;
        }
        case OP_SLICE: {
            struct sf_slice_attrs *attrs = node->attrs;
            fprintf(f, "start: ");
            _print_vec_i32(f, attrs->num_dims, attrs->start);
            fprintf(f, ", shape: ");
            _print_vec_i32(f, attrs->num_dims, attrs->shape);
            break;
        }
        case OP_SQUEEZE: {
            fprintf(f, "axes: ");
            struct sf_squeeze_attrs *attrs = node->attrs;
            _print_vec_i32(f, attrs->num_axes, attrs->axes);
            break;
        }
        case OP_RESHAPE: {
            fprintf(f, "new_shape: ");
            struct sf_reshape_attrs *attrs = node->attrs;
            _print_vec_i32(f, attrs->num_dims, attrs->shape);
            break;
        }
        case OP_TRANSPOSE: {
            fprintf(f, "axes: ");
            struct sf_transpose_attrs *attrs = node->attrs;
            _print_vec_i32(f, attrs->num_dims, attrs->axes);
            break;
        }
        case OP_REDUCE_SUM: case OP_REDUCE_AVG: case OP_REDUCE_VAR:
        case OP_REDUCE_STD: case OP_REDUCE_MIN: case OP_REDUCE_MAX:{
            fprintf(f, "axes: ");
            struct sf_reduce_attrs *attrs = node->attrs;
            _print_vec_i32(f, attrs->num_axes, attrs->axes);
            fprintf(f, ", keep_dims: %d", attrs->keep_dims);
            break;
        }
        case OP_CAST: {
            struct sf_cast_attrs *attrs = node->attrs;
            fprintf(f, "dtype: %s", sf_get_dtype_name(attrs->dtype));
            break;
        }
        case OP_GEMM: {
            struct sf_gemm_attrs *attrs = node->attrs;
            fprintf(f, "alpha: %f, beta: %f, trans_a: %d, trans_b: %d",
                    attrs->alpha, attrs->beta, attrs->trans_a, attrs->trans_b);
            break;
        }
    }
}


// print node to file
void sf_print_node(FILE *f, struct sf_node *node)
{
    fprintf(f, "%s", sf_get_dtype_name(node->o_desc.dtype));
    _print_vec_i32(f, node->o_desc.num_dims, node->o_desc.shape);
    fprintf(f, " %%%d = %s(", node->index, sf_get_op_name(node));

    for (int i=0; i<node->num_args; i++) {
        fprintf(f, i ? ", %%%d" : "%%%d" , node->args[i]->index);
    }
    if (node->attrs != NULL) {
        fprintf(f, ") {");
        _print_node_attr(f, node);
        fprintf(f, "}\n");
    } else {
        fprintf(f, ")\n");
    }
}


// print graph to file in SSA format
void sf_print_graph(FILE *f, struct sf_graph *graph)
{
    for (int i=0; i<graph->nodes->cnt; i++) {
        sf_print_node(f, graph->nodes->buf[i]);
    }
    for (int i=0; i<graph->outputs->cnt; i++) {
        struct sf_node *out = graph->outputs->buf[i];
        fprintf(f, i ? ", %%%d" : "(%%%d", out->index);
    }
    fprintf(f, ")\n\n");
}


