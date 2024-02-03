
#include <math.h>

#include "mutator.h"


// graph mutator
struct sf_mutator
{
    struct sf_graph *graph;
    struct sf_dict *memo_map;
    sf_transform_func func;
};


// mapping an old node to a new node with a mutator
struct sf_node *sf_mutator_map(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_node = sf_read_dict(mut->memo_map, node);
    if (new_node == NULL) {
        new_node = mut->func(mut, node);
        sf_infer_tensor_desc(new_node);
        sf_write_dict(mut->memo_map, node, new_node);
    }
    return new_node;
}


// run mutator
static void sf_mutator_run(struct sf_mutator *mut)
{
    struct sf_list *old_outs = mut->graph->outputs;
    struct sf_list *old_nodes = mut->graph->nodes;
    mut->graph->outputs = sf_create_list();
    mut->graph->nodes = sf_create_list();

    sf_clear_dict(mut->memo_map);

    // generate new outs from old graph
    for (int i=0; i<old_outs->cnt; i++) {
        struct sf_node *node = sf_mutator_map(mut, old_outs->buf[i]);
        sf_list_append(mut->graph->outputs, node);
    }

    // free memory
    for (int i=0; i<old_nodes->cnt; i++) {
        struct sf_node *node = old_nodes->buf[i];
        if (node->attrs != NULL) {
            sf_shared_memory_dec(node->attrs);
        }
        sf_free(node);
    }
    sf_discard_list(old_nodes);
    sf_discard_list(old_outs);
}


// run graph transforms
void sf_run_graph_transforms(struct sf_graph *graph, int num,
                             sf_transform_func func_list[])
{
    struct sf_dict *memo = sf_create_dict();
    struct sf_mutator mut = {.graph = graph, .memo_map = memo};

    for (int i=0; i<num; i++) {
        mut.func = func_list[i];
        sf_mutator_run(&mut);
    }
    sf_discard_dict(memo);
}


// mapping a graph to another equivalent graph (identity transform)
struct sf_node *sf_identity_transform(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// remove identity nodes
struct sf_node *sf_remove_identity(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }

    if (node->op_type == OP_IDENTITY) {
        return new_args[0];
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// convert (squeeze, flatten) to reshape
struct sf_node *sf_replace_with_reshape(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }

    if (node->op_type == OP_FLATTEN || node->op_type == OP_SQUEEZE) {
        struct sf_tensor_desc desc = node->o_desc;
        return sf_create_reshape_node(mut->graph, new_args[0], desc.num_dims, desc.shape);
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// merge consecutive nodes
struct sf_node *sf_merge_consecutive_nodes(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }

    // reshape(reshape(x)) ==> reshape(x)
    if (node->op_type == OP_RESHAPE) {
        struct sf_reshape_attrs *attrs = node->attrs;
        struct sf_node *arg = new_args[0];
        if (arg->op_type == OP_RESHAPE) {
            return sf_create_reshape_node(mut->graph, arg->args[0],
                                          attrs->num_dims, attrs->shape);
        }
    }

    // transpose(transpose(x)) ==> transpose(x)
    if (node->op_type == OP_TRANSPOSE) {
        struct sf_node *arg = new_args[0];
        if (arg->op_type == OP_TRANSPOSE) {
            struct sf_transpose_attrs *attrs0 = arg->attrs;
            struct sf_transpose_attrs *attrs1 = node->attrs;
            int axes[SF_MAX_DIMS], num = attrs1->num_dims;
            for (int i=0; i<num; i++) {
                axes[i] = attrs0->axes[attrs1->axes[i]];
            }
            for (int i=0; i<num; i++) {
                if (axes[i] != i) {
                    return sf_create_transpose_node(mut->graph, arg->args[0], num, axes);
                }
            }
            return arg->args[0];
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// convert batch-norm to mul and add node
struct sf_node *sf_batchnorm_to_mul_add(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }

    if (node->op_type == OP_BATCHNORM) {
        struct sf_bn_attrs *attrs = node->attrs;
        if (new_args[1]->op_type == OP_CONST && new_args[2]->op_type == OP_CONST
         && new_args[3]->op_type == OP_CONST && new_args[4]->op_type == OP_CONST) {
            struct sf_const_attrs *attrs1 = new_args[1]->attrs, *attrs2 = new_args[2]->attrs;
            struct sf_const_attrs *attrs3 = new_args[3]->attrs, *attrs4 = new_args[4]->attrs;
            if (attrs1->data_desc.dtype == SF_FLOAT32 && attrs2->data_desc.dtype == SF_FLOAT32
             && attrs3->data_desc.dtype == SF_FLOAT32 && attrs4->data_desc.dtype == SF_FLOAT32) {
                const int num = attrs1->data_desc.shape[0];
                struct sf_tensor_desc desc = {SF_FLOAT32, 4, {1, 1, 1, 1}};
                desc.shape[find_axis(attrs->layout, 'C')] = num;

                struct sf_node *alpha = sf_create_const_node(mut->graph, desc, NULL);
                struct sf_node *beta = sf_create_const_node(mut->graph, desc, NULL);
                struct sf_const_attrs *attrs_a = alpha->attrs, *attrs_b = beta->attrs;
                float *new_scale = (float*)(attrs_a->data);
                float *new_bias = (float*)(attrs_b->data);
                const float *scale = (const float*)(attrs1->data);
                const float *bias = (const float*)(attrs2->data);
                const float *mean = (const float*)(attrs3->data);
                const float *var = (const float*)(attrs4->data);
                const float epsilon = attrs->epsilon;

                for (int i=0; i<num; i++) {
                    new_scale[i] = scale[i] / sqrt(var[i] + epsilon);
                    new_bias[i] = bias[i] - new_scale[i] * mean[i];
                }
                struct sf_node *mul = sf_create_mul_node(mut->graph, new_args[0], alpha);
                struct sf_node *add = sf_create_add_node(mut->graph, mul, beta);
                return add;
            }
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


static int _is_broadcast_c(const char *layout, int channel, struct sf_tensor_desc desc)
{
    int shape[SF_MAX_DIMS] = {1, 1, 1, 1};
    shape[find_axis(layout, 'C')] = channel;

    for (int i=0; i<desc.num_dims; i++) {
        if (desc.shape[i] != shape[i]) return 0;
    }
    return 1;
}


// fuse (mul, add, relu) into conv
struct sf_node *sf_fuse_conv_mul_add_relu(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }

    if (node->op_type == OP_MUL) {
        struct sf_node *conv = new_args[0], *scale = new_args[1];
        struct sf_conv_attrs *attrs = conv->attrs;
        if (conv->op_type == OP_CONV && attrs->has_relu == 0) {
            struct sf_node *args[] = {conv->args[0], conv->args[1], conv->args[2]};
            const int axis = find_axis(attrs->w_layout, 'O');
            const int outc = args[1]->o_desc.shape[axis];
            if (_is_broadcast_c(attrs->x_layout, outc, scale->o_desc)) {
                int shape[SF_MAX_DIMS] = {1, 1, 1, 1}; shape[axis] = outc;
                scale = sf_create_reshape_node(mut->graph, scale, 4, shape);
                args[1] = sf_create_mul_node(mut->graph, args[1], scale);
                return sf_clone_node(mut->graph, conv, args);
            }
        }
    }
    if (node->op_type == OP_ADD) {
        struct sf_node *conv = new_args[0], *beta = new_args[1];
        struct sf_conv_attrs *attrs = conv->attrs;
        if (conv->op_type == OP_CONV && attrs->has_relu == 0) {
            const int axis = find_axis(attrs->x_layout, 'C');
            const int outc = conv->o_desc.shape[axis];
            if (_is_broadcast_c(attrs->x_layout, outc, beta->o_desc)) {
                beta = sf_create_reshape_node(mut->graph, beta, 1, &outc);
                struct sf_node *bias = (conv->num_args == 3) ? conv->args[2] : NULL;
                if (bias != NULL) {
                    bias = sf_create_add_node(mut->graph, bias, beta);
                } else {
                    bias = beta;
                }
                return sf_create_conv_node(mut->graph, conv->args[0], conv->args[1], bias,
                                           attrs->x_layout, attrs->w_layout, attrs->pad_h0, attrs->pad_h1,
                                           attrs->pad_w0, attrs->pad_w1, attrs->stride_h, attrs->stride_w,
                                           attrs->dilate_h, attrs->dilate_w, attrs->has_relu);
            }
        }
    }
    if (node->op_type == OP_RELU) {
        struct sf_node *conv = new_args[0];
        if (conv->op_type == OP_CONV) {
            struct sf_conv_attrs *attrs = conv->attrs;
            return sf_create_conv_node(mut->graph, conv->args[0], conv->args[1], conv->args[2],
                                       attrs->x_layout, attrs->w_layout, attrs->pad_h0, attrs->pad_h1,
                                       attrs->pad_w0, attrs->pad_w1, attrs->stride_h, attrs->stride_w,
                                       attrs->dilate_h, attrs->dilate_w, 1);
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// convert tensor layout to (NHWC, OHWI)
struct sf_node *sf_convert_layout_NHWC_OHWI(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }

    if (node->op_type == OP_CONV) {
        struct sf_conv_attrs *attrs = node->attrs;
        if (strcmp(attrs->x_layout, "NHWC") != 0) {
            new_args[0] = sf_create_layout_trans_node(mut->graph, new_args[0], attrs->x_layout, "NHWC");
        }
        if (strcmp(attrs->w_layout, "OHWI") != 0) {
            new_args[1] = sf_create_layout_trans_node(mut->graph, new_args[1], attrs->w_layout, "OHWI");
        }
        struct sf_node *bias = (node->num_args == 3) ? new_args[2] : NULL;
        struct sf_node *conv = sf_create_conv_node(mut->graph, new_args[0], new_args[1], bias, "NHWC",
                                                   "OHWI", attrs->pad_h0, attrs->pad_h1, attrs->pad_w0,
                                                   attrs->pad_w1, attrs->stride_h, attrs->stride_w,
                                                   attrs->dilate_h, attrs->dilate_w, attrs->has_relu);
        if (strcmp(attrs->x_layout, "NHWC") != 0) {
            return sf_create_layout_trans_node(mut->graph, conv, "NHWC", attrs->x_layout);
        }
        return conv;
    }
    if (node->op_type == OP_MAX_POOL || node->op_type == OP_AVG_POOL ||
        node->op_type == OP_G_MAX_POOL || node->op_type == OP_G_AVG_POOL) {
        struct sf_pool_attrs *attrs = node->attrs;
        if (strcmp(attrs->layout, "NHWC") != 0) {
            struct sf_node *trans = sf_create_layout_trans_node(mut->graph, new_args[0], attrs->layout, "NHWC");
            struct sf_node *pool = sf_create_pool_node(mut->graph, trans, attrs->pad_h0, attrs->pad_h1,
                                                       attrs->pad_w0, attrs->pad_w1, attrs->stride_h, attrs->stride_w,
                                                       attrs->kernel_h, attrs->kernel_w, node->op_type, "NHWC");
            return sf_create_layout_trans_node(mut->graph, pool, "NHWC", attrs->layout);
        }
    }
    if (node->op_type == OP_BATCHNORM) {
        struct sf_bn_attrs *attrs = node->attrs;
        if (strcmp(attrs->layout, "NHWC") != 0) {
            new_args[0] = sf_create_layout_trans_node(mut->graph, new_args[0], attrs->layout, "NHWC");
            struct sf_node *norm = sf_create_batchnorm_node(mut->graph, new_args[0], new_args[1], new_args[2],
                                                            new_args[3], new_args[4], attrs->epsilon, "NHWC");
            return sf_create_layout_trans_node(mut->graph, norm, "NHWC", attrs->layout);
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// calc(transpose(x)) ==> transpose(calc(x))
struct sf_node *sf_swap_transpose(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }

    if (node->num_args > 0 && new_args[0]->op_type == OP_TRANSPOSE) {
        struct sf_transpose_attrs *attrs = new_args[0]->attrs;
        int sexa[SF_MAX_DIMS] = {0};
        for (int i=0; i<attrs->num_dims; i++) {
            sexa[attrs->axes[i]] = i;
        }
        if (node->op_type == OP_ADD || node->op_type == OP_SUB
         || node->op_type == OP_MUL || node->op_type == OP_DIV
         || node->op_type == OP_RELU || node->op_type == OP_SIGMOID) {
            for (int i=0; i<node->num_args; i++) {
                new_args[i] = sf_create_transpose_node(mut->graph, new_args[i], attrs->num_dims, sexa);
            }
            struct sf_node *calc = sf_clone_node(mut->graph, node, new_args);
            return sf_create_transpose_node(mut->graph, calc, attrs->num_dims, attrs->axes);
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


static struct sf_node *_fold_const_transpose(struct sf_graph *graph,
                                             struct sf_node *trans,
                                             struct sf_node **args)
{
    struct sf_tensor_desc o_desc = trans->o_desc;
    struct sf_tensor_desc x_desc = args[0]->o_desc;
    struct sf_transpose_attrs *trans_attrs = trans->attrs;
    struct sf_const_attrs *const_attrs = args[0]->attrs;
    const int dims = trans_attrs->num_dims;

    struct sf_node *out = sf_create_const_node(graph, o_desc, NULL);
    struct sf_const_attrs *out_attrs = out->attrs;
    float *new_data = (float*)(out_attrs->data);
    const float *old_data = (const float*)(const_attrs->data);

    int steps[SF_MAX_DIMS] = {0}, index[SF_MAX_DIMS] = {0}, cnt = 0;

    for (int n=1, i=dims-1; i>=0; i--) {
        steps[i] = n; n *= x_desc.shape[i];
    }
    while (index[0] < o_desc.shape[0]) {
        int offset = 0;
        for (int i=0; i<dims; i++) {
            offset += steps[trans_attrs->axes[i]] * index[i];
        }
        new_data[cnt++] = old_data[offset];
        index[dims-1]++;
        for (int i=dims-1; i>0; i--) {
            if (index[i] == o_desc.shape[i]) {
                index[i] = 0; index[i-1]++;
            }
        }
    }
    return sf_create_const_node(graph, o_desc, new_data);
}


static struct sf_node *_fold_const_broadcast_op(struct sf_graph *graph,
                                                struct sf_node *node,
                                                struct sf_node **args)
{
    struct sf_tensor_desc x_desc = args[0]->o_desc;
    struct sf_tensor_desc y_desc = args[1]->o_desc;
    struct sf_tensor_desc o_desc = x_desc;
    const int dims = o_desc.num_dims;
    int sx[SF_MAX_DIMS] = {0}, sy[SF_MAX_DIMS] = {0};

    for (int nx=1, ny=1, i=dims-1; i>=0; i--) {
        sx[i] = nx; nx *= x_desc.shape[i];
        sy[i] = ny; ny *= y_desc.shape[i];
    }
    for (int i=0; i<dims; i++) {
        int a = x_desc.shape[i], b = y_desc.shape[i];
        o_desc.shape[i] = (a > b) ? a : b;
        sx[i] *= x_desc.shape[i] == o_desc.shape[i];
        sy[i] *= y_desc.shape[i] == o_desc.shape[i];
    }

    struct sf_node *out = sf_create_const_node(graph, o_desc, NULL);
    float *z_data = (float*)(((struct sf_const_attrs*)(out->attrs))->data);
    float *x_data = (float*)(((struct sf_const_attrs*)(args[0]->attrs))->data);
    float *y_data = (float*)(((struct sf_const_attrs*)(args[1]->attrs))->data);
    int idx[SF_MAX_DIMS] = {0};

    while (idx[0] < o_desc.shape[0]) {
        int ox = 0, oy = 0;
        for (int i=0; i<dims; i++) {
            ox += sx[i] * idx[i];
            oy += sy[i] * idx[i];
        }
        float x = x_data[ox], y = y_data[oy], z;
        switch (node->op_type) {
            case OP_ADD: z = x + y; break;
            case OP_SUB: z = x - y; break;
            case OP_MUL: z = x * y; break;
            case OP_DIV: z = x / y; break;
            default: z = 0;
        }
        *z_data++ = z;
        idx[dims-1]++;

        for (int i=dims-1; i>0; i--) {
            if (idx[i] == o_desc.shape[i]) {
                idx[i] = 0; idx[i-1]++;
            }
        }
    }
    return out;
}


// convert constant expr to const node
struct sf_node *sf_fold_const(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    int const_num = 0;
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
        if (new_args[i]->op_type == OP_CONST) {
            if (new_args[i]->o_desc.dtype == SF_FLOAT32) {
                const_num++;
            }
        }
    }
    if (const_num == node->num_args && node->num_args > 0) {
        if (node->op_type == OP_RESHAPE) {
            struct sf_const_attrs *attrs = new_args[0]->attrs;
            return sf_create_const_node(mut->graph, node->o_desc, attrs->data);
        }
        if (node->op_type == OP_TRANSPOSE) {
            return _fold_const_transpose(mut->graph, node, new_args);
        }
        if (node->op_type == OP_ADD || node->op_type == OP_SUB
         || node->op_type == OP_MUL || node->op_type == OP_DIV) {
            return _fold_const_broadcast_op(mut->graph, node, new_args);
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}



