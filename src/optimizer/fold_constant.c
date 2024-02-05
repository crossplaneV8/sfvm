
#include "mutator.h"


// transpose(const()) ==> const()
static struct sf_node *_fold_const_transpose(struct sf_graph *graph,
                                             struct sf_node *trans,
                                             struct sf_node **args)
{
    struct sf_tensor_desc o_desc = trans->o_desc;
    struct sf_tensor_desc x_desc = args[0]->o_desc;
    struct sf_transpose_attrs *attrs = trans->attrs;
    const int dims = attrs->num_dims, *axes = attrs->axes;

    struct sf_node *out = sf_create_const_node(graph, o_desc, NULL);
    float *dst = (float*)(((struct sf_const_attrs*)(out->attrs))->data);
    float *src = (float*)(((struct sf_const_attrs*)(args[0]->attrs))->data);

    int steps[SF_MAX_DIMS] = {0}, index[SF_MAX_DIMS] = {0};

    for (int n=1, i=dims-1; i>=0; i--) {
        steps[i] = n; n *= x_desc.shape[i];
    }
    while (index[0] < o_desc.shape[0]) {
        int offset = 0;
        for (int i=0; i<dims; i++) {
            offset += steps[axes[i]] * index[i];
        }
        *dst++ = src[offset];
        index[dims-1]++;
        for (int i=dims-1; i>0; i--) {
            if (index[i] == o_desc.shape[i]) {
                index[i] = 0; index[i-1]++;
            }
        }
    }
    return out;
}


// broadcast_op(const(), const()) ==> const()
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


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
    int num_const = 0;
    for (int i=0; i<node->num_args; i++) {
        if (new_args[i]->op_type == OP_CONST && new_args[i]->o_desc.dtype == SF_FLOAT32) {
            num_const++;
        }
    }
    if (num_const == node->num_args && node->num_args > 0) {
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


// convert constant expr to const node
struct sf_mutator sf_fold_constant(void)
{
    return (struct sf_mutator){.visit = _visit};
}


