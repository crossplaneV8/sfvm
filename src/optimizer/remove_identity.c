
#include "mutator.h"


static int _all_equals(struct sf_node *node, float value)
{
    if (node->op_type == OP_CONST && node->o_desc.dtype == SF_FLOAT32) {
        struct sf_const_attrs *attrs = node->attrs;
        const float *data = (const float*)(attrs->data);
        const size_t num = sf_tensor_prod(node->o_desc);

        for (int i=0; i<num; i++) {
            if (data[i] != value) return 0;
        }
        return 1;
    }
    return 0;
}


static int _same_shape(struct sf_node *x, struct sf_node *y)
{
    struct sf_tensor_desc x_desc = x->o_desc, y_desc = y->o_desc;
    if (x_desc.num_dims == y_desc.num_dims) {
        for (int i=0; i<x_desc.num_dims; i++) {
            if (x_desc.shape[i] != y_desc.shape[i]) return 0;
        }
        return 1;
    }
    return 0;
}


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
    if (node->op_type == OP_IDENTITY) {
        return new_args[0];
    }

    // add(x, 0) ==> x
    if (node->op_type == OP_ADD) {
        if (_same_shape(new_args[1], node) && _all_equals(new_args[0], 0)) {
            return new_args[1];
        }
        if (_same_shape(new_args[0], node) && _all_equals(new_args[1], 0)) {
            return new_args[0];
        }
    }

    // mul(x, 1) ==> x
    if (node->op_type == OP_MUL) {
        if (_same_shape(new_args[1], node) && _all_equals(new_args[0], 1)) {
            return new_args[1];
        }
        if (_same_shape(new_args[0], node) && _all_equals(new_args[1], 1)) {
            return new_args[0];
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// remove identity nodes
struct sf_mutator sf_remove_identity(void)
{
    return (struct sf_mutator){.visit = _visit};
}


