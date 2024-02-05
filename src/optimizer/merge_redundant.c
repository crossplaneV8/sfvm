
#include "mutator.h"


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
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


// merge consecutive reshape or transpose nodes
struct sf_mutator sf_merge_redundant(void)
{
    return (struct sf_mutator){.visit = _visit};
}


