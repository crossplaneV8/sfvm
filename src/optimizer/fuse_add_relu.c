
#include "mutator.h"


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
    if (node->op_type == OP_RELU) {
        struct sf_node *add = new_args[0];
        if (add->op_type == OP_ADD) {
            if (_same_shape(add->args[0], add->args[1])) {
                return sf_create_add_relu_node(mut->graph, add->args[0], add->args[1]);
            }
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// relu(add(x, y)) ==> add_relu(x, y)
struct sf_mutator sf_fuse_add_relu(void)
{
    return (struct sf_mutator){.visit = _visit};
}


