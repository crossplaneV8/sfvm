
#include "mutator.h"


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
    if (node->op_type == OP_FLATTEN || node->op_type == OP_SQUEEZE) {
        struct sf_tensor_desc desc = node->o_desc;
        return sf_create_reshape_node(mut->graph, new_args[0], desc.num_dims, desc.shape);
    }
    if (node->op_type == OP_TRANSPOSE) {
        struct sf_transpose_attrs *attrs = node->attrs;
        struct sf_tensor_desc desc = node->o_desc;
        int axis = -1, cnt = 0;
        for (int i=0; i<attrs->num_dims; i++) {
            if (desc.shape[i] > 1) {
                cnt += attrs->axes[i] < axis;
                axis = attrs->axes[i];
            }
        }
        if (cnt == 0) {
            return sf_create_reshape_node(mut->graph, new_args[0], desc.num_dims, desc.shape);
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// convert (squeeze, flatten, transpose) to reshape
struct sf_mutator sf_convert_to_reshape(void)
{
    return (struct sf_mutator){.visit = _visit};
}

