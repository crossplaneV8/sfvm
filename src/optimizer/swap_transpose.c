
#include "mutator.h"


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
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


// calc(transpose(x)) ==> transpose(calc(x))
struct sf_mutator sf_swap_transpose(void)
{
    return (struct sf_mutator){.visit = _visit};
}


