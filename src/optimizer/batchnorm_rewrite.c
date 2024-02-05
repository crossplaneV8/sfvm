
#include "mutator.h"


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
    if (node->op_type == OP_BATCHNORM) {
        struct sf_bn_attrs *attrs = node->attrs;
        if (new_args[1]->op_type == OP_CONST && new_args[1]->o_desc.dtype == SF_FLOAT32
         && new_args[2]->op_type == OP_CONST && new_args[2]->o_desc.dtype == SF_FLOAT32
         && new_args[3]->op_type == OP_CONST && new_args[3]->o_desc.dtype == SF_FLOAT32
         && new_args[4]->op_type == OP_CONST && new_args[4]->o_desc.dtype == SF_FLOAT32) {
            const int num = new_args[1]->o_desc.shape[0];
            struct sf_tensor_desc desc = {SF_FLOAT32, 4, {1, 1, 1, 1}};
            desc.shape[find_axis(attrs->layout, 'C')] = num;

            struct sf_node *alpha = sf_create_const_node(mut->graph, desc, NULL);
            struct sf_node *beta = sf_create_const_node(mut->graph, desc, NULL);

            float *scale = (float*)(((struct sf_const_attrs*)(new_args[1]->attrs))->data);
            float *bias = (float*)(((struct sf_const_attrs*)(new_args[2]->attrs))->data);
            float *mean = (float*)(((struct sf_const_attrs*)(new_args[3]->attrs))->data);
            float *var = (float*)(((struct sf_const_attrs*)(new_args[4]->attrs))->data);
            float *new_scale = (float*)(((struct sf_const_attrs*)(alpha->attrs))->data);
            float *new_bias = (float*)(((struct sf_const_attrs*)(beta->attrs))->data);
            float epsilon = attrs->epsilon;

            for (int i=0; i<num; i++) {
                new_scale[i] = scale[i] / sqrt(var[i] + epsilon);
                new_bias[i] = bias[i] - new_scale[i] * mean[i];
            }
            struct sf_node *mul = sf_create_mul_node(mut->graph, new_args[0], alpha);
            struct sf_node *add = sf_create_add_node(mut->graph, mul, beta);
            return add;
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// convert batch-norm to mul and add
struct sf_mutator sf_batchnorm_to_mul_add(void)
{
    return (struct sf_mutator){.visit = _visit};
}

