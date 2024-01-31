
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
        if (node->op_type == OP_CONST) {
            sf_shared_memory_dec(node->const_attrs.shared_data);
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


// merge consecutive nodes
struct sf_node *sf_merge_consecutive_nodes(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }

    // reshape(reshape(x)) ==> reshape(x)
    if (node->op_type == OP_RESHAPE) {
        struct sf_node *arg = new_args[0];
        if (arg->op_type == OP_RESHAPE) {
            return sf_create_reshape_node(mut->graph, arg->args[0],
                                          node->reshape_attrs.num_dims,
                                          node->reshape_attrs.shape);
        }
    }

    // transpose(transpose(x)) ==> transpose(x)
    if (node->op_type == OP_TRANSPOSE) {
        struct sf_node *arg = new_args[0];
        if (arg->op_type == OP_TRANSPOSE) {
            int axes[SF_MAX_DIMS], num = node->transpose_attrs.num_dims;
            for (int i=0; i<num; i++) {
                axes[i] = arg->transpose_attrs.axes[node->transpose_attrs.axes[i]];
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
        if (new_args[1]->op_type == OP_CONST && new_args[1]->const_attrs.data_desc.dtype == SF_FLOAT32
         && new_args[2]->op_type == OP_CONST && new_args[2]->const_attrs.data_desc.dtype == SF_FLOAT32
         && new_args[3]->op_type == OP_CONST && new_args[3]->const_attrs.data_desc.dtype == SF_FLOAT32
         && new_args[4]->op_type == OP_CONST && new_args[4]->const_attrs.data_desc.dtype == SF_FLOAT32) {
            const int num = new_args[1]->const_attrs.data_desc.shape[0];
            const float *scale = new_args[1]->const_attrs.shared_data;
            const float *bias = new_args[2]->const_attrs.shared_data;
            const float *mean = new_args[3]->const_attrs.shared_data;
            const float *var = new_args[4]->const_attrs.shared_data;

            float *new_scale = sf_malloc(mut->graph->alloc, num * sizeof(float));
            float *new_bias = sf_malloc(mut->graph->alloc, num * sizeof(float));
            const float epsilon = node->bn_attrs.epsilon;

            for (int i=0; i<num; i++) {
                new_scale[i] = scale[i] / sqrt(var[i] + epsilon);
                new_bias[i] = bias[i] - new_scale[i] * mean[i];
            }
            struct sf_tensor_desc desc = {SF_FLOAT32, 4, {1, 1, 1, 1}};
            for (int i=0; i<4; i++) {
                if (node->bn_attrs.layout[i] == 'C') {
                    desc.shape[i] = num; break;
                }
            }
            struct sf_node *alpha = sf_create_const_node(mut->graph, desc, new_scale);
            struct sf_node *beta = sf_create_const_node(mut->graph, desc, new_bias);
            struct sf_node *mul = sf_create_mul_node(mut->graph, new_args[0], alpha);
            struct sf_node *add = sf_create_add_node(mut->graph, mul, beta);
            return add;
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


