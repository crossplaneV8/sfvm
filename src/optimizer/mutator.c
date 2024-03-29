
#include "mutator.h"


// recursively mapping old nodes to new nodes with a mutator
static struct sf_node *sf_mutator_map(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_node = sf_read_dict(mut->memo, node);
    if (new_node == NULL) {
        struct sf_node *new_args[node->num_args];
        for (int i=0; i<node->num_args; i++) {
            new_args[i] = sf_mutator_map(mut, node->args[i]);
        }
        new_node = mut->visit(mut, node, new_args);
        sf_infer_tensor_desc(new_node);
        sf_write_dict(mut->memo, node, new_node);
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

    // generate new outs from old graph
    for (int i=0; i<old_outs->cnt; i++) {
        struct sf_node *node = sf_mutator_map(mut, old_outs->buf[i]);
        sf_set_graph_output(mut->graph, node);
    }

    // free memory
    for (int i=0; i<old_nodes->cnt; i++) {
        struct sf_node *node = old_nodes->buf[i];
        if (node->attrs != NULL) {
            sf_shared_memory_detach(node->attrs);
        }
        sf_free(node);
    }
    sf_discard_list(old_nodes);
    sf_discard_list(old_outs);
}


// run mutators on graph
static void sf_run_mutators(struct sf_graph *graph, int num, struct sf_mutator muts[])
{
    struct sf_dict *memo = sf_create_dict();

    for (int i=0; i<num; i++) {
        struct sf_mutator *mut = &muts[i];
        mut->graph = graph;
        mut->memo = memo;

        if (mut->init != NULL) {
            mut->init(mut);     // init backup objects
        }
        sf_mutator_run(mut);    // run mutator

        if (mut->clean != NULL) {
            mut->clean(mut);    // clean backup objects
        }
        sf_clear_dict(memo);
    }
    sf_discard_dict(memo);
}


// run graph optimizations
void sf_run_optimization(struct sf_graph *graph)
{
    // graph mutator list
    struct sf_mutator mutators[] = {
        sf_convert_to_reshape(),
        sf_batchnorm_to_mul_add(),
        sf_remove_identity(),
        sf_fuse_conv_mul_add_relu(),
        sf_convert_layout_NHWC(),
        sf_pack_conv_weight(),
        sf_convert_to_reshape(),
        sf_swap_transpose(),
        sf_merge_redundant(),
        sf_fuse_add_relu(),
        sf_fold_constant(),
        sf_remove_unreachable(),
    };
    int num = sizeof(mutators) / sizeof(mutators[0]);
    sf_graph_infer_tensor_desc(graph);
    sf_run_mutators(graph, num, mutators);
}


