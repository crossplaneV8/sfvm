
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


// clone an existing node with same type and attributes
struct sf_node *sf_clone_node(struct sf_graph *graph, struct sf_node *node,
                              struct sf_node **new_args)
{
    struct sf_node *new_node = sf_malloc(graph->alloc, sizeof(struct sf_node));
    memcpy(new_node, node, sizeof(struct sf_node));

    for (int i=0; i<node->num_args; i++) {
        new_node->args[i] = new_args[i];
    }
    if (node->op_type == OP_CONST) {
        sf_shared_memory_inc(node->const_attrs.data);
    }
    return new_node;
}


// clone nodes recursively
struct sf_node *sf_identity_transform(struct sf_mutator *mut, struct sf_node *node)
{
    struct sf_node *new_args[SF_MAX_ARGS];
    for (int i=0; i<node->num_args; i++) {
        new_args[i] = sf_mutator_map(mut, node->args[i]);
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// free all nodes in the list
static void _clear_nodes(struct sf_list *nodes)
{
    for (int i=0; i<nodes->cnt; i++) {
        struct sf_node *node = nodes->buf[i];
        if (node->op_type == OP_CONST) {
            sf_shared_memory_dec(node->const_attrs.data);
        }
        sf_free(node);
    }
    nodes->cnt = 0;
}


// visit graph nodes in DFS order
static void _topo_sort(struct sf_list *list, struct sf_node *node)
{
    if (sf_list_find(list, node) < 0) {
        for (int i=0; i<node->num_args; i++) {
            _topo_sort(list, node->args[i]);
        }
        sf_list_append(list, node);
    }
}


// run mutator
static void sf_mutator_run(struct sf_mutator *mut)
{
    struct sf_list *old_outs = mut->graph->outputs;
    struct sf_list *new_outs = sf_create_list();
    sf_clear_dict(mut->memo_map);

    // generate new outs from old graph
    for (int i=0; i<old_outs->cnt; i++) {
        sf_list_append(new_outs, sf_mutator_map(mut, old_outs->buf[i]));
    }

    // replace old outs with new ones
    sf_discard_list(old_outs);
    mut->graph->outputs = new_outs;

    // replace old nodes with new ones
    _clear_nodes(mut->graph->nodes);
    for (int i=0; i<new_outs->cnt; i++) {
        _topo_sort(mut->graph->nodes, new_outs->buf[i]);
    }
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


