
#include "mutator.h"


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
    return sf_clone_node(mut->graph, node, new_args);
}


// remove unreachable nodes in the graph
struct sf_mutator sf_remove_unreachable(void)
{
    return (struct sf_mutator){.visit = _visit};
}


