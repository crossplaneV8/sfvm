
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include "graph.h"


struct sf_mutator;

typedef struct sf_node* (*sf_transform_func)(struct sf_mutator*, struct sf_node*);


// run graph transforms
void sf_run_graph_transforms(struct sf_graph *graph, int num,
                             sf_transform_func func_list[]);


// clone nodes recursively
struct sf_node *sf_identity_transform(struct sf_mutator *mut, struct sf_node *node);



#ifdef __cplusplus
    }
#endif

