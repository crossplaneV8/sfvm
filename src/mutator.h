
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

// remove identity nodes
struct sf_node *sf_remove_identity(struct sf_mutator *mut, struct sf_node *node);

// convert (squeeze, flatten) to reshape
struct sf_node *sf_replace_with_reshape(struct sf_mutator *mut, struct sf_node *node);

// merge consecutive nodes
struct sf_node *sf_merge_consecutive_nodes(struct sf_mutator *mut, struct sf_node *node);

// convert batch-norm to mul and add node
struct sf_node *sf_batchnorm_to_mul_add(struct sf_mutator *mut, struct sf_node *node);

// fuse (mul, add, relu) into conv
struct sf_node *sf_fuse_conv_mul_add_relu(struct sf_mutator *mut, struct sf_node *node);

// convert tensor layout to (NHWC, OHWI)
struct sf_node *sf_convert_layout_NHWC_OHWI(struct sf_mutator *mut, struct sf_node *node);

// calc(transpose(x)) ==> transpose(calc(x))
struct sf_node *sf_swap_transpose(struct sf_mutator *mut, struct sf_node *node);

// convert constant expr to const node
struct sf_node *sf_fold_const(struct sf_mutator *mut, struct sf_node *node);


#ifdef __cplusplus
    }
#endif

