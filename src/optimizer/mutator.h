
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include <math.h>

#include "graph/graph.h"


struct sf_mutator;

typedef void (*sf_mutator_init)(struct sf_mutator*);
typedef void (*sf_mutator_clean)(struct sf_mutator*);
typedef struct sf_node* (*sf_mutator_visit)(struct sf_mutator*, struct sf_node*, struct sf_node**);


// graph mutator
struct sf_mutator
{
    struct sf_graph *graph;     // target graph
    struct sf_dict *memo;       // memo map {old_node ==> new_node}
    void *backup;               // backup objects

    sf_mutator_init init;       // init backup objects
    sf_mutator_clean clean;     // clean backup objects
    sf_mutator_visit visit;     // create new nodes while visiting old nodes
};



// run graph optimizations
void sf_run_optimization(struct sf_graph *graph);


// remove unreachable nodes in the graph
struct sf_mutator sf_remove_unreachable(void);

// remove identity nodes
struct sf_mutator sf_remove_identity(void);

// convert (squeeze, flatten, transpose) to reshape
struct sf_mutator sf_convert_to_reshape(void);

// convert batch-norm to mul and add
struct sf_mutator sf_batchnorm_to_mul_add(void);

// fuse (mul, add, relu) into conv
struct sf_mutator sf_fuse_conv_mul_add_relu(void);

// relu(add(x, y)) ==> add_relu(x, y)
struct sf_mutator sf_fuse_add_relu(void);

// convert tensor layout to NHWC
struct sf_mutator sf_convert_layout_NHWC(void);

// convert conv weight layout to NK16 or OHWI
struct sf_mutator sf_pack_conv_weight(void);

// calc(transpose(x)) ==> transpose(calc(x))
struct sf_mutator sf_swap_transpose(void);

// merge consecutive reshape or transpose nodes
struct sf_mutator sf_merge_redundant(void);

// convert constant expr to const node
struct sf_mutator sf_fold_constant(void);


#ifdef __cplusplus
    }
#endif

