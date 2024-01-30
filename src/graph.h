
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include <stdio.h>

#include "node.h"


// create an empty graph
struct sf_graph *sf_create_graph(void);


// discard an existing graph
void sf_discard_graph(struct sf_graph *graph);


// convert onnx file to sfvm graph
struct sf_graph *sf_load_graph_from_onnx(const char *path);


// set data type and shape of input tensor
void sf_set_in_desc(struct sf_graph *graph, const char *name, struct sf_tensor_desc desc);


// infer tensor descriptor of a node recursively
void sf_infer_tensor_desc_dfs(struct sf_node *node);


// infer data type and shape of all nodes in the graph
void sf_graph_infer_tensor_desc(struct sf_graph *graph);


// print graph to file in SSA format
void sf_print_graph(FILE *f, struct sf_graph *graph, int with_desc, int with_attrs);



#ifdef __cplusplus
    }
#endif

