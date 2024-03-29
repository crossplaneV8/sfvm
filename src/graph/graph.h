
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


// set node as graph output
void sf_set_graph_output(struct sf_graph *graph, struct sf_node *node);


// infer data type and shape of a node
void sf_infer_tensor_desc(struct sf_node *node);


// infer data type and shape of all nodes in the graph
void sf_graph_infer_tensor_desc(struct sf_graph *graph);


// print node to file
void sf_print_node(FILE *f, struct sf_node *node);


// print graph to file in SSA format
void sf_print_graph(FILE *f, struct sf_graph *graph);



#ifdef __cplusplus
    }
#endif

