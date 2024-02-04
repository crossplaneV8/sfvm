
#include <stdio.h>
#include "sfvm.h"


// run graph optimizations
void sf_run_optimization(struct sf_graph *graph)
{
    // pass function list
    sf_transform_func func_list[] = {
        sf_replace_with_reshape,
        sf_batchnorm_to_mul_add,
        sf_fuse_conv_mul_add_relu,
        sf_convert_layout_NHWC_OHWI,
        sf_replace_with_reshape,
        sf_swap_transpose,
        sf_merge_consecutive_nodes,
        sf_fold_const,
        sf_identity_transform,
    };
    int num = sizeof(func_list) / sizeof(func_list[0]);
    sf_run_graph_transforms(graph, num, func_list);
}


int main(void)
{
    const char *onnx_path = "./model/resnet18.onnx";
    struct sf_graph *graph = sf_load_graph_from_onnx(onnx_path);

    if (graph != NULL) {
        struct sf_tensor_desc in_desc = {SF_FLOAT32, 4, {4, 3, 224, 224}};
        sf_set_in_desc(graph, "input.1", in_desc);  // set input tensor dtype and shape
        sf_graph_infer_tensor_desc(graph);          // inference dtype and shape of other nodes

        printf("graph before optimization:\n");
        sf_print_graph(stdout, graph);

        sf_run_optimization(graph); // run optimization

        printf("graph after optimization:\n");
        sf_print_graph(stdout, graph);

        sf_discard_graph(graph);    // free memory
    } else {
        printf("error: failed to load onnx model\n");
    }
    getchar();
    return 0;
}

