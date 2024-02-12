
#include "mutator.h"


// convert weight layout to [O/16, H*W*I, 16]
static struct sf_node *_pack_nk16(struct sf_graph *graph, struct sf_node *weight,
                                  const char *old_layout, const char **new_layout)
{
    struct sf_tensor_desc desc = weight->o_desc;
    int o = desc.shape[find_axis(old_layout, 'O')];
    int h = desc.shape[find_axis(old_layout, 'H')];
    int w = desc.shape[find_axis(old_layout, 'W')];
    int i = desc.shape[find_axis(old_layout, 'I')];

    if (strcmp(old_layout, "OHWI") != 0) {
        weight = sf_create_layout_trans_node(graph, weight, old_layout, "OHWI");
    }
    if (o % 16 == 0) {
        weight = sf_create_reshape_node(graph, weight, 3, (int[]){o/16, 16, h*w*i});
        weight = sf_create_transpose_node(graph, weight, 3, (int[]){0, 2, 1});
        *new_layout = "NK16";
    } else {
        *new_layout = "OHWI";
    }
    return weight;
}


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
    if (node->op_type == OP_CONV) {
        struct sf_conv_attrs *attrs = node->attrs;
        const char *layout = NULL;
        if (strcmp(attrs->w_layout, "NK16") != 0) {
            new_args[1] = _pack_nk16(mut->graph, new_args[1], attrs->w_layout, &layout);
        }
        struct sf_node *bias = (node->num_args == 3) ? new_args[2] : NULL;
        return sf_create_conv_node(mut->graph, new_args[0], new_args[1], bias, attrs->x_layout, layout,
                                   attrs->pad_h0, attrs->pad_h1, attrs->pad_w0, attrs->pad_w1,
                                   attrs->stride_h, attrs->stride_w, attrs->dilate_h, attrs->dilate_w,
                                   attrs->kernel_h, attrs->kernel_w, attrs->kernel_o, attrs->has_relu);
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// convert conv weight layout to NK16
struct sf_mutator sf_pack_conv_weight(void)
{
    return (struct sf_mutator){.visit = _visit};
}

