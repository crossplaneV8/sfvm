
#include "mutator.h"


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
    if (node->op_type == OP_CONV) {
        struct sf_conv_attrs *attrs = node->attrs;
        if (strcmp(attrs->x_layout, "NHWC") != 0) {
            new_args[0] = sf_create_layout_trans_node(mut->graph, new_args[0], attrs->x_layout, "NHWC");
        }
        if (strcmp(attrs->w_layout, "OHWI") != 0) {
            new_args[1] = sf_create_layout_trans_node(mut->graph, new_args[1], attrs->w_layout, "OHWI");
        }
        struct sf_node *bias = (node->num_args == 3) ? new_args[2] : NULL;
        struct sf_node *conv = sf_create_conv_node(mut->graph, new_args[0], new_args[1], bias, "NHWC",
                                                   "OHWI", attrs->pad_h0, attrs->pad_h1, attrs->pad_w0,
                                                   attrs->pad_w1, attrs->stride_h, attrs->stride_w,
                                                   attrs->dilate_h, attrs->dilate_w, attrs->has_relu);
        if (strcmp(attrs->x_layout, "NHWC") != 0) {
            return sf_create_layout_trans_node(mut->graph, conv, "NHWC", attrs->x_layout);
        }
        return conv;
    }
    if (node->op_type == OP_MAX_POOL || node->op_type == OP_AVG_POOL ||
        node->op_type == OP_G_MAX_POOL || node->op_type == OP_G_AVG_POOL) {
        struct sf_pool_attrs *attrs = node->attrs;
        if (strcmp(attrs->layout, "NHWC") != 0) {
            struct sf_node *trans = sf_create_layout_trans_node(mut->graph, new_args[0], attrs->layout, "NHWC");
            struct sf_node *pool = sf_create_pool_node(mut->graph, trans, attrs->pad_h0, attrs->pad_h1,
                                                       attrs->pad_w0, attrs->pad_w1, attrs->stride_h, attrs->stride_w,
                                                       attrs->kernel_h, attrs->kernel_w, node->op_type, "NHWC");
            return sf_create_layout_trans_node(mut->graph, pool, "NHWC", attrs->layout);
        }
    }
    if (node->op_type == OP_BATCHNORM) {
        struct sf_bn_attrs *attrs = node->attrs;
        if (strcmp(attrs->layout, "NHWC") != 0) {
            new_args[0] = sf_create_layout_trans_node(mut->graph, new_args[0], attrs->layout, "NHWC");
            struct sf_node *norm = sf_create_batchnorm_node(mut->graph, new_args[0], new_args[1], new_args[2],
                                                            new_args[3], new_args[4], attrs->epsilon, "NHWC");
            return sf_create_layout_trans_node(mut->graph, norm, "NHWC", attrs->layout);
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// convert tensor layout to (NHWC, OHWI)
struct sf_mutator sf_convert_layout_NHWC_OHWI(void)
{
    return (struct sf_mutator){.visit = _visit};
}


