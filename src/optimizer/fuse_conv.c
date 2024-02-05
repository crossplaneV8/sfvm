
#include "mutator.h"


static int _is_broadcast_c(const char *layout, int channel, struct sf_tensor_desc desc)
{
    int shape[SF_MAX_DIMS] = {1, 1, 1, 1};
    shape[find_axis(layout, 'C')] = channel;

    for (int i=0; i<desc.num_dims; i++) {
        if (desc.shape[i] != shape[i]) return 0;
    }
    return 1;
}


// create new nodes while visiting old nodes
static struct sf_node *_visit(struct sf_mutator *mut, struct sf_node *node, struct sf_node **new_args)
{
    if (node->op_type == OP_MUL) {
        struct sf_node *conv = new_args[0], *scale = new_args[1];
        struct sf_conv_attrs *attrs = conv->attrs;
        if (conv->op_type == OP_CONV && attrs->has_relu == 0) {
            struct sf_node *args[] = {conv->args[0], conv->args[1], conv->args[2]};
            const int axis = find_axis(attrs->w_layout, 'O');
            const int outc = args[1]->o_desc.shape[axis];
            if (_is_broadcast_c(attrs->x_layout, outc, scale->o_desc)) {
                int shape[SF_MAX_DIMS] = {1, 1, 1, 1}; shape[axis] = outc;
                scale = sf_create_reshape_node(mut->graph, scale, 4, shape);
                args[1] = sf_create_mul_node(mut->graph, args[1], scale);
                return sf_clone_node(mut->graph, conv, args);
            }
        }
    }
    if (node->op_type == OP_ADD) {
        struct sf_node *conv = new_args[0], *beta = new_args[1];
        struct sf_conv_attrs *attrs = conv->attrs;
        if (conv->op_type == OP_CONV && attrs->has_relu == 0) {
            const int axis = find_axis(attrs->x_layout, 'C');
            const int outc = conv->o_desc.shape[axis];
            if (_is_broadcast_c(attrs->x_layout, outc, beta->o_desc)) {
                beta = sf_create_reshape_node(mut->graph, beta, 1, &outc);
                struct sf_node *bias = (conv->num_args == 3) ? conv->args[2] : NULL;
                if (bias != NULL) {
                    bias = sf_create_add_node(mut->graph, bias, beta);
                } else {
                    bias = beta;
                }
                return sf_create_conv_node(mut->graph, conv->args[0], conv->args[1], bias,
                                           attrs->x_layout, attrs->w_layout, attrs->pad_h0, attrs->pad_h1,
                                           attrs->pad_w0, attrs->pad_w1, attrs->stride_h, attrs->stride_w,
                                           attrs->dilate_h, attrs->dilate_w, attrs->has_relu);
            }
        }
    }
    if (node->op_type == OP_RELU) {
        struct sf_node *conv = new_args[0];
        if (conv->op_type == OP_CONV) {
            struct sf_conv_attrs *attrs = conv->attrs;
            return sf_create_conv_node(mut->graph, conv->args[0], conv->args[1], conv->args[2],
                                       attrs->x_layout, attrs->w_layout, attrs->pad_h0, attrs->pad_h1,
                                       attrs->pad_w0, attrs->pad_w1, attrs->stride_h, attrs->stride_w,
                                       attrs->dilate_h, attrs->dilate_w, 1);
        }
    }
    return sf_clone_node(mut->graph, node, new_args);
}


// fuse (mul, add, relu) into conv
struct sf_mutator sf_fuse_conv_mul_add_relu(void)
{
    return (struct sf_mutator){.visit = _visit};
}


