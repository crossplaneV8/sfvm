
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx.pb-c.h"
#include "graph/graph.h"


static Onnx__ModelProto *_load_onnx_model(const char *path)
{
    Onnx__ModelProto *model = NULL;
    FILE *fp = fopen(path, "rb");

    if (fp != NULL) {
        fseek(fp, 0L, SEEK_END);
        size_t size = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        void *buf = malloc(size);

        if (buf != NULL) {
            if (fread(buf, size, 1, fp) == 1) {
                model = onnx__model_proto__unpack(NULL, size, buf);
            }
            free(buf);
        }
        fclose(fp);
    }
    return model;
}


static void _discard_onnx_model(Onnx__ModelProto *model)
{
    protobuf_c_message_free_unpacked((ProtobufCMessage*)model, NULL);
}


static Onnx__TensorProto *_get_init_by_name(Onnx__GraphProto *graph, const char *name)
{
    for (int i=0; i<graph->n_initializer; i++) {
        if (strcmp(graph->initializer[i]->name, name) == 0) {
            return graph->initializer[i];
        }
    }
    return NULL;
}


static void _get_tensor_info(Onnx__TensorProto *init, struct sf_tensor_desc *desc, void **data)
{
    if (init->has_data_type) {
        switch (init->data_type) {
            case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:       desc->dtype = SF_INT8;  break;
            case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:      desc->dtype = SF_INT16; break;
            case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:      desc->dtype = SF_INT32; break;
            case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:      desc->dtype = SF_INT64; break;
            case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:    desc->dtype = SF_FLOAT16; break;
            case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:      desc->dtype = SF_FLOAT32; break;
            case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:     desc->dtype = SF_FLOAT64; break;
            default: break;
        }
    }
    if (init->float_data != NULL) {
        *data = init->float_data;
    } else if (init->double_data != NULL) {
        *data = init->double_data;
    } else if (init->int32_data != NULL) {
        *data = init->int32_data;
    } else if (init->int64_data != NULL) {
        *data = init->int64_data;
    } else if (init->raw_data.data != NULL) {
        *data = init->raw_data.data;
    }
    desc->num_dims = (int)(init->n_dims);
    for (int i=0; i<desc->num_dims; i++) {
        desc->shape[i] = (int)(init->dims[i]);
    }
}


static struct sf_node *_tensor_to_const_node(struct sf_graph *graph, Onnx__TensorProto *tens)
{
    struct sf_tensor_desc desc = {SF_UNKNOWN};
    void *data = NULL;
    _get_tensor_info(tens, &desc, &data);
    return sf_create_const_node(graph, desc, data);
}


static void _convert_conv(struct sf_graph *dst, Onnx__GraphProto *src,
                          Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input >= 2 && node->n_input <= 3);
    assert(node->n_output == 1);

    struct sf_node *bias = (node->n_input == 3) ? args[2] : NULL;
    int pad_h0 = 0, pad_h1 = 0, pad_w0 = 0, pad_w1 = 0;
    int stride_h = 0, stride_w = 0;
    int dilate_h = 0, dilate_w = 0;

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];

        if (strcmp(attr->name, "group") == 0) {
            if (attr->has_i && attr->i != 1) {
                printf("group convolution not implemented!\n"); abort();
            }
        }
        if (strcmp(attr->name, "pads") == 0) {
            assert(attr->n_ints == 4);
            pad_h0 = attr->ints[0];
            pad_h1 = attr->ints[2];
            pad_w0 = attr->ints[1];
            pad_w1 = attr->ints[3];
        }
        if (strcmp(attr->name, "strides") == 0) {
            assert(attr->n_ints == 2);
            stride_h = attr->ints[0];
            stride_w = attr->ints[1];
        }
        if (strcmp(attr->name, "dilations") == 0) {
            assert(attr->n_ints == 2);
            dilate_h = attr->ints[0];
            dilate_w = attr->ints[1];
        }
    }
    sf_create_conv_node(dst, args[0], args[1], bias, "NCHW", "OIHW",
                        pad_h0, pad_h1, pad_w0, pad_w1, stride_h,
                        stride_w, dilate_h, dilate_w, 0, 0, 0, 0);
}


static void _convert_batchnorm(struct sf_graph *dst, Onnx__GraphProto *src,
                               Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input == 5);
    assert(node->n_output == 1);

    double eps = 1e-5;

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];
        if (strcmp(attr->name, "epsilon") == 0) {
            eps = attr->f;
        }
    }
    sf_create_batchnorm_node(dst, args[0], args[1], args[2], args[3], args[4], eps, "NCHW");
}


static void _convert_ReLU(struct sf_graph *dst, Onnx__GraphProto *src,
                          Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input == 1);
    assert(node->n_output == 1);
    sf_create_relu_node(dst, args[0]);
}


static void _convert_sigmoid(struct sf_graph *dst, Onnx__GraphProto *src,
                             Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input == 1);
    assert(node->n_output == 1);
    sf_create_sigmoid_node(dst, args[0]);
}


static void _convert_softmax(struct sf_graph *dst, Onnx__GraphProto *src,
                             Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input == 1);
    assert(node->n_output == 1);

    int axis = -1;

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];
        if (strcmp(attr->name, "axis") == 0) {
            axis = (int)(attr->i);
        }
    }
    sf_create_softmax_node(dst, args[0], axis);
}


static void _convert_add(struct sf_graph *dst, Onnx__GraphProto *src,
                         Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input == 2);
    assert(node->n_output == 1);
    sf_create_add_node(dst, args[0], args[1]);
}


static void _convert_pool(struct sf_graph *dst, Onnx__GraphProto *src,
                          Onnx__NodeProto *node, struct sf_node **args,
                          enum sf_op_type pool_type)
{
    assert(node->n_input == 1);
    assert(node->n_output == 1);

    int pad_h0 = 0, pad_h1 = 0, pad_w0 = 0, pad_w1 = 0;
    int stride_h = 0, stride_w = 0;
    int kernel_h = 0, kernel_w = 0;

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];

        if (strcmp(attr->name, "pads") == 0) {
            assert(attr->n_ints == 4);
            pad_h0 = attr->ints[0];
            pad_h1 = attr->ints[2];
            pad_w0 = attr->ints[1];
            pad_w1 = attr->ints[3];
        }
        if (strcmp(attr->name, "strides") == 0) {
            assert(attr->n_ints == 2);
            stride_h = attr->ints[0];
            stride_w = attr->ints[1];
        }
        if (strcmp(attr->name, "kernel_shape") == 0) {
            assert(attr->n_ints == 2);
            kernel_h = attr->ints[0];
            kernel_w = attr->ints[1];
        }
        if (strcmp(attr->name, "dilations") == 0) {
            assert(attr->n_ints == 2);
            if (attr->ints[0] != 1 || attr->ints[1] != 1) {
                printf("dilated pooling not implemented!\n"); abort();
            }
        }
    }
    sf_create_pool_node(dst, args[0], pad_h0, pad_h1, pad_w0, pad_w1,
                        stride_h, stride_w, kernel_h, kernel_w, pool_type, "NCHW");
}


static void _convert_global_pool(struct sf_graph *dst, Onnx__GraphProto *src,
                                 Onnx__NodeProto *node, struct sf_node **args,
                                 enum sf_op_type pool_type)
{
    assert(node->n_input == 1);
    assert(node->n_output == 1);
    sf_create_global_pool_node(dst, args[0], pool_type, "NCHW");
}


static void _convert_gemm(struct sf_graph *dst, Onnx__GraphProto *src,
                          Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input >= 2 && node->n_input <= 3);
    assert(node->n_output == 1);

    float alpha = 1.0, beta = 1.0;
    int trans_a = 0, trans_b = 0;

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];

        if (strcmp(attr->name, "alpha") == 0) {
            alpha = attr->f;
        }
        if (strcmp(attr->name, "beta") == 0) {
            beta = attr->f;
        }
        if (strcmp(attr->name, "transA") == 0) {
            trans_a = attr->i;
        }
        if (strcmp(attr->name, "transB") == 0) {
            trans_b = attr->i;
        }
    }
    struct sf_node *c = (node->n_input == 3) ? args[2] : NULL;
    sf_create_gemm_node(dst, args[0], args[1], c, alpha, beta, trans_a, trans_b);
}


static void _convert_matmul(struct sf_graph *dst, Onnx__GraphProto *src,
                            Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input == 2);
    assert(node->n_output == 1);

    float alpha = 1.0, beta = 0.0;
    int trans_a = 0, trans_b = 0;
    sf_create_gemm_node(dst, args[0], args[1], NULL, alpha, beta, trans_a, trans_b);
}


static void _convert_identity(struct sf_graph *dst, Onnx__GraphProto *src,
                              Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input >= 1);
    assert(node->n_output >= 1);
    sf_create_identity_node(dst, args[0]);
}


static void _convert_concat(struct sf_graph *dst, Onnx__GraphProto *src,
                            Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input >= 1 && node->n_input <= SF_MAX_ARGS);
    assert(node->n_output == 1);
    int axis = 0;

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];
        if (strcmp(attr->name, "axis") == 0) {
            axis = (int)(attr->i);
        }
    }
    sf_create_concat_node(dst, axis, node->n_input, args);
}


static void _convert_flatten(struct sf_graph *dst, Onnx__GraphProto *src,
                             Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input == 1);
    assert(node->n_output == 1);
    int axis = 1;

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];
        if (strcmp(attr->name, "axis") == 0) {
            axis = (int)(attr->i);
        }
    }
    sf_create_flatten_node(dst, args[0], axis);
}


static void _convert_squeeze(struct sf_graph *dst, Onnx__GraphProto *src,
                             Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input >= 1 && node->n_input <= 2);
    assert(node->n_output == 1);

    int num = 0, axes[SF_MAX_DIMS] = {0};

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];

        if (strcmp(attr->name, "axes") == 0) {
            num = attr->n_ints;
            for (int j=0; j<num; j++) {
                axes[j] = (int)(attr->ints[j]);
            }
        }
    }
    if (node->n_input == 2) {
        if (args[1]->op_type == OP_CONST) {
            struct sf_const_attrs *attrs = args[1]->attrs;
            const struct sf_tensor_desc desc = attrs->data_desc;
            assert(desc.dtype == SF_INT64 && desc.num_dims == 1);
            const int64_t *data = (const int64_t*)(attrs->data);
            num = desc.shape[0];
            for (int i=0; i<num; i++) {
                axes[i] = (int)(data[i]);
            }
        } else {
            printf("squeeze node does not support non-constant axes"); abort();
        }
    }
    sf_create_squeeze_node(dst, args[0], num, axes);
}


static void _convert_reshape(struct sf_graph *dst, Onnx__GraphProto *src,
                             Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input >= 1 && node->n_input <= 2);
    assert(node->n_output == 1);

    int num = 0, shape[SF_MAX_DIMS] = {0};

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];

        if (strcmp(attr->name, "shape") == 0) {
            num = attr->n_ints;
            for (int j=0; j<num; j++) {
                shape[j] = (int)(attr->ints[j]);
            }
        }
    }
    if (node->n_input == 2) {
        if (args[1]->op_type == OP_CONST) {
            struct sf_const_attrs *attrs = args[1]->attrs;
            const struct sf_tensor_desc desc = attrs->data_desc;
            assert(desc.dtype == SF_INT64 && desc.num_dims == 1);
            const int64_t *data = (const int64_t*)(attrs->data);
            num = desc.shape[0];
            for (int i=0; i<num; i++) {
                shape[i] = (int)(data[i]);
            }
        } else {
            printf("reshape node does not support non-constant shape"); abort();
        }
    }
    sf_create_reshape_node(dst, args[0], num, shape);
}


static void _convert_transpose(struct sf_graph *dst, Onnx__GraphProto *src,
                               Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_input == 1);
    assert(node->n_output == 1);

    int num_dims = 0, axes[SF_MAX_DIMS] = {0};

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];

        if (strcmp(attr->name, "perm") == 0) {
            num_dims = attr->n_ints;
            for (int j=0; j<num_dims; j++) {
                axes[j] = (int)(attr->ints[j]);
            }
        }
    }
    sf_create_transpose_node(dst, args[0], num_dims, axes);
}


static void _convert_reduce(struct sf_graph *dst, Onnx__GraphProto *src,
                            Onnx__NodeProto *node, struct sf_node **args,
                            enum sf_op_type type)
{
    assert(node->n_input >= 1 && node->n_input <= 2);
    assert(node->n_output == 1);

    int num_axes = 0;
    int axes[SF_MAX_DIMS];
    int keep_dims = 1;

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];

        if (strcmp(attr->name, "axes") == 0) {
            num_axes = attr->n_ints;
            for (int j=0; j<num_axes; j++) {
                axes[j] = attr->ints[j];
            }
        }
        if (strcmp(attr->name, "keepdims") == 0) {
            keep_dims = (int)(attr->i);
        }
    }
    if (node->n_input == 2) {
        if (args[1]->op_type == OP_CONST) {
            struct sf_const_attrs *attrs = args[1]->attrs;
            const struct sf_tensor_desc desc = attrs->data_desc;
            assert(desc.dtype == SF_INT64 && desc.num_dims == 1);
            const int64_t *data = (const int64_t*)(attrs->data);
            num_axes = desc.shape[0];
            for (int i=0; i<num_axes; i++) {
                axes[i] = (int)(data[i]);
            }
        } else {
            printf("reduce node does not support non-constant axes"); abort();
        }
    }
    sf_create_reduce_node(dst, args[0], num_axes, axes, keep_dims, type);
}


static void _convert_constant(struct sf_graph *dst, Onnx__GraphProto *src,
                              Onnx__NodeProto *node, struct sf_node **args)
{
    assert(node->n_output == 1);

    for (int i=0; i<node->n_attribute; i++) {
        Onnx__AttributeProto *attr = node->attribute[i];
        if (strcmp(attr->name, "value") == 0) {
            if (attr->t != NULL) {
                _tensor_to_const_node(dst, attr->t); return;
            }
        }
    }
}


static void _convert_node(struct sf_graph *dst, Onnx__GraphProto *src,
                          Onnx__NodeProto *node, struct sf_node **args)
{
    if (strcmp(node->op_type, "Conv") == 0) {
        _convert_conv(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "BatchNormalization") == 0) {
        _convert_batchnorm(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Relu") == 0) {
        _convert_ReLU(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Sigmoid") == 0) {
        _convert_sigmoid(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Softmax") == 0) {
        _convert_softmax(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Add") == 0) {
        _convert_add(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "AveragePool") == 0) {
        _convert_pool(dst, src, node, args, OP_AVG_POOL);
    }
    else if (strcmp(node->op_type, "MaxPool") == 0) {
        _convert_pool(dst, src, node, args, OP_MAX_POOL);
    }
    else if (strcmp(node->op_type, "GlobalAveragePool") == 0) {
        _convert_global_pool(dst, src, node, args, OP_G_AVG_POOL);
    }
    else if (strcmp(node->op_type, "GlobalMaxPool") == 0) {
        _convert_global_pool(dst, src, node, args, OP_G_MAX_POOL);
    }
    else if (strcmp(node->op_type, "Gemm") == 0) {
        _convert_gemm(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "MatMul") == 0) {
        _convert_matmul(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Identity") == 0) {
        _convert_identity(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Dropout") == 0) {
        _convert_identity(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Concat") == 0) {
        _convert_concat(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Flatten") == 0) {
        _convert_flatten(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Squeeze") == 0) {
        _convert_squeeze(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Reshape") == 0) {
        _convert_reshape(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "Transpose") == 0) {
        _convert_transpose(dst, src, node, args);
    }
    else if (strcmp(node->op_type, "ReduceSum") == 0) {
        _convert_reduce(dst, src, node, args, OP_REDUCE_SUM);
    }
    else if (strcmp(node->op_type, "ReduceMean") == 0) {
        _convert_reduce(dst, src, node, args, OP_REDUCE_AVG);
    }
    else if (strcmp(node->op_type, "ReduceMax") == 0) {
        _convert_reduce(dst, src, node, args, OP_REDUCE_MAX);
    }
    else if (strcmp(node->op_type, "ReduceMin") == 0) {
        _convert_reduce(dst, src, node, args, OP_REDUCE_MIN);
    }
    else if (strcmp(node->op_type, "Constant") == 0) {
        _convert_constant(dst, src, node, args);
    }
    else {
        printf("unhandled op: %s\n", node->op_type);
        abort();
    }
}


static int _find_name(struct sf_list *names, const char *name)
{
    for (int i=0; i<names->cnt; i++) {
        if (strcmp(name, names->buf[i]) == 0) {
            return i;
        }
    }
    return -1;
}


// convert onnx file to sfvm graph
struct sf_graph *sf_load_graph_from_onnx(const char *path)
{
    Onnx__ModelProto *model = _load_onnx_model(path);

    if (model != NULL) {
        Onnx__GraphProto *src = model->graph;
        struct sf_graph *dst = sf_create_graph();
        struct sf_list *names = sf_create_list();

        // convert inputs
        for (int i=0; i<src->n_input; i++) {
            char *name = src->input[i]->name;
            if (_get_init_by_name(src, name) == NULL) {
                struct sf_tensor_desc desc = {SF_UNKNOWN};
                sf_create_input_node(dst, name, desc);
                sf_list_append(names, name);
            }
        }

        // convert weights
        for (int i=0; i<src->n_initializer; i++) {
            _tensor_to_const_node(dst, src->initializer[i]);
            sf_list_append(names, src->initializer[i]->name);
        }

        // convert other nodes
        for (int i=0; i<src->n_node; i++) {
            Onnx__NodeProto *node = src->node[i];
            struct sf_node *args[SF_MAX_ARGS];
            for (int j=0; j<node->n_input; j++) {
                int idx = _find_name(names, node->input[j]);
                args[j] = dst->nodes->buf[idx];
            }
            _convert_node(dst, src, node, args);
            sf_list_append(names, node->output[0]);
        }

        // gather outputs
        for (int i=0; i<src->n_output; i++) {
            int idx = _find_name(names, src->output[i]->name);
            sf_list_append(dst->outputs, dst->nodes->buf[idx]);
        }

        _discard_onnx_model(model);
        sf_discard_list(names);
        return dst;
    }
    return NULL;
}


