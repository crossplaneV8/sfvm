
#include "backend.h"


// builder (converts graph to engine)
struct sf_builder
{
    struct sf_graph *graph;         // source graph

    int num_regs;                   // num of all registers
    int num_idle;                   // num of idle registers
    struct sf_reg_info *reg_info;   // list of register infos
    int *idle_list;                 // list of idle registers
    int *reg_map;                   // node id ==> register id

    int num_code, buf_len;
    int *code_buf;                  // buffer for generated codes

    struct sf_list *inputs;         // list of input nodes
};


static struct sf_builder *_create_builder(struct sf_graph *graph)
{
    struct sf_builder *builder = malloc(sizeof(struct sf_builder));
    memset(builder, 0, sizeof(struct sf_builder));

    builder->graph = graph;

    builder->num_regs = 0;
    builder->num_idle = 0;

    const int num = graph->nodes->cnt;
    builder->reg_info = malloc(num * sizeof(struct sf_reg_info));
    builder->idle_list = malloc(num * sizeof(int));
    builder->reg_map = malloc(num * sizeof(int));

    builder->num_code = 0;
    builder->buf_len = 256;
    builder->code_buf = malloc(builder->buf_len * sizeof(int));

    builder->inputs = sf_create_list();

    return builder;
}


static void _discard_builder(struct sf_builder *builder)
{
    if (builder != NULL) {
        if (builder->reg_info != NULL) {
            free(builder->reg_info);
        }
        if (builder->idle_list != NULL) {
            free(builder->idle_list);
        }
        if (builder->reg_map != NULL) {
            free(builder->reg_map);
        }
        if (builder->code_buf != NULL) {
            free(builder->code_buf);
        }
        if (builder->inputs != NULL) {
            sf_discard_list(builder->inputs);
        }
        free(builder);
    }
}


// generate inference engine from builder
static struct sf_engine *_make_engine(struct sf_builder *builder)
{
    struct sf_engine *engine = malloc(sizeof(struct sf_engine));
    memset(engine, 0, sizeof(struct sf_engine));

    engine->alloc = sf_create_allocator();
    engine->num_regs = builder->num_regs;
    engine->reg_info = sf_malloc(engine->alloc, engine->num_regs * sizeof(struct sf_reg_info));
    memcpy(engine->reg_info, builder->reg_info, engine->num_regs * sizeof(struct sf_reg_info));

    engine->addr = sf_malloc(engine->alloc, engine->num_regs * sizeof(void*));
    for (int i=0; i<engine->num_regs; i++) {
        engine->addr[i] = sf_malloc(engine->alloc, engine->reg_info[i].size);
        if (engine->reg_info[i].data != NULL) {
            memcpy(engine->addr[i], engine->reg_info[i].data, engine->reg_info[i].size);
            engine->reg_info[i].data = engine->addr[i];
        }
    }
    engine->num_code = builder->num_code;
    engine->vm_code = sf_malloc(engine->alloc, engine->num_code * sizeof(int));
    memcpy(engine->vm_code, builder->code_buf, engine->num_code * sizeof(int));

    // extract I/O info
    engine->num_i = builder->inputs->cnt;
    engine->num_o = builder->graph->outputs->cnt;
    engine->i_names = sf_malloc(engine->alloc, engine->num_i * sizeof(void*));
    engine->i_regs = sf_malloc(engine->alloc, engine->num_i * sizeof(int));
    engine->o_regs = sf_malloc(engine->alloc, engine->num_o * sizeof(int));

    for (int i=0; i<engine->num_i; i++) {
        struct sf_node *input = builder->inputs->buf[i];
        struct sf_input_attrs *attrs = input->attrs;
        engine->i_regs[i] = builder->reg_map[input->index];
        engine->i_names[i] = sf_malloc(engine->alloc, SF_MAX_STR_LEN);
        strcpy(engine->i_names[i], attrs->name);
    }
    for (int i=0; i<engine->num_o; i++) {
        struct sf_node *output = builder->graph->outputs->buf[i];
        engine->o_regs[i] = builder->reg_map[output->index];
    }
    return engine;
}


// discard inference engine
void sf_discard_engine(struct sf_engine *engine)
{
    if (engine != NULL) {
        sf_discard_allocator(engine->alloc);
        free(engine);
    }
}


static void _push_code(struct sf_builder *builder, int code)
{
    if (builder->num_code == builder->buf_len) {
        int *tmp = builder->code_buf;
        builder->buf_len *= 2;
        builder->code_buf = malloc(builder->buf_len * sizeof(int));
        memcpy(builder->code_buf, tmp, builder->num_code * sizeof(int));
        free(tmp);
    }
    builder->code_buf[builder->num_code++] = code;
}


static void _push_codes(struct sf_builder *builder, int num, const int *codes)
{
    for (int i=0; i<num; i++) {
        _push_code(builder, codes[i]);
    }
}


// allocate unique register (for input or constant node)
static int _alloc_reg_unique(struct sf_builder *builder, size_t size)
{
    const int reg = builder->num_regs++;
    builder->reg_info[reg].size = size;
    builder->reg_info[reg].ref_cnt = 0xffff;
    builder->reg_info[reg].data = NULL;
    return reg;
}


// allocate reusable register
static int _alloc_reg(struct sf_builder *builder, size_t size)
{
    if (builder->num_idle > 0) {
        int reg = builder->idle_list[--(builder->num_idle)];
        if (size > builder->reg_info[reg].size) {
            builder->reg_info[reg].size = size;
        }
        return reg;
    }
    const int reg = builder->num_regs++;
    builder->reg_info[reg].size = size;
    builder->reg_info[reg].ref_cnt = 0;
    builder->reg_info[reg].data = NULL;
    return reg;
}


// gen code for input node
static int _gen_input(struct sf_builder *builder, struct sf_node *node)
{
    sf_list_append(builder->inputs, node);
    return _alloc_reg_unique(builder, sf_tensor_size(node->o_desc));
}


// gen code for constant node
static int _gen_const(struct sf_builder *builder, struct sf_node *node)
{
    int reg = _alloc_reg_unique(builder, sf_tensor_size(node->o_desc));
    struct sf_const_attrs *attrs = node->attrs;
    builder->reg_info[reg].data = attrs->data;
    return reg;
}


static int _same_shape(struct sf_node *x, struct sf_node *y)
{
    struct sf_tensor_desc x_desc = x->o_desc, y_desc = y->o_desc;
    if (x_desc.num_dims == y_desc.num_dims) {
        for (int i=0; i<x_desc.num_dims; i++) {
            if (x_desc.shape[i] != y_desc.shape[i]) return 0;
        }
        return 1;
    }
    return 0;
}


// replace high rank broadcast with low rank broadcast
static int _simplify_broadcast(struct sf_node *node, int *x_shape, int *y_shape)
{
    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    struct sf_tensor_desc y_desc = node->args[1]->o_desc;
    struct sf_tensor_desc z_desc = node->o_desc;

    int flag[SF_MAX_DIMS] = {0};

    for (int i=0; i<x_desc.num_dims; i++) {
        if (z_desc.shape[i] == 1) {flag[i] = 0;}
        else if (x_desc.shape[i] < z_desc.shape[i]) {flag[i] = 1;}
        else if (y_desc.shape[i] < z_desc.shape[i]) {flag[i] = 2;}
        else {flag[i] = 3;}
    }

    const int fuse_map[4][4] = {
        {1, 1, 1, 1},
        {1, 1, 0, 0},
        {1, 0, 1, 0},
        {1, 0, 0, 1},
    };
    int cnt = 1;
    x_shape[0] = x_desc.shape[0];
    y_shape[0] = y_desc.shape[0];

    for (int i=1; i<x_desc.num_dims; i++) {
        if (fuse_map[flag[i-1]][flag[i]]) {
            x_shape[cnt-1] *= x_desc.shape[i];
            y_shape[cnt-1] *= y_desc.shape[i];
        } else {
            x_shape[cnt] = x_desc.shape[i];
            y_shape[cnt] = y_desc.shape[i];
            cnt++;
        }
    }
    return cnt;
}


// gen code for add node
static int _gen_add(struct sf_builder *builder, struct sf_node *node)
{
    struct sf_node *x = node->args[0], *y = node->args[1];
    int x_reg = builder->reg_map[x->index];
    int y_reg = builder->reg_map[y->index];
    int z_reg = -1;

    if (_same_shape(x, node) && builder->reg_info[x_reg].ref_cnt == 1) {
        z_reg = x_reg;  // inplace calc
    } else if (_same_shape(y, node) && builder->reg_info[y_reg].ref_cnt == 1) {
        z_reg = y_reg;  // inplace calc
    } else {
        z_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));
    }

    int x_shape[SF_MAX_DIMS], y_shape[SF_MAX_DIMS];
    int dims = _simplify_broadcast(node, x_shape, y_shape);

    if (1 <= dims && dims <= 4) {
        const int instr[] = {
            (int)VM_ADD_1D_F32,
            (int)VM_ADD_2D_F32,
            (int)VM_ADD_3D_F32,
            (int)VM_ADD_4D_F32,
        };
        _push_code(builder, instr[dims-1]);
        _push_code(builder, x_reg);
        _push_code(builder, y_reg);
        _push_code(builder, z_reg);
        _push_codes(builder, dims, x_shape);
        _push_codes(builder, dims, y_shape);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        assert(0);
    }
    return z_reg;
}


// gen code for add-relu node
static int _gen_add_relu(struct sf_builder *builder, struct sf_node *node)
{
    int x_reg = builder->reg_map[node->args[0]->index];
    int y_reg = builder->reg_map[node->args[1]->index];
    int z_reg = -1;

    if (builder->reg_info[x_reg].ref_cnt == 1) {
        z_reg = x_reg;  // inplace calc
    } else if (builder->reg_info[y_reg].ref_cnt == 1) {
        z_reg = y_reg;  // inplace calc
    } else {
        z_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));
    }
    _push_code(builder, (int)VM_ADD_RELU_F32);
    _push_code(builder, x_reg);
    _push_code(builder, y_reg);
    _push_code(builder, z_reg);
    _push_code(builder, sf_tensor_prod(node->o_desc));
    return z_reg;
}


// gen code for convolution node
static int _gen_conv(struct sf_builder *builder, struct sf_node *node)
{
    struct sf_conv_attrs *attrs = node->attrs;

    int x_reg = builder->reg_map[node->args[0]->index];
    int w_reg = builder->reg_map[node->args[1]->index];
    int y_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));
    int b_reg = 0;
    if (node->num_args == 3) {
        b_reg = builder->reg_map[node->args[2]->index];
    }
    if (strcmp(attrs->x_layout, "NHWC") == 0) {
        if (strcmp(attrs->w_layout, "OHWI") == 0) {
            _push_code(builder, (int)VM_CONV_NHWC_OHWI_F32);
        } else {
            assert(strcmp(attrs->w_layout, "NK16") == 0);
            _push_code(builder, (int)VM_CONV_NHWC_NK16_F32);
        }
        _push_code(builder, x_reg);
        _push_code(builder, w_reg);
        _push_code(builder, b_reg);
        _push_code(builder, y_reg);

        int conv_params[] = {
            attrs->pad_h0, attrs->pad_w0,
            attrs->stride_h, attrs->stride_w,
            attrs->kernel_h, attrs->kernel_w,
            attrs->dilate_h, attrs->dilate_w,
            attrs->has_relu,
        };
        _push_codes(builder, 4, node->args[0]->o_desc.shape);
        _push_codes(builder, 4, node->o_desc.shape);
        _push_codes(builder, 9, conv_params);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        assert(0);
    }
    return y_reg;
}


// gen code for maxpool node
static int _gen_maxpool(struct sf_builder *builder, struct sf_node *node)
{
    struct sf_pool_attrs *attrs = node->attrs;
    int pool_params[] = {
        attrs->pad_h0, attrs->pad_w0,
        attrs->stride_h, attrs->stride_w,
        attrs->kernel_h, attrs->kernel_w,
    };
    int x_reg = builder->reg_map[node->args[0]->index];
    int y_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));

    if (strcmp(attrs->layout, "NHWC") == 0) {
        _push_code(builder, (int)VM_MAX_POOL_NHWC_F32);
        _push_code(builder, x_reg);
        _push_code(builder, y_reg);
        _push_codes(builder, 4, node->args[0]->o_desc.shape);
        _push_codes(builder, 4, node->o_desc.shape);
        _push_codes(builder, 6, pool_params);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        assert(0);
    }
    return y_reg;
}


// gen code for global-avgpool node
static int _gen_gl_avgpool(struct sf_builder *builder, struct sf_node *node)
{
    struct sf_pool_attrs *attrs = node->attrs;
    int x_reg = builder->reg_map[node->args[0]->index];
    int y_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));

    if (strcmp(attrs->layout, "NHWC") == 0) {
        _push_code(builder, (int)VM_GAVG_POOL_NHWC_F32);
        _push_code(builder, x_reg);
        _push_code(builder, y_reg);
        _push_codes(builder, 4, node->args[0]->o_desc.shape);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        assert(0);
    }
    return y_reg;
}


// gen code for relu node
static int _gen_relu(struct sf_builder *builder, struct sf_node *node)
{
    int x_reg = builder->reg_map[node->args[0]->index];
    int y_reg = -1;
    if (builder->reg_info[x_reg].ref_cnt == 1) {
        y_reg = x_reg;
    } else {
        y_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));
    }
    _push_code(builder, (int)VM_RELU_F32);
    _push_code(builder, x_reg);
    _push_code(builder, y_reg);
    _push_code(builder, sf_tensor_prod(node->o_desc));
    return y_reg;
}


// gen code for reshape node
static int _gen_reshape(struct sf_builder *builder, struct sf_node *node)
{
    int x_reg = builder->reg_map[node->args[0]->index];
    int y_reg = -1;
    if (builder->reg_info[x_reg].ref_cnt == 1) {
        y_reg = x_reg;
    } else {
        y_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));
    }
    _push_code(builder, (int)VM_COPY_F32);
    _push_code(builder, x_reg);
    _push_code(builder, y_reg);
    _push_code(builder, sf_tensor_prod(node->o_desc));
    return y_reg;
}


// replace high rank transpose with low rank transpose
static int _simplify_transpose(struct sf_node *node, int *new_shape, int *new_axes)
{
    struct sf_tensor_desc x_desc = node->args[0]->o_desc;
    struct sf_transpose_attrs *attrs = node->attrs;
    int shape[SF_MAX_DIMS], index[SF_MAX_DIMS], cnt = 0;

    for (int i=0; i<x_desc.num_dims; i++) {
        shape[i] = x_desc.shape[i];
    }
    for (int i=1; i<x_desc.num_dims; i++) {
        if (attrs->axes[i-1] + 1 == attrs->axes[i]) {
            shape[attrs->axes[i]] *= shape[attrs->axes[i-1]];
            shape[attrs->axes[i-1]] = 1;
        }
    }
    for (int i=0, n=0; i<x_desc.num_dims; i++) {
        if (shape[i] > 1) {
            new_shape[n] = shape[i];
            index[i] = n++;
        }
    }
    for (int i=0; i<x_desc.num_dims; i++) {
        if (shape[attrs->axes[i]] > 1) {
            new_axes[cnt++] = index[attrs->axes[i]];
        }
    }
    return cnt;
}


// gen code for transpose node
static int _gen_transpose(struct sf_builder *builder, struct sf_node *node)
{
    int x_reg = builder->reg_map[node->args[0]->index];
    int y_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));

    int new_shape[SF_MAX_DIMS], new_axes[SF_MAX_DIMS];
    int dims = _simplify_transpose(node, new_shape, new_axes);

    if (2 <= dims && dims <= 4) {
        const int instr[] = {
            (int)VM_TRANSPOSE_2D_F32,
            (int)VM_TRANSPOSE_3D_F32,
            (int)VM_TRANSPOSE_4D_F32,
        };
        _push_code(builder, instr[dims-2]);
        _push_code(builder, x_reg);
        _push_code(builder, y_reg);
        _push_codes(builder, dims, new_shape);
        _push_codes(builder, dims, new_axes);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        assert(0);
    }
    return y_reg;
}


// gen code for gemm node
static int _gen_gemm(struct sf_builder *builder, struct sf_node *node)
{
    struct sf_gemm_attrs *attrs = node->attrs;
    int x_reg = builder->reg_map[node->args[0]->index];
    int w_reg = builder->reg_map[node->args[1]->index];
    int y_reg = _alloc_reg(builder, sf_tensor_size(node->o_desc));
    int b_reg = 0;
    if (node->num_args == 3) {
        b_reg = builder->reg_map[node->args[2]->index];
    }
    _push_code(builder, (int)VM_GEMM_F32);
    _push_code(builder, x_reg);
    _push_code(builder, w_reg);
    _push_code(builder, b_reg);
    _push_code(builder, y_reg);
    _push_code(builder, attrs->trans_a);
    _push_code(builder, attrs->trans_b);
    _push_code(builder, node->o_desc.shape[0]);
    _push_code(builder, node->o_desc.shape[1]);
    _push_code(builder, node->args[1]->o_desc.shape[attrs->trans_b]);
    _push_code(builder, 0); // has_relu = 0
    return y_reg;
}


// build graph with a builder
static void _build_graph(struct sf_builder *builder, struct sf_graph *graph)
{
    for (int i=0; i<graph->nodes->cnt; i++) {
        struct sf_node *node = graph->nodes->buf[i];
        int reg = -1;

        switch (node->op_type) {
            case OP_INPUT:      reg = _gen_input(builder, node); break;
            case OP_CONST:      reg = _gen_const(builder, node); break;
            case OP_ADD:        reg = _gen_add(builder, node); break;
            case OP_ADD_RELU:   reg = _gen_add_relu(builder, node); break;
            case OP_CONV:       reg = _gen_conv(builder, node); break;
            case OP_MAX_POOL:   reg = _gen_maxpool(builder, node); break;
            case OP_G_AVG_POOL: reg = _gen_gl_avgpool(builder, node); break;
            case OP_RELU:       reg = _gen_relu(builder, node); break;
            case OP_RESHAPE:    reg = _gen_reshape(builder, node); break;
            case OP_TRANSPOSE:  reg = _gen_transpose(builder, node); break;
            case OP_GEMM:       reg = _gen_gemm(builder, node); break;

            default: break;
        }

        if (reg != -1) {
            builder->reg_info[reg].ref_cnt += node->ref_num;
            builder->reg_map[node->index] = reg;
        } else {
            printf("not implemented:\n");
            sf_print_node(stdout, node);
            assert(0);
        }

        // update ref-cnt, move idle registers into idle-list
        for (int j=0; j<node->num_args; j++) {
            int r = builder->reg_map[node->args[j]->index];
            if (--(builder->reg_info[r].ref_cnt) == 0) {
                builder->idle_list[builder->num_idle++] = r;
            }
        }
    }
    _push_code(builder, (int)VM_STOP);
}


// generate inference engine from graph
struct sf_engine *sf_engine_from_graph(struct sf_graph *graph)
{
    struct sf_builder *builder = _create_builder(graph);
    _alloc_reg_unique(builder, 0);  // register 0 is null
    _build_graph(builder, graph);

    struct sf_engine *engine = _make_engine(builder);
    _discard_builder(builder);

    return engine;
}


