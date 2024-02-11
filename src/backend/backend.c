
#include "backend.h"


// create a new inference engine
struct sf_engine *sf_create_engine(void)
{
    struct sf_engine *engine = malloc(sizeof(struct sf_engine));
    memset(engine, 0, sizeof(struct sf_engine));

    engine->alloc = sf_create_allocator();

    engine->reg_cnt = 0;
    engine->reg_len = 32;
    engine->info = sf_malloc(engine->alloc, engine->reg_len * sizeof(struct sf_reg_info));

    engine->code_cnt = 0;
    engine->code_len = 256;
    engine->vm_code = sf_malloc(engine->alloc, engine->code_len * sizeof(int));

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


static void _push_code(struct sf_engine *engine, int code)
{
    if (engine->code_cnt == engine->code_len) {
        int *tmp = engine->vm_code;
        engine->code_len *= 2;
        engine->vm_code = sf_malloc(engine->alloc, engine->code_len * sizeof(int));
        memcpy(engine->vm_code, tmp, engine->code_cnt * sizeof(int));
        sf_free(tmp);
    }
    engine->vm_code[engine->code_cnt++] = code;
}


static void _push_codes(struct sf_engine *engine, int num, const int *codes)
{
    for (int i=0; i<num; i++) {
        _push_code(engine, codes[i]);
    }
}


static int _push_reg(struct sf_engine *engine)
{
    if (engine->reg_cnt == engine->reg_len) {
        struct sf_reg_info *tmp = engine->info;
        engine->reg_len *= 2;
        engine->info = sf_malloc(engine->alloc, engine->reg_len * sizeof(struct sf_reg_info));
        memcpy(engine->info, tmp, engine->reg_cnt * sizeof(struct sf_reg_info));
        sf_free(tmp);
    }
    return engine->reg_cnt++;
}


static int _get_rank(size_t size)
{
    int rank = 0;
    while ((64 << rank) < size) {
        rank++;
    }
    return rank;
}


static int _alloc_reg_unique(struct sf_engine *engine, struct sf_node *node)
{
    int reg = _push_reg(engine);
    size_t size = sf_tensor_size(node->o_desc);
    engine->info[reg].rank = _get_rank(size);
    engine->info[reg].size = size;
    engine->info[reg].ref_cnt = 1000000000;
    engine->info[reg].data = NULL;
    return reg;
}


static int _alloc_reg_shared(struct sf_engine *engine, struct sf_node *node)
{
    size_t size = sf_tensor_size(node->o_desc);
    int rank = _get_rank(size);

    for (int i=0; i<engine->reg_cnt; i++) {
        if (engine->info[i].ref_cnt == 0 && engine->info[i].rank == rank) {
            if (size > engine->info[i].size) {
                engine->info[i].size = size;
            }
            return i;
        }
    }
    int reg = _push_reg(engine);
    engine->info[reg].rank = rank;
    engine->info[reg].size = size;
    engine->info[reg].ref_cnt = 0;
    engine->info[reg].data = NULL;
    return reg;
}


static int _gen_input(struct sf_engine *engine, struct sf_node *node)
{
    return _alloc_reg_unique(engine, node);
}


static int _gen_const(struct sf_engine *engine, struct sf_node *node)
{
    int reg = _alloc_reg_unique(engine, node);
    struct sf_const_attrs *attrs = node->attrs;
    engine->info[reg].data = attrs->data;
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


static int _gen_add(struct sf_engine *engine, struct sf_node *node)
{
    struct sf_node *x = node->args[0], *y = node->args[1];
    int x_reg = engine->reg_map[x->index];
    int y_reg = engine->reg_map[y->index];
    int z_reg = -1;

    if (_same_shape(x, node) && engine->info[x_reg].ref_cnt == 1) {
        z_reg = x_reg;  // inplace calc
    } else if (_same_shape(y, node) && engine->info[y_reg].ref_cnt == 1) {
        z_reg = y_reg;  // inplace calc
    } else {
        z_reg = _alloc_reg_shared(engine, node);
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
        _push_code(engine, instr[dims-1]);
        _push_code(engine, x_reg);
        _push_code(engine, y_reg);
        _push_code(engine, z_reg);
        _push_codes(engine, dims, x_shape);
        _push_codes(engine, dims, y_shape);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        abort();
    }
    return z_reg;
}


static int _is_gemm_equivalent(struct sf_node *node)
{
    struct sf_conv_attrs *attrs = node->attrs;
    const int *x_shape = node->args[0]->o_desc.shape;
    const int *w_shape = node->args[1]->o_desc.shape;

    if (attrs->pad_h0 == 0 && attrs->pad_h1 == 0 && attrs->pad_w0 == 0 && attrs->pad_w1 == 0) {
        if (w_shape[1] == 1 && w_shape[2] == 1 && attrs->stride_h == 1 && attrs->stride_w == 1) {
            return 1;
        }
        if (w_shape[1] == 1 && w_shape[2] == x_shape[2] && attrs->stride_h == 1) {
            return 1;
        }
        if (w_shape[1] == x_shape[1] && w_shape[2] == x_shape[2]) {
            return 1;
        }
    }
    return 0;
}


static int _gen_conv(struct sf_engine *engine, struct sf_node *node)
{
    struct sf_conv_attrs *attrs = node->attrs;

    int x_reg = engine->reg_map[node->args[0]->index];
    int w_reg = engine->reg_map[node->args[1]->index];
    int y_reg = _alloc_reg_shared(engine, node);
    int b_reg = -1;
    if (node->num_args == 3) {
        b_reg = engine->reg_map[node->args[2]->index];
    }
    if (strcmp(attrs->x_layout, "NHWC") == 0 && strcmp(attrs->w_layout, "OHWI") == 0) {
        const int *x_shape = node->args[0]->o_desc.shape;
        const int *w_shape = node->args[1]->o_desc.shape;
        const int *y_shape = node->o_desc.shape;

        if (0 && _is_gemm_equivalent(node)) {
            _push_code(engine, (int)VM_GEMM_F32);
            _push_code(engine, x_reg);
            _push_code(engine, w_reg);
            _push_code(engine, b_reg);
            _push_code(engine, y_reg);
            _push_code(engine, 0);
            _push_code(engine, 1);
            _push_code(engine, y_shape[0] * y_shape[1] * y_shape[2]);
            _push_code(engine, w_shape[0]);
            _push_code(engine, w_shape[1] * w_shape[2] * w_shape[3]);
            _push_code(engine, attrs->has_relu);
        } else {
            _push_code(engine, (int)VM_CONV_NHWC_OHWI_F32);
            _push_code(engine, x_reg);
            _push_code(engine, w_reg);
            _push_code(engine, b_reg);
            _push_code(engine, y_reg);

            int conv_params[] = {
                attrs->pad_h0, attrs->pad_w0,
                attrs->stride_h, attrs->stride_w,
                w_shape[1], w_shape[2],
                attrs->dilate_h, attrs->dilate_w,
                attrs->has_relu,
            };
            _push_codes(engine, 4, x_shape);
            _push_codes(engine, 4, y_shape);
            _push_codes(engine, 9, conv_params);
        }
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        abort();
    }
    return y_reg;
}


static int _gen_maxpool(struct sf_engine *engine, struct sf_node *node)
{
    struct sf_pool_attrs *attrs = node->attrs;
    int pool_params[] = {
        attrs->pad_h0, attrs->pad_w0,
        attrs->stride_h, attrs->stride_w,
        attrs->kernel_h, attrs->kernel_w,
    };
    int x_reg = engine->reg_map[node->args[0]->index];
    int y_reg = _alloc_reg_shared(engine, node);

    if (strcmp(attrs->layout, "NHWC") == 0) {
        _push_code(engine, (int)VM_MAX_POOL_NHWC_F32);
        _push_code(engine, x_reg);
        _push_code(engine, y_reg);
        _push_codes(engine, 4, node->args[0]->o_desc.shape);
        _push_codes(engine, 4, node->o_desc.shape);
        _push_codes(engine, 6, pool_params);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        abort();
    }
    return y_reg;
}


static int _gen_gl_avgpool(struct sf_engine *engine, struct sf_node *node)
{
    struct sf_pool_attrs *attrs = node->attrs;
    int x_reg = engine->reg_map[node->args[0]->index];
    int y_reg = _alloc_reg_shared(engine, node);

    if (strcmp(attrs->layout, "NHWC") == 0) {
        _push_code(engine, (int)VM_GAVG_POOL_NHWC_F32);
        _push_code(engine, x_reg);
        _push_code(engine, y_reg);
        _push_codes(engine, 4, node->args[0]->o_desc.shape);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        abort();
    }
    return y_reg;
}


static int _gen_relu(struct sf_engine *engine, struct sf_node *node)
{
    int x_reg = engine->reg_map[node->args[0]->index];
    int y_reg = -1;
    if (engine->info[x_reg].ref_cnt == 1) {
        y_reg = x_reg;
    } else {
        y_reg = _alloc_reg_shared(engine, node);
    }
    _push_code(engine, (int)VM_RELU_F32);
    _push_code(engine, x_reg);
    _push_code(engine, y_reg);
    _push_code(engine, sf_tensor_prod(node->o_desc));
    return y_reg;
}


static int _gen_reshape(struct sf_engine *engine, struct sf_node *node)
{
    int x_reg = engine->reg_map[node->args[0]->index];
    int y_reg = -1;
    if (engine->info[x_reg].ref_cnt == 1) {
        y_reg = x_reg;
    } else {
        y_reg = _alloc_reg_shared(engine, node);
    }
    _push_code(engine, (int)VM_COPY_F32);
    _push_code(engine, x_reg);
    _push_code(engine, y_reg);
    _push_code(engine, sf_tensor_prod(node->o_desc));
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


static int _gen_transpose(struct sf_engine *engine, struct sf_node *node)
{
    int x_reg = engine->reg_map[node->args[0]->index];
    int y_reg = _alloc_reg_shared(engine, node);

    int new_shape[SF_MAX_DIMS], new_axes[SF_MAX_DIMS];
    int dims = _simplify_transpose(node, new_shape, new_axes);

    if (2 <= dims && dims <= 4) {
        const int instr[] = {
            (int)VM_TRANSPOSE_2D_F32,
            (int)VM_TRANSPOSE_3D_F32,
            (int)VM_TRANSPOSE_4D_F32,
        };
        _push_code(engine, instr[dims-2]);
        _push_code(engine, x_reg);
        _push_code(engine, y_reg);
        _push_codes(engine, dims, new_shape);
        _push_codes(engine, dims, new_axes);
    }
    else {
        printf("not implemented:\n");
        sf_print_node(stdout, node);
        abort();
    }
    return y_reg;
}


static int _gen_gemm(struct sf_engine *engine, struct sf_node *node)
{
    struct sf_gemm_attrs *attrs = node->attrs;
    int x_reg = engine->reg_map[node->args[0]->index];
    int w_reg = engine->reg_map[node->args[1]->index];
    int y_reg = _alloc_reg_shared(engine, node);
    int b_reg = -1;
    if (node->num_args == 3) {
        b_reg = engine->reg_map[node->args[2]->index];
    }
    _push_code(engine, (int)VM_GEMM_F32);
    _push_code(engine, x_reg);
    _push_code(engine, w_reg);
    _push_code(engine, b_reg);
    _push_code(engine, y_reg);
    _push_code(engine, attrs->trans_a);
    _push_code(engine, attrs->trans_b);
    _push_code(engine, node->o_desc.shape[0]);
    _push_code(engine, node->o_desc.shape[1]);
    _push_code(engine, node->args[1]->o_desc.shape[attrs->trans_b]);
    _push_code(engine, 0);  // relu
    return y_reg;
}


static void _update_io_info(struct sf_engine *engine, struct sf_graph *graph)
{
    struct sf_list *nodes = graph->nodes, *outs = graph->outputs;
    struct sf_node *inputs[256];
    engine->o_cnt = outs->cnt;
    engine->i_cnt = 0;

    for (int i=0; i<nodes->cnt; i++) {
        struct sf_node *node = nodes->buf[i];
        if (node->op_type == OP_INPUT) {
            inputs[engine->i_cnt++] = node;
        }
    }
    engine->i_names = sf_malloc(engine->alloc, engine->i_cnt * sizeof(char*));
    engine->i_regs = sf_malloc(engine->alloc, engine->i_cnt * sizeof(int));
    engine->o_regs = sf_malloc(engine->alloc, engine->o_cnt * sizeof(int));

    for (int i=0; i<engine->i_cnt; i++) {
        engine->i_names[i] = sf_malloc(engine->alloc, SF_MAX_STR_LEN);
        engine->i_regs[i] = engine->reg_map[inputs[i]->index];
        struct sf_input_attrs *attrs = inputs[i]->attrs;
        strcpy(engine->i_names[i], attrs->name);
    }
    for (int i=0; i<outs->cnt; i++) {
        struct sf_node *node = outs->buf[i];
        engine->o_regs[i] = engine->reg_map[node->index];
    }
}


static void _init_regs(struct sf_engine *engine)
{
    void **buf = sf_malloc(engine->alloc, (1 + engine->reg_cnt) * sizeof(void*));
    engine->addr = buf + 1;
    engine->addr[-1] = NULL;

    for (int i=0; i<engine->reg_cnt; i++) {
        engine->addr[i] = sf_malloc(engine->alloc, engine->info[i].size);
        if (engine->info[i].data != NULL) {
            memcpy(engine->addr[i], engine->info[i].data, engine->info[i].size);
        }
    }
}


// generate inference engine from graph
struct sf_engine *sf_engine_from_graph(struct sf_graph *graph)
{
    struct sf_engine *engine = sf_create_engine();
    engine->reg_map = sf_malloc(engine->alloc, graph->nodes->cnt * sizeof(int));

    for (int i=0; i<graph->nodes->cnt; i++) {
        struct sf_node *node = graph->nodes->buf[i];
        int reg = -1;

        switch (node->op_type) {
            case OP_INPUT:      reg = _gen_input(engine, node); break;
            case OP_CONST:      reg = _gen_const(engine, node); break;
            case OP_ADD:        reg = _gen_add(engine, node); break;
            case OP_CONV:       reg = _gen_conv(engine, node); break;
            case OP_MAX_POOL:   reg = _gen_maxpool(engine, node); break;
            case OP_G_AVG_POOL: reg = _gen_gl_avgpool(engine, node); break;
            case OP_RELU:       reg = _gen_relu(engine, node); break;
            case OP_RESHAPE:    reg = _gen_reshape(engine, node); break;
            case OP_TRANSPOSE:  reg = _gen_transpose(engine, node); break;
            case OP_GEMM:       reg = _gen_gemm(engine, node); break;

            default: break;
        }

        for (int j=0; j<node->num_args; j++) {
            int r = engine->reg_map[node->args[j]->index];
            engine->info[r].ref_cnt--;
        }

        if (reg != -1) {
            engine->info[reg].ref_cnt += node->ref_num;
            engine->reg_map[i] = reg;
        } else {
            printf("not implemented:\n");
            sf_print_node(stdout, node);
            abort();
        }
    }
    _push_code(engine, (int)VM_STOP);
    _update_io_info(engine, graph);
    _init_regs(engine);
    return engine;
}

