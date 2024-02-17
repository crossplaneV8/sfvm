
#include "test.h"


static struct sf_graph *_make_conv_graph(struct sf_tensor_desc x_desc,
                                         struct sf_tensor_desc w_desc)
{
    struct sf_graph *graph = sf_create_graph();
    struct sf_node *x = sf_create_input_node(graph, "x", x_desc);
    struct sf_node *w = sf_create_input_node(graph, "w", w_desc);
    struct sf_node *y = sf_create_conv_node(graph, x, w, NULL, "NCHW", "OIHW",
                                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
    sf_set_graph_output(graph, y);
    return graph;
}


void test_conv(void)
{
    printf("tiny convolution test:\n");

    const float x_data[5*5] = {
        1, 2, 3, 0, 1,
        1, 1, 2, 2, 2,
        0, 1, 0, 1, 0,
        2, 2, 1, 0, 3,
        1, 2, 4, 5, 3,
    };
    const float w_data[3*3] = {
        1, 0, 0,
        1, 2, 1,
        0, 1, 1,
    };
    const float y_data[5*5] = {
        6, 11, 12,  8,  4,
        4,  7, 10, 12,  6,
        5,  6,  4,  7,  6,
        9, 13, 14, 12, 10,
        4, 11, 17, 18, 11,
    };
    struct sf_tensor_desc x_desc = {SF_FLOAT32, 4, {1, 1, 5, 5}};
    struct sf_tensor_desc w_desc = {SF_FLOAT32, 4, {1, 1, 3, 3}};
    struct sf_graph *graph = _make_conv_graph(x_desc, w_desc);

    sf_run_optimization(graph);
    struct sf_engine *engine = sf_engine_from_graph(graph);

    float *x_buf = sf_get_input_addr(engine, "x");
    float *w_buf = sf_get_input_addr(engine, "w");
    float *y_buf = sf_get_output_addr(engine, 0);
    memcpy(x_buf, x_data, sizeof(x_data));
    memcpy(w_buf, w_data, sizeof(w_data));

    sf_engine_run(engine);

    double max_err = _max_error(y_buf, y_data, 5*5);
    double rms_err = _rms_error(y_buf, y_data, 5*5);
    printf("max err = %f\n", max_err);
    printf("rms err = %f\n", rms_err);
    assert(max_err < 1e-2);
    assert(rms_err < 1e-3);

    // free memory
    sf_discard_engine(engine);
    sf_discard_graph(graph);

    printf("passed\n\n");
}


