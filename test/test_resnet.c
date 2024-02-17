
#include "test.h"


// read data from binary file
static size_t _load_data(const char *path, void *buf, size_t size)
{
    size_t num = 0;
    FILE *f = fopen(path, "rb");
    if (f != NULL) {
        num = fread(buf, 1, size, f);
        fclose(f);
    }
    return num;
}


void test_resnet(void)
{
    printf("resnet accuracy test:\n");

    const char *i_data_file = "./test/data/in_10x3x224x224_f32.bin";
    const char *o_data_file = "./test/data/out_10x1000_f32.bin";
    const char *model_path = "./test/model/resnet18.onnx";
    const char *input_name = "data";

    struct sf_graph *graph = sf_load_graph_from_onnx(model_path);

    if (graph != NULL) {
        struct sf_tensor_desc in_desc = {SF_FLOAT32, 4, {10,3,224,224}};
        sf_set_in_desc(graph, input_name, in_desc);
        sf_run_optimization(graph);

        struct sf_engine *engine = sf_engine_from_graph(graph);
        float *i_data = sf_get_input_addr(engine, input_name);
        float *o_data = sf_get_output_addr(engine, 0);
        static float o_data_gt[10*1000];

        _load_data(i_data_file, i_data, 10*3*224*224*sizeof(float));
        _load_data(o_data_file, o_data_gt, 10*1000*sizeof(float));
        sf_engine_run(engine);

        double max_err = _max_error(o_data, o_data_gt, 10*1000);
        double rms_err = _rms_error(o_data, o_data_gt, 10*1000);
        printf("max err = %f\n", max_err);
        printf("rms err = %f\n", rms_err);
        assert(max_err < 1e-2);
        assert(rms_err < 1e-3);

        // free memory
        sf_discard_engine(engine);
        sf_discard_graph(graph);
    } else {
        printf("error: failed to load model file\n");
        assert(0);
    }
    printf("passed\n\n");
}

