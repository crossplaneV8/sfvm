
#include "test.h"


static double _get_time(void)
{
    struct timezone tz = {0};
	struct timeval time;
	gettimeofday(&time, &tz);
	return (double)time.tv_sec + 0.000001*time.tv_usec;
}


// single engine throughput test
static void _test_fps(struct sf_engine *engine, int batch)
{
    printf("batch size: %d\n", batch);

    for (int i=0; i<10; i++) {
        const int num = 32;
        double t0 = _get_time();
        for (int j=0; j<num; j++) {
            sf_engine_run(engine);
        }
        double t1 = _get_time();
        printf("%.2f FPS\n", (num*batch)/(t1-t0));
    }
}


void test_perf(void)
{
    const char *model_path = "./test/model/resnet18.onnx";
    const char *input_name = "data";
    const int batch = 8;

    struct sf_graph *graph = sf_load_graph_from_onnx(model_path);

    if (graph != NULL) {
        struct sf_tensor_desc in_desc = {SF_FLOAT32, 4, {batch, 3, 224, 224}};
        sf_set_in_desc(graph, input_name, in_desc);
        sf_run_optimization(graph);
        struct sf_engine *engine = sf_engine_from_graph(graph);

        _test_fps(engine, batch);

        // free memory
        sf_discard_engine(engine);
        sf_discard_graph(graph);
    } else {
        printf("error: failed to load model file\n");
        assert(0);
    }
}


