
#include "test.h"


static double _get_time(void)
{
    struct timezone tz = {0};
	struct timeval time;
	gettimeofday(&time, &tz);
	return (double)time.tv_sec + 0.000001*time.tv_usec;
}


// single engine throughput test
static void _test_fps_single_engine(struct sf_engine *engine)
{
    printf("single engine:\n");
    for (int i=0; i<32; i++) {
        const int num = 64;
        double t0 = _get_time();
        for (int j=0; j<num; j++) {
            sf_engine_run(engine);
        }
        double t1 = _get_time();
        printf("%.2f FPS\n", num/(t1-t0));
    }
}


// multi engine throughput test
static void _test_fps_multi_engine(struct sf_engine *engine, int threads)
{
    struct sf_engine *engine_clones[threads];
    for (int t=0; t<threads; t++) {
        engine_clones[t] = sf_clone_engine(engine);
    }

    printf("%d parallel engines:\n", threads);
    for (int i=0; i<32; i++) {
        const int num = 32;
        double t0 = _get_time();

        #pragma omp parallel for
        for (int t=0; t<threads; t++) {
            for (int j=0; j<num; j++) {
                sf_engine_run(engine_clones[t]);
            }
        }
        double t1 = _get_time();
        printf("%.2f FPS\n", (threads*num)/(t1-t0));
    }

    for (int t=0; t<threads; t++) {
        sf_discard_engine(engine_clones[t]);
    }
}


void test_perf(void)
{
    const char *model_path = "./test/model/resnet18.onnx";
    const char *input_name = "data";
    struct sf_graph *graph = sf_load_graph_from_onnx(model_path);

    if (graph != NULL) {
        struct sf_tensor_desc in_desc = {SF_FLOAT32, 4, {1, 3, 224, 224}};
        sf_set_in_desc(graph, input_name, in_desc);
        sf_run_optimization(graph);
        struct sf_engine *engine = sf_engine_from_graph(graph);

        int num_threads = omp_get_max_threads();
        _test_fps_single_engine(engine);
        _test_fps_multi_engine(engine, num_threads);

        // free memory
        sf_discard_engine(engine);
        sf_discard_graph(graph);
    } else {
        printf("error: failed to load model file\n");
        assert(0);
    }
}


