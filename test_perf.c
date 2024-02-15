
#include <stdio.h>
#include "sfvm.h"


#include <sys/time.h>
static double _get_time(void)
{
    struct timezone tz = {0};
	struct timeval time;
	gettimeofday( &time, &tz );
	return (double)time.tv_sec + 0.000001*time.tv_usec;
}


// multi thread throughput test
static void _test_fps_multi_thread(struct sf_engine *engine, int threads)
{
    struct sf_engine *engine_clones[threads];
    for (int t=0; t<threads; t++) {
        engine_clones[t] = sf_clone_engine(engine);
    }

    for (int i=0; i<100; i++) {
        const int num = 40;
        double t0 = _get_time();

        #pragma omp parallel for
        for (int t=0; t<threads; t++) {
            for (int j=0; j<num; j++) {
                sf_engine_run(engine_clones[t]);
            }
        }
        double t1 = _get_time();
        printf("total: %.2f FPS\n", (threads*num)/(t1-t0));
    }

    for (int t=0; t<threads; t++) {
        sf_discard_engine(engine_clones[t]);
    }
}


int main(void)
{
    const char *model_path = "./demo/model/resnet18.onnx";
    const char *input_name = "data";
    struct sf_graph *graph = sf_load_graph_from_onnx(model_path);

    if (graph != NULL) {
        // set input tensor dtype and shape
        struct sf_tensor_desc in_desc = {SF_FLOAT32, 4, {1, 3, 224, 224}};
        sf_set_in_desc(graph, input_name, in_desc);

        // inference dtype and shape of other nodes
        sf_graph_infer_tensor_desc(graph);

        printf("graph before optimization:\n\n");
        sf_print_graph(stdout, graph);

        // run graph optimization
        sf_run_optimization(graph);

        printf("graph after optimization:\n\n");
        sf_print_graph(stdout, graph);

        // build inference engine
        struct sf_engine *engine = sf_engine_from_graph(graph);

        printf("instructions:\n\n");
        sf_print_code(stdout, engine);

        printf("multi thread throughput test:\n\n");
        const int num_threads = 8;
        _test_fps_multi_thread(engine, num_threads);

        // free memory
        sf_discard_engine(engine);
        sf_discard_graph(graph);
    } else {
        printf("error: failed to load model file\n");
    }
    getchar();
    return 0;
}


