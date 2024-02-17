
#include <stdio.h>
#include "sfvm.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


// read image data from file (src: u8[H,W,C] ==> dst: f32[C,H,W])
static void _load_image(const char *path, float *img, int ci, int hi, int wi)
{
    int w = 0, h = 0, n = 0;
    uint8_t *data = stbi_load(path, &w, &h, &n, ci);

    if (data != NULL) {
        memset(img, 0, ci * hi * wi * sizeof(float));
        int _h = h < hi ? h : hi;
        int _w = w < wi ? w : wi;

        for (int z=0; z<ci; z++) {
            for (int y=0; y<_h; y++) {
                for (int x=0; x<_w; x++) {
                    float val = (float)data[y*w*ci + x*ci + z];
                    img[z*hi*wi + y*wi + x] = val / 255.0;
                }
            }
        }
        stbi_image_free(data);
    }
}


// get index of maximum element
static int _argmax_f32(const float *data, int num)
{
    float top = data[0];
    int index = 0;
    for (int i=1; i<num; i++) {
        if (data[i] > top) {
            top = data[i];
            index = i;
        }
    }
    return index;
}


// run image classification test
static void _test_image_classification(struct sf_engine *engine, const char *input_name)
{
    #include "labels.h"
    const char *test_images[] = {
        "./test/image/test_img_0.png",
        "./test/image/test_img_1.png",
        "./test/image/test_img_2.png",
        "./test/image/test_img_3.png",
        "./test/image/test_img_4.png",
        "./test/image/test_img_5.png",
        "./test/image/test_img_6.png",
        "./test/image/test_img_7.png",
        "./test/image/test_img_8.png",
        "./test/image/test_img_9.png",
    };
    float *i_data = sf_get_input_addr(engine, input_name);
    float *o_data = sf_get_output_addr(engine, 0);

    for (int i=0; i<sizeof(test_images)/sizeof(void*); i++) {
        _load_image(test_images[i], i_data, 3, 224, 224);
        sf_engine_run(engine);
        int id = _argmax_f32(o_data, 1000);
        printf("%s ==> %s\n", test_images[i], imagenet_labels[id]);
    }
}


int main(void)
{
    const char *model_path = "./test/model/resnet18.onnx";
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

        printf("inference engine:\n\n");
        sf_print_engine(stdout, engine);

        printf("image classification test:\n\n");
        _test_image_classification(engine, input_name);

        // free memory
        sf_discard_engine(engine);
        sf_discard_graph(graph);
    } else {
        printf("error: failed to load model file\n");
    }
    getchar();
    return 0;
}

