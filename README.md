# SFVM (Simple Fast Virtual Machine)

SFVM是一个纯C代码实现的轻量且高效的AI编译器， 试图解决当前主流AI编译器代码过重、不易理解和维护、编译慢等问题。   


# 开发状态
前端图转换已适配ONNX   
中间表示(graph)支持的优化pass包括:   
```c
// 消除不可达节点
remove_unreachable_nodes()

// 消除恒等映射节点 e.g.(add 0, mul 1)
remove_identity_nodes()

// 将squeeze, flatten, transpose替换为reshape
convert_to_reshape()

// 将batch-norm替换为mul, add
batchnorm_to_mul_add()

// 融合conv, mul, add, relu
fuse_conv_mul_add_relu()

// 将conv, pooling统一为NHWC layout
convert_layout_NHWC()

// 将权重pack成NK16格式 (消除GEMM内的矩阵pack开销)
pack_conv_weight()

// transpose节点后移
swap_transpose()

// 合并相邻的reshape或transpose
merge_redundant_nodes()

// 常量折叠
fold_constant()
```
后端实现了x86 CPU版的VM，已适配ResNet50所需的float32算子，关键算子采用AVX256 SIMD极致优化。   
特别是convolution算子，采用了implicit GEMM算法，相比普通实现性能提升显著。   


# 运行demo
```bash
cd sfvm
gcc -O2 -mavx2 -mfma -Isrc -Isrc/onnx src/base/*.c src/graph/*.c src/onnx/*.c src/optimizer/*.c src/backend/*.c src/compute_lib/*.c demo/*.c -o demo -s
./demo
```
也可用CodeBlocks IDE打开工程文件sfvm.cbp编译运行。   


# 性能测试
```bash
cd sfvm
gcc -O2 -fopenmp -mavx2 -mfma -Isrc -Isrc/onnx src/base/*.c src/graph/*.c src/onnx/*.c src/optimizer/*.c src/backend/*.c src/compute_lib/*.c test_perf.c -o test -lgomp -s
./test
```
CPU: i7-13700F
| model     | precision | threads | FPS  | TFLOPS |
| --------- | --------- | ------- | ---- | ------ |
| ResNet-18 | float32   | 8       | 202  | 0.74   |
| ResNet-50 | float32   | 8       | 79   | 0.62   |

