# SFVM
Simple & Fast Virtual Machine

# 目标
SFVM是一个纯C代码实现的轻量且高效的AI编译器，试图解决当前主流AI编译器代码过重、不易理解和维护、图优化pass性能差、编译时内存占用大、难以实现JIT编译等问题。   

该项目内部实现了优化程度极高的内存池分配器、引用计数器、哈希表、DAG等底层轮子，图优化pass执行效率极高，经测试ResNet-18转换出的IR图跑单个图变换pass耗时仅为μs级，比当前主流AI编译器快几个数量级。
有望在普通CPU上做到从原始模型到VM指令的JIT编译，目前世界上应该还没有先例。

# 进度
前端已经适配了ONNX。   
中间表示(DAG)已经设计完成，图变换pass机制已实现，测试改进中。   
后端代码生成计划支持CPU和GPU两套VM，待开发。（预计需要半年时间）   

# 运行demo
```bash
cd sfvm
gcc -O2 -Isrc -Isrc/onnx src/base/*.c src/graph/*.c src/optimizer/*.c src/onnx/*.c demo.c -o demo -s
./demo
```
也可用CodeBlocks IDE打开工程文件sfvm.cbp编译运行

