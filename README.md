## Introduction

This project fork from Tencent FeatherCNN(https://github.com/Tencent/FeatherCNN), but we do some extensions such as reduce memory footprint, fp16(half float) winograd conv (max 30% speedup), fix16(short) sgemm support (max 20% speedup), and add some common cv img process arm neon api such as img resize, submean, bgr2rgb. We also introduce NCNN direct conv op into this framework to get better performance for some special net model, and fix some bugs.

Thanks Tencent FeatherCNN & NCNN team.

Contact Info:

Author Email: tianylijun@163.com
QQ Group: 420089534

## BenchMark
<img src="https://raw.githubusercontent.com/tianylijun/FeatherCNNEx/master/benchmark/MeiZu_Benchmark.jpeg">
