## Introduction

This project fork from Tencent FeatherCNN(https://github.com/Tencent/FeatherCNN), but we do Follow extensions：
1. reduce memory footprint(one pingpang buffer for each branch)
2. fp16(half float) winograd conv (max 30% speedup)
3. fix16(short) & fp16(half float) sgemm support (max 20% speedup)
4. model encrpty (AES CBC ensrpty)
5. Add some common cv img process arm neon api(such as img resize, submean, bgr2rgb. nv122rgb_roi)
6. Introduce NCNN(https://github.com/Tencent/ncnn) direct conv op into this framework to get better performance for some special model.
7. Int8 SGEMM feature support(not test yet).

Thanks Tencent FeatherCNN & NCNN team.

## Future Work:
We are now trying int8 SGEMM, hoping for good performance, :)



## Contact Info:
Hoping for your contribution
Author Email: tianylijun@163.com
QQ Group: 420089534
Wechat: iuaufnael

## BenchMark
<img src="https://raw.githubusercontent.com/tianylijun/FeatherCNNEx/master/benchmark/FeatherCNNExVSNCNN.jpeg">
<img src="https://raw.githubusercontent.com/tianylijun/FeatherCNNEx/master/benchmark/MeiZu_Benchmark.jpeg">
