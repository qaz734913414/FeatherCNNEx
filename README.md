## Introduction

This project fork from Tencent FeatherCNN(https://github.com/Tencent/FeatherCNN), but we do follow extensions：
1. reduce memory footprint(one pingpang buffer for each branch)
2. fp16(half float) winograd conv (max 30% speedup)
3. TinySGEMM (very cache friendly, very fast)
3. model encrpty (AES CBC encrpty)
4. add some common cv img process arm neon api(such as img resize, submean, bgr2rgb. nv122rgb_roi)
5. TinyDWConv, a depthwise conv no need do extern padding, support asymmetric pad pattern such as tf_pad (speedup more than 50%, enjoy it :) )
6. introduce NCNN(https://github.com/Tencent/ncnn) direct conv op into this framework to get better performance for some special model.

Thanks Tencent FeatherCNN & NCNN team.
## Performance:

<img src="https://raw.githubusercontent.com/tianylijun/FeatherCNNEx/master/benchmark/FeatherCNN-NCNN-TFLITE.png">

## Future Work:
1. Depthwise reconstruct

## Contact Info:
QQ Group: 420089534<br>
Author Email: tianylijun@163.com<br>
