## Introduction

This project fork from Tencent FeatherCNN(https://github.com/Tencent/FeatherCNN), but we do follow extensions：
1. reduce memory footprint(one pingpang buffer for each branch)
2. fp16(half float) winograd conv (max 30% speedup)
3. fix16(short) & fp16(half float) sgemm support (max 20% speedup)
4. model encrpty (AES CBC encrpty)
5. add some common cv img process arm neon api(such as img resize, submean, bgr2rgb. nv122rgb_roi)
6. introduce NCNN(https://github.com/Tencent/ncnn) direct conv op into this framework to get better performance for some special model.
7. int8 SGEMM feature support(only 1x1 sgemm support, future we will support more conv model).

Thanks Tencent FeatherCNN & NCNN team.

## Future Work:
1. SGEMM reconstruct
2. More int8 conv model support.

## Contact Info:
QQ Group: 420089534<br>
Author Email: tianylijun@163.com<br>
