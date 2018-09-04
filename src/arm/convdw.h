#pragma once

void convdw3x3s1_neon(float *input, int group, int w, int inChannelSize, float *output, int outh, int outw, int outChannelSize, const float* kernel, const float* bias, unsigned num_threads);
void convdw3x3s2_neon(float *input, int group, int w, int inChannelSize, float *output, int outh, int outw, int outChannelSize, const float* kernel, const float* bias, unsigned num_threads);
