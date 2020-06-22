# Jetson NLP
NLP in NVIDIA Jetson Platform

## Installation

## Benchmarks

### Jetson Nano

| Model                | DistillBERT SQuAD | DistillBERT SQuAD |
|----------------------|:-----------------:|:-----------------:|
| Runtime              |   items/seconds   |   tokens/seconds  |
| Pytorch-GPU          |        28.9       |        462        |
| Pytorch-CPU(4 cores) |        3.8        |         61        |
| onnx-CPU (1 core)    |        2.46       |         39        |
| onnx-CUDA            |                   |                   |
| onnx-TRT             |                   |                   | 
