# Jetson NLP
NLP in NVIDIA Jetson Platform

## Installation

   
### 1. Hugginface Transformers Installation

https://gist.github.com/arijitx/c20379394852242a2fa03f76b9ee4e4f<br>
Find a better version here : https://benjcunningham.org/installing-transformers-on-jetson-nano.html

####  1.1 Install sentencepiece

    git clone https://github.com/google/sentencepiece
    cd /path/to/sentencepiece
    mkdir build
    cd build
    cmake ..
    make -j $(nproc)
    sudo make install
    sudo ldconfig -v
    cd .. 
    cd python
    python3 setup.py install

#### 1.2 Install tokenizers

    curl https://sh.rustup.rs -sSf | sh
    rustc --version
    exit
    restart
    pip3 install tokenizers
    
#### 1.3 Install transformers

    Install transformers
    pip3 install transformers

### 2. Onnx Runtime Installation

#### 2.1 Update CMAKE to 3.12 + 

    sudo apt-get install libssl-dev
    Downloading cmake3.14 from ‘https://cmake.org/download/ 32’
    tar -zxvf cmake-3.14.0.tar.gz
    cd cmake-3.14.0
    sudo ./bootstrap //20 mimutes
    sudo make
    sudo make install
    cmake --version //return the version of cmake
    
#### 2.2 Building Onnxruntime 

Jetson TX1/TX2/Nano (ARM64 Builds)<br>
https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#TensorRT<br>

ONNX Runtime v1.2.0 or higher requires TensorRT 7 support, at this moment, the compatible TensorRT and CUDA libraries in JetPack 4.4 is still under developer preview stage. Therefore, we suggest using ONNX Runtime v1.1.2 with JetPack 4.3 which has been validated.
    
    git clone --single-branch --recursive --branch v1.1.2 https://github.com/Microsoft/onnxruntime
 
Indicate CUDA compiler. It's optional, cmake can automatically find the correct cuda.
  
    export CUDACXX="/usr/local/cuda/bin/nvcc"
 
Modify tools/ci_build/build.py

    - "-Donnxruntime_DEV_MODE=" + ("OFF" if args.android else "ON"),
    + "-Donnxruntime_DEV_MODE=" + ("OFF" if args.android else "OFF"),
 
Modify cmake/CMakeLists.txt

    -  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
    +  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_53,code=sm_53") # Jetson TX1/Nano 
    +  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_62,code=sm_62") # Jetson TX2
 
Build onnxruntime with --use_tensorrt flag

    ./build.sh --config Release --update --build --build_wheel --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --tensorrt_home /usr/lib/aarch64-linux-gnu
 
 
See instructions for additional information and tips.
### 3. Convert huggin face model to Onnx
https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb<br>

Update code in src/transformers/convert_graph_to_onnx.py

      export(
          nlp.model,
          model_args,
          f=output,
          input_names=ordered_input_names,
          output_names=output_names,
          dynamic_axes=dynamic_axes,
          do_constant_folding=True,
          opset_version=opset,
      )
      
Dump Onnx Model

    python3 convert_graph_to_onnx.py onnx/dbert_squad.onnx --pipeline question-answering --model  distilbert-base-uncased-distilled-squad --tokenizer distilbert-base-uncased-distilled-squad --framework pt
    
Run benchmark.py

    https://gist.github.com/arijitx/1400d3d4e07fc517d6c5bfea506c2353

## Benchmarks

### Jetson Nano

|                    | Pytorch-GPU | Pytorch-CPU(4 cores) | onnx-CPU (1 core) | onnx-CUDA | onnx-TRT |
|--------------------|-------------|----------------------|-------------------|-----------|----------|
| Distill BERT SQuAD |         462 |                   61 |                39 |           |          |
