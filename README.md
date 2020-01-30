# Tutorial on deploying Tensorflow using Tensorflow's C API (CPU only)

Tensorflow model is commonly developed using Python. There are however an C API available that allowed developer deploy like an old, classic way of C program that we familiar with. You would think that the *Famous* Tensorflow would have Documentation about how to compile simple C solution with Tensorflow but as up until now (TF2.1) there so little information about that. I'm here to share my finding.

This article will explain how to run common C program using Tensorflow's C API 2.1. The environment that I will use throughout the article is as follow:

- OS : Linux ( Tested and worked on un fresh Ubuntu 19.10/OpenSuse Tumbleweed)
- Latest GCC
- Tensorflow from [Github](https://github.com/tensorflow/tensorflow) (master branch 2.1)

Also, i would to credits Vlad Dovgalecs and his [article](https://medium.com/@vladislavsd/undocumented-tensorflow-c-api-b527c0b4ef6) at Medium as this tutorial largely based and improved from his findings.

# Tutorial structure
 This article will be a bit lenghty. but here is what we will do, step by step:

 1. Get Tensorflow C API from Github to get the C API headers/binaries
 2. Build a simpliest model using Python & Tensorflow and export it to tf model that can be read by C API
 3. Build a simple C code and compile it with `gcc` and run it like a normal execution file.

So here go,

# 1. Getting the Tensorflow C API
As far as i know, there are 2 ways to get those C API header.  
- Downloads from the Tensorflow website (tends not to be up to date binaries) **OR**
- Clone and compile from the source code (Long process, but if things doesn't work, we can debug and look at the API)

So I gonna show how to compile their code and use their binaries.

## Step A: clone their projects
create a folder and clone the project  

``` 
git clone  https://github.com/tensorflow/tensorflow.git
```
  
## Step B: Install the tools that is required for the compilation (Bazel, Numpy)

You would need [Bazel](https://bazel.build/) to compile. Install it on your environment

Ubuntu :   
``` 
sudo apt update && sudo apt install bazel-1.2.1 
```
  
OpenSuse :  
``` 
sudo zypper install bazel 
```

Whichever platform you use, make sure the the bazel version is 1.2.1 as this is what the Tensorflow 2.1 is currently using. Might change in future. 

Next, we would need to install `Numpy` Pyhton's package (Why would we need a Python package to build a C API??). You can install it however you want as long as it can be referenced back during compilation. But I prefer to install it through [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and have seperate virtual environment for the build. Here's how:

Install Miniconda :
```bash 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
sudo chmod 777 Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh 
# follow the default installation direction
```

Create a new environtment + Numpy:
```bash
conda create -n tf-build python=3.7 numpy
```
we use this environtment later in step D.

## Step C: Apply patch to the source code (IMPORTANT!)

Tensorflow 2.1 source code has a bug that will make you build failed. Refer to this [issue](https://github.com/clearlinux/distribution/issues/1151). The fix is to apply patch [here](https://github.com/clearlinux-pkgs/tensorflow/blob/master/Add-grpc-fix-for-gettid.patch). I included a file in this repo that can be use as the patch.
```bash
# copy/download the "p.patch" file from my repo and past at the root of Tensorflow source code.
git apply p.patch
```
In future this might be fixed and not relevant.

## Step D: Compile the code

By referring to the Tensorflow [documentation](https://www.tensorflow.org/install/lang_c) and github [Readme](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README.md). Here's how we compile it. We need to activate out conda env first for it refer to Numpy


```bash
conda activate tf-build # skip this if you already have numpy installed globally

# make sure you're at the root of the Tensorflow source code.
bazel test -c opt tensorflow/tools/lib_package:libtensorflow_test # note that this will take very long to compile
bazel build -c opt tensorflow/tools/lib_package:libtensorflow_test
```
 Let me **WARN** you again. It takes 2 hours to compile on a VM with Ubuntu with 6 Core configuration. My friend with a 2 core laptop basic froze trying to compile this. Here an advice. Run in some server with good CPU/RAM.

copy the file at `bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz` and paste to you're desired folder. untar it like below:
```
tar -C /usr/local -xzf libtensorflow.tar.gz
```
I untar it at my home folder instead as I was just trying it out.

CONGRATULATION!! YOU MADE IT. compiling tensorflow at least.

# 2. Simple model with Python

For this step, we will create a model using `tf.keras.layers` class and saved the model for us to load later use C API. Refer the full code at `model.py` in this repo.

## Step A: Write the model
here is simple model where is has a custom `tf.keras.layers.Model`, with single `dense` layer. Which is initialized with `ones`. Hence the output of this model (from the `def call()`) will produce an output that is similar to the input.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class testModel(tf.keras.Model):
    def __init__(self):
        super(testModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1, kernel_initializer='Ones', activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense1(inputs)

input_data = np.asarray([[10]])
module = testModel()
module._set_inputs(input_data)
print(module(input_data))

# Export the model to a SavedModel
module.save('model', save_format='tf')
```

Eversince Tensorflow 2.0, Eager execution allow us to run a model without drafting the graph and run through `session`. But in order to save the model ( refer to this line `module.save('model', save_format='tf')`), the graph need to be build before it can save. hence we will need to call the model at least once for it to create the graph. Calling `print(module(input_data))` will force it to create the graph.

Next run the code:
```
python model.py
```
You should get an output as below:
```
2020-01-30 11:46:25.400334: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
2020-01-30 11:46:25.421717: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3699495000 Hz
2020-01-30 11:46:25.422615: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561bef5ac2a0 executing computations on platform Host. Devices:
2020-01-30 11:46:25.422655: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
2020-01-30 11:46:25.422744: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
tf.Tensor([[10.]], shape=(1, 1), dtype=float32)
```

You should also see a folder created called `model`.

## Step B: Verified the saved model

When we saved a model, it will create a folder and bunch of files inside it. It basicly store the weights and the graphs of the model. Tensorflow has a tool to dive into this files for us to match the input tensor and the output tensor. It is called `saved_model_cli`. It is a command line tool and comes together when you install Tensorflow.

BUT WAIT!, we haven't install tensorflow !!. so basicly there is two way to get `saved_model_cli`
- Install tensorflow
- Build from source code and looks for `saved_model_cli`

for this i will just install tensorflow in seperate conda environment and call it there, we only need to use it once anyway. so here we go

Install tensorflow in seperate conda environment :

```bash
conda create -n tf python=3.7 tensorflow
```

Activate the environment:
```
conda activate tf
```

