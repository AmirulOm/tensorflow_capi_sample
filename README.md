# Tutorial on deploying Tensorflow using Tensorflow's C API (CPU only)

Here is use case that I believe some Non Data Engineer/Data Scientist is facing. 

**How do I deliver an Tensorflow model that I trained in Python but deploy it in pure C/C++ code on the client side without setup python environment at their side and on top of that all files has to be in binaries??**

The answer for that is to use the Tensorflow C or C++ API. In this article we only look how to use the C API (not the C++/tensorflowlite) that runs only in CPU. 

You would think that the *famous* Tensorflow would have documentation about how to compile simple C solution with Tensorflow but as up until now (TF2.1) there so little to none information about that. I'm here to share my finding.

This article will explain how to run common C program using Tensorflow's C API 2.1. The environment that I will use throughout the article is as follow:

- OS : Linux ( Tested and worked on un fresh Ubuntu 19.10/OpenSuse Tumbleweed)
- Latest GCC
- Tensorflow from [Github](https://github.com/tensorflow/tensorflow) (master branch 2.1)
- No GPU

Also, i would to credits Vlad Dovgalecs and his [article](https://medium.com/@vladislavsd/undocumented-tensorflow-c-api-b527c0b4ef6) at Medium as this tutorial largely based and improved from his findings.

# Tutorial structure
 This article will be a bit lenghty. but here is what we will do, step by step:

 1. Clone Tensorflow source code and compile to get the C API headers/binaries
 2. Build a simpliest model using Python & Tensorflow and export it to tf model that can be read by C API
 3. Build a simple C code and compile it with `gcc` and run it like a normal execution file.

So here we go,

# 1. Getting the Tensorflow C API
As far as i know, there are 2 ways to get those C API header.  
- Download the precompiled Tensorflow C API from website (tends not to be up to date binaries) **OR**
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

Create a new environtment + Numpy named tf-build:
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
 Let me **WARN** you again. It takes 2 hours to compile on a VM with Ubuntu with 6 Core configuration. My friend with a 2 core laptop basicly frozed trying to compile this. Here an advice. Run in some server with good CPU/RAM.

copy the file at `bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz` and paste to you're desired folder. untar it like below:
```
tar -C /usr/local -xzf libtensorflow.tar.gz
```
I untar it at my home folder instead of at `/usr/local` as I was just trying it out.

CONGRATULATION!! YOU MADE IT. compiling tensorflow at least.

# 2. Simple model with Python

For this step, we will create a model using `tf.keras.layers` class and saved the model for us to load later use C API. Refer the full code at `model.py` in the [repo](https://github.com/AmirulOm/tensorflow_capi_sample/blob/master/model.py).

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

You should also see a folder created called `model` created.

## Step B: Verified the saved model

When we saved a model, it will create a folder and bunch of files inside it. It's basicly store the weights and the graphs of the model. Tensorflow has a tool to dive into this files for us to match the input tensor and the output tensor. It is called `saved_model_cli`. It is a command line tool and comes together when you install Tensorflow.

BUT WAIT!, we haven't install tensorflow !!. so basicly there is two way to get `saved_model_cli`
- Install tensorflow
- Build from source code and looks for `saved_model_cli`

for this I will just install tensorflow in seperate conda environment and call it there, we only need to use it once anyway. so here we go

Install tensorflow in seperate conda environment :

```bash
conda create -n tf python=3.7 tensorflow
```

Activate the environment:
```
conda activate tf
```

by now you should be able to call `saved_model_cli` through command line.

We would need to extract the graph name for the input tensor and output tensor and use that info during calling C API later on. Here's how:

```bash
saved_model_cli show --dir <path_to_saved_model_folder> 
```

running this and replaced the appropriate path, you should get an output like below:
```
The given SavedModel contains the following tag-sets:
serve
```
use this tag-set to further drill into the tensor graph, here's how:
```
saved_model_cli show --dir <path_to_saved_model_folder> --tag_set serve 
```
and you should get an output like below:
```
The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
SignatureDef key: "__saved_model_init_op"
SignatureDef key: "serving_default"
```

using `serving_default` signature key into command to print out the tensor node:
```
saved_model_cli show --dir <path_to_saved_model_folder> --tag_set serve --signature_def serving_default
```

and you should get an output like below:
```
The given SavedModel SignatureDef contains the following input(s):
  inputs['input_1'] tensor_info:
      dtype: DT_INT64
      shape: (-1, 1)
      name: serving_default_input_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```
here we would need the name `serving_default_input_1` and `StatefulPartitionedCall` later to be use in the C API.

# 3. Building C/C++ code

Third part is to write the C code that use the Tensorflow C API and import the Python saved model. The full code can be refer at [here](https://github.com/AmirulOm/tensorflow_capi_sample/blob/master/main.c). 

There is no C API proper documentation, so if something went wrong, it's best to look back at ther C header in the [source code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h) (You can also debug using GDB and step by step learn how the C header works)


## Step A: Write C code
On empty file, import the tensorflow C API as follow:

```cpp
#include <stdlib.h>
#include <stdio.h>
#include "tensorflow/c/c_api.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main()
{
}
```
Note that you have `NoOpDeallocator` void function declared, we will use it later

Next need to load the savedmodel and the session using `TF_LoadSessionFromSavedModel` API. 

```cpp

    //********* Read model
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();

    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = "model/"; // Path of the model
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;

    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }
```

Next we grab the tensor node from the graph by their name. Remember earlier we search for tensor name using `saved_model_cli`?. here where we use it back when we call `TF_GraphOperationByName()`. In this example, `serving_default_input_1` is our input tensor and `StatefulPartitionedCall` is out output tensor.

```cpp
    //****** Get input tensor
    int NumInputs = 1;
    TF_Output* Input = malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
	    printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    
    Input[0] = t0;
    
    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output* Output = malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else	
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
    
    Output[0] = t2;
```

Next we will need to allocate the new tensor locally using `TF_NewTensor`, set the input value  and later we will pass to session run. *NOTE that `ndata` is total byte size of your data, not lenght of the array*

Here we set the input tensor with value of 20. and we should see the output value as 20 as well.

```cpp
    //********* Allocate data for inputs & outputs
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = malloc(sizeof(TF_Tensor*)*NumOutputs);

    int ndims = 2;
    int64_t dims[] = {1,1};
    int64_t data[] = {20};
    int ndata = sizeof(int64_t); // This is tricky, it number of bytes not number of element

    TF_Tensor* int_tensor = TF_NewTensor(TF_INT64, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL)
    {
        printf("TF_NewTensor is OK\n");
    }
    else
	printf("ERROR: Failed TF_NewTensor\n");
    
    InputValues[0] = int_tensor;
```

Next we can run the model by invoking `TF_SessionRun` API. Here's how:

```cpp    
    // //Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);

    if(TF_GetCode(Status) == TF_OK)
    {
        printf("Session is OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }

    // //Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);
```
Lastly, we want get back the output value from the output tensor using `TF_TensorData` that extract data from the tensor object. Since we know the size of the output which is 1, i can directly print it. Else use `TF_GraphGetTensorNumDims` or other API that is available in `c_api.h` or `tf_tensor.h` 

```cpp

    void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = buff;
    printf("Result Tensor :\n");
    printf("%f\n",offsets[0]);
    return 0;
```

## Step B: Compile the code

Compile it as below:

```bash
gcc -I<path_of_tensorflow_api>/include/ -L<path_of_tensorflow_api>/lib main.c -ltensorflow -o main.out
```

## Step C: Run it

Before you run it. You'll need to make sure the C library is exported in your environment

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_of_tensorflow_api>/lib 
```

RUN IT

```
./main.out
```

You should get an output like below. Notice that the output value is 20 like out input. you can change the model and initiliaze the kernel with weight of value 2 and see if it reflected to other value.

```
2020-01-31 09:47:48.842680: I tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: model/
2020-01-31 09:47:48.844252: I tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2020-01-31 09:47:48.844295: I tensorflow/cc/saved_model/loader.cc:264] Reading SavedModel debug info (if present) from: model/
2020-01-31 09:47:48.844385: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
2020-01-31 09:47:48.859883: I tensorflow/cc/saved_model/loader.cc:203] Restoring SavedModel bundle.
2020-01-31 09:47:48.908997: I tensorflow/cc/saved_model/loader.cc:152] Running initialization op on SavedModel bundle at path: model/
2020-01-31 09:47:48.923127: I tensorflow/cc/saved_model/loader.cc:333] SavedModel load for tags { serve }; Status: success: OK. Took 80457 microseconds.
TF_LoadSessionFromSavedModel OK
TF_GraphOperationByName serving_default_input_1 is OK
TF_GraphOperationByName StatefulPartitionedCall is OK
TF_NewTensor is OK
Session is OK
Result Tensor :
20.000000
```

END

