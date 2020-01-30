#include <stdlib.h>
#include <stdio.h>
#include "tensorflow/c/c_api.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main()
{
    //********* Read model
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();

    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = "/home/amirul/project/tf/model/";
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;

    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
if(TF_GetCode(Status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel is ok\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }

    //****** Prepare input
    //TODO : need to use saved_model_cli to read saved_model arch
    int NumInputs = 1;
    TF_Output* Input = malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
    // TF_Output t1 = {TF_GraphOperationByName(Graph, “<node1>”), <idx1>};
    if(t0.oper == NULL)
    {
        printf("ERROR\n");
    }
    Input[0] = t0;
    // Input[1] = t1;
//     int n_ops = 100;
//   for (int i=0; i<n_ops; i++)
//   {
//     size_t pos = i;
//     printf("%s\n", TF_OperationName(TF_GraphNextOperation(Graph, &pos)) );//<< "\n";
//   }
    //********* Define Output
    int NumOutputs = 1;
    TF_Output* Output = malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if(t2.oper == NULL)
    {
        printf("ERROR\n");
    }
    Output[0] = t2;

    //Provide data for inputs & outputs
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = malloc(sizeof(TF_Tensor*)*NumOutputs);


    int ndims = 2;
    int64_t* dims = malloc(sizeof(int64_t)*2);
    int ndata = sizeof(int64_t);
    int64_t* data = malloc(sizeof(int64_t));
    dims[0] = 1;
    dims[1] = 1;
    data[0] = 20;
 printf("Input [0][0] : %li\n",data[0]);
    TF_Tensor* int_tensor = TF_NewTensor(TF_INT64, dims, ndims, data, ndata, &NoOpDeallocator, 0);

    //int *mydata = malloc(sizeof(int)*1);
    // mydata[0] = 10;
    // int64_t dims[] = {1, 1};
    // /* create tensors with data here */
    // TF_Tensor* tensor0 = TF_NewTensor(TF_INT32,dims,2, mydata,sizeof(int), &NoOpDeallocator, NULL);
    // tensor1 = ...;
// printf("bb\n");


    InputValues[0] = int_tensor;
    // InputValues[1] = tensor1;

    // //Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);
// printf("loll2\n");
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("Session is ok\n");
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


    void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = buff;
    printf("Result Tensor of size (1,5):\n");
    for (size_t i = 0; i < 5; ++i)
    {
        printf("%f\n",offsets[i]);
    }


    // // Create a 1-dim tensor holding an integer

    // int64_t dims = NULL;
    // int ndims = 0;

    // //Create an array of strings
    // const char* sarr;
    // int nstr;

    // TF_Tensor* str_tensor = TF_NewTensor(TF_String, nstr, ndims, base, bsize, free_array, base);

    // size_t tsize = 0;
    // for (int i = 0; i < nstr; i++) {
    //     tsize += TF_StringEncodedSize(strlen(sarr[i])) + sizeof(TF_DataTypeSize(TF_UINT64));
    // }

    // char* base = malloc(sizeof(char)*tsize;
    // char* start = sizeof(TF_DataTypeSize(TF_UINT64))*nstr + base;
    // char* dest = start;
    // size_t dest_len = tsize - (size_t)(start - base);
    // uint64_t* offsets = (uint64_t*)(base);

    // for (int i = 0; i < nstr; i++) {
    // *offsets = (dest - start);
    // offsets++;
    // size_t a = TF_StringEncode(sarr[i], strlen(sarr[i]), dest, dest_len, Status);
    // dest += a;
    // dest_len -= a;
    // }

    // int64_t dimvec[] = {1, nstr};
    // size_t ndims = 2;

    // TF_Tensor* tarr = TF_NewTensor(TF_STRING, dimvec, ndims, base, tsize, free_array, base);

    // //Note: The Python equivalent of the created tensor (with some sample data):
    // // s = [[“some”, “interesting”, ”data”, “here”]]

    // // unpack a tensor of strings

    // // First, prepare the array of strings:
    // char** out[];
    // size_t nout;

    // // Get the Tensor of strings from the array of output tensors:
    // TF_Tensor* tout = OutputValues[0]; // assuming we want the first one

    // // Get shape information of the tensor:
    // nout = (size_t)TF_Dim(tout, 1)]; // assuming the number of strings is 2nd dim

    // //Prepare arrays for encoded data and offsets:
    // void* buff = TF_TensorData(tout);
    // int64_t* offsets = buff;

    // //Prepare utility pointers:
    // char* data = buff + nout * sizeof(int64_t);
    // char* buff_end = buff + TF_TensorByteSize(tout);

    // //Allocate pointer to arrays of strings:
    // *out = calloc(nout, sizeof(char*));

    // //Decode every string and copy it into the array of strings:
    // for (int i = 0; i < nout; i++) {
    //     char* start = buff + offsets[i];
    //     const char* dest;
    //     size_t n;
    //     TF_StringDecode(start, buff_end — start, &dest, &n, Status);
    //     if (TF_GetCode(Status) == TF_OK)
    //     {
    //         (*out)[i] = malloc(n + 1);
    //         memcpy((*out)[i], dest, n);
    //         *((*out)[i] + n) = ‘\0’;
    //     }
    // }
}