#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

#define BUFFER_IDENTIFIER (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR)

using namespace std;
 
int main()
{
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
     
    char string[MEM_SIZE];
     
    FILE *fp;
    char fileName[] = "./vec_add_kernel.cl";
    char *source_str;
    size_t source_size;
     
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
     
    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
     
    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
     
    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
     
    /* Create Arrays */
    float * src_a_host = (float*) malloc(sizeof(float) * ARR_SIZE);
    float * src_b_host = (float*) malloc(sizeof(float) * ARR_SIZE);
    float * res_host = (float*) malloc(sizeof(float) * ARR_SIZE);

    int i;
    for (i = 0; i < ARR_SIZE; i++) 
        src_a_host[i] = src_b_host[i] = (float) i;
    
    /* Create Memory Buffer */
    const int mem_size = ARR_SIZE * sizeof(float);
    cl_mem src_a_dev = clCreateBuffer(context, BUFFER_IDENTIFIER, mem_size, src_a_host, &ret);
    cl_mem src_b_dev = clCreateBuffer(context, BUFFER_IDENTIFIER, mem_size, src_b_host, &ret);
    cl_mem res_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &ret);
     
    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
            (const size_t *)&source_size, &ret);
     
    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
     
    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "vec_add_gpu", &ret);
     
    /* Set OpenCL Kernel Parameters */
    const int num = ARR_SIZE;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_a_dev);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_b_dev);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_dev);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), &num);
    assert(ret == CL_SUCCESS);
     
    /* Set the work-item */
    const size_t local_work_size = 512;
    const size_t global_work_size = (num / local_work_size + 1) * local_work_size;
    
    /* Execute OpenCL Kernel */
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL); 
     
    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, res_dev, CL_TRUE, 0,
            ARR_SIZE * sizeof(float), res_host, 0, NULL, NULL);
     
    /* Display Result */
    puts("===>Display Results ...");
    for (i = 0; i < ARR_SIZE; i++) {
        if (i != 0 && i % 10 == 0)
            printf("\n");
        printf("%10.3f ", res_host[i]);
    }
    puts("\nDone");
    /* Check result */
    puts("===>Checking results ...");
    for (i = 0; i < ARR_SIZE; i++)
        if (res_host[i] != src_a_host[i] + src_b_host[i])
            printf("[WRONG] %d is wrong\n", i);
    puts("Done");
     
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(src_a_dev);
    ret = clReleaseMemObject(src_b_dev);
    ret = clReleaseMemObject(res_dev);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
     
    free(source_str);
     
    return 0;
}

