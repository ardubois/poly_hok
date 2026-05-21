#include "erl_nif.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <dlfcn.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

CUcontext context = NULL;
void init_cuda(ErlNifEnv *env)
{
  if (context == NULL)
  {
    CUresult err;
    // CUdevice   device;
    int device = 0;
    // printf("aqui!\n");
    cuInit(0);

    // err = cuDeviceGet(&device, 0);
    // if(err != CUDA_SUCCESS)
    //{ char message[200];
    //  const char *error;
    //  cuGetErrorString(err, &error);
    //  strcpy(message,"Error get device (init_cuda): ");
    //  strcat(message, error);
    //  enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    //}

    err = cuCtxCreate(&context, 0, device);
    // printf("Context created: %p",context);
    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error INIT CUDA: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  }
  else
  {
    cuCtxSetCurrent(context);
  }
}

#define MX_ROWS(matrix) (((uint32_t *)matrix)[0])
#define MX_COLS(matrix) (((uint32_t *)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t *)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t *)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t *)matrix)[0]) * (((uint32_t *)matrix)[1]) + 2)

ErlNifResourceType *KERNEL_TYPE;
ErlNifResourceType *ARRAY_TYPE;
ErlNifResourceType *PINNED_ARRAY;
ErlNifResourceType *PTX_TYPE;

void dev_array_destructor(ErlNifEnv *env, void *res)
{
  CUdeviceptr *dev_array = (CUdeviceptr *)res;
  cuMemFree(*dev_array);
}

void dev_pinned_array_destructor(ErlNifEnv *env, void *res)
{
  float **dev_array = (float **)res;
  cudaFreeHost(*dev_array);
}

static int
load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info)
{

  KERNEL_TYPE =
      enif_open_resource_type(env, NULL, "kernel", NULL, ERL_NIF_RT_CREATE, NULL);
  ARRAY_TYPE =
      enif_open_resource_type(env, NULL, "gpu_ref", dev_array_destructor, ERL_NIF_RT_CREATE, NULL);
  PINNED_ARRAY =
      enif_open_resource_type(env, NULL, "pinned_array", dev_pinned_array_destructor, ERL_NIF_RT_CREATE, NULL);
  PTX_TYPE =
      enif_open_resource_type(env, NULL, "ptx", NULL, ERL_NIF_RT_CREATE, NULL);
  return 0;
}

/////////////////////////////
////////
///////////////////////
//////////////////////////////
//////////////// BEGIN CUDA DRIVER API CODE
/////////////////////////////
/////////////////////////////
/////////////////////////////

///////////////////////
/////////
//////// BEGIN JIT COMPILATION KERNEL
////////
///////////////////

void fail_cuda(ErlNifEnv *env, CUresult result, const char *obs)
{

  char message[1000];
  const char *error;
  strcpy(message, "Error  CUDA ");
  strcat(message, obs);
  strcat(message, ": ");
  cuGetErrorString(result, &error);
  strcat(message, error);
  enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
}

void fail_nvrtc(ErlNifEnv *env, nvrtcResult result, const char *obs)
{

  char message[1000];

  strcpy(message, "Error  NVRTC ");
  strcat(message, obs);
  strcat(message, ": ");
  strcat(message, nvrtcGetErrorString(result));
  enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
}

ERL_NIF_TERM compile_to_ptx(ErlNifEnv *env, char* program_source)
{
  nvrtcResult rv;

  // create nvrtc program
  nvrtcProgram prog;
  rv = nvrtcCreateProgram(
      &prog,
      program_source,
      nullptr,
      0,
      nullptr,
      nullptr);
  if (rv != NVRTC_SUCCESS)
    fail_nvrtc(env, rv, "nvrtcCreateProgram");

  int size_options = 10;
  const char *options[10] = {
      "--include-path=/lib/erlang/usr/include/",
      "--include-path=/usr/include/",
      "--include-path=/usr/lib/",
      "--include-path=/usr/include/x86_64-linux-gnu/",
      "--include-path=/usr/include/c++/11",
      "--include-path=/usr/include/x86_64-linux-gnu/c++/11",
      "--include-path=/usr/include/c++/11/backward",
      "--include-path=/usr/lib/gcc/x86_64-linux-gnu/11/include",
      "--include-path=/usr/include/i386-linux-gnu/",
      "--include-path=/usr/local/include"};

  rv = nvrtcCompileProgram(prog, size_options, options);
  if (rv != NVRTC_SUCCESS)
  {
    nvrtcResult erro_g = rv;
    size_t log_size;
    rv = nvrtcGetProgramLogSize(prog, &log_size);
    if(rv != NVRTC_SUCCESS)
      fail_nvrtc(env,rv,"nvrtcGetProgramLogSize");
    //auto log = std::make_unique<char[]>(log_size);
    char log[log_size];
    rv = nvrtcGetProgramLog(prog, log);
    if(rv != NVRTC_SUCCESS)
      fail_nvrtc(env,rv,"nvrtcGetProgramLog");
    assert(log[log_size - 1] == '\0');

    printf("Compilation error; log: %s\n", log);

    fail_nvrtc(env,erro_g,"nvrtcCompileProgram");
    //return enif_make_int(env, 0);
  }
  // get ptx code
  size_t ptx_size;
  rv = nvrtcGetPTXSize(prog, &ptx_size);
  if(rv != NVRTC_SUCCESS)
    fail_nvrtc(env,rv,"nvrtcGetPTXSize");
  char *ptx_source = (char *)enif_alloc_resource(PTX_TYPE, ptx_size * sizeof(char));
  assert(ptx_source != NULL);
  nvrtcGetPTX(prog, ptx_source);

  if(rv != NVRTC_SUCCESS) fail_nvrtc(env, rv, "nvrtcGetPTX");
  assert(ptx_source[ptx_size - 1] == '\0');

  nvrtcDestroyProgram(&prog);

  ERL_NIF_TERM ptxResource = enif_make_resource(env, ptx_source);

  enif_release_resource(ptx_source);

  return ptxResource;
}

static ERL_NIF_TERM jit_compile_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ERL_NIF_TERM e_name = argv[0];
  unsigned int size_name;
  if (!enif_get_list_length(env,e_name,&size_name)) {
    return enif_make_badarg(env);
  }

  char kernel_name[size_name + 1];

  enif_get_string(env, e_name, kernel_name, size_name + 1, ERL_NIF_LATIN1);

  // get code
  ERL_NIF_TERM e_code = argv[1];
  unsigned int size_code;
  if (!enif_get_list_length(env,e_code,&size_code)) {
    return enif_make_badarg(env);
  }

  char code[size_code+1];
  enif_get_string(env,e_code,code,size_code+1,ERL_NIF_LATIN1);

  return compile_to_ptx(env, code);
}

static ERL_NIF_TERM jit_launch_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
	char *ptx = NULL;

	if (!enif_get_resource(env, argv[0], PTX_TYPE, (void **)&ptx)) {
		return enif_make_badarg(env);
	}

	ERL_NIF_TERM e_name = argv[1];
	unsigned int size_name;
	if (!enif_get_list_length(env,e_name,&size_name)) {
		return enif_make_badarg(env);
	}

	/* @todo same issue */
	char kernel_name[size_name+1];
	enif_get_string(env, e_name, kernel_name, size_name + 1, ERL_NIF_LATIN1);

	const ERL_NIF_TERM *tuple_blocks;
	const ERL_NIF_TERM *tuple_threads;
	int arity;

	CUmodule   module;
	CUfunction function;
	CUresult   err;

	if (!enif_get_tuple(env, argv[2], &arity, &tuple_blocks)) {
		printf ("spawn: blocks argument is not a tuple");
	}

	if (!enif_get_tuple(env, argv[3], &arity, &tuple_threads)) {
		printf ("spawn:threads argument is not a tuple");
	}
	int64_t b1,b2,b3,t1,t2,t3;

	enif_get_int64(env,tuple_blocks[0],&b1);
	enif_get_int64(env,tuple_blocks[1],&b2);
	enif_get_int64(env,tuple_blocks[2],&b3);
	enif_get_int64(env,tuple_threads[0],&t1);
	enif_get_int64(env,tuple_threads[1],&t2);
	enif_get_int64(env,tuple_threads[2],&t3);


	int64_t size_args;
	if (!enif_get_int64(env, argv[4], &size_args)) {
		return enif_make_badarg(env);
	}

	CUdeviceptr arrays[size_args];
	float floats[size_args];
	int ints[size_args];
	double doubles[size_args];
	int64_t arrays_ptr=0;
	int64_t floats_ptr=0;
	int64_t doubles_ptr=0;
	int64_t ints_ptr=0;

	init_cuda(env);
	err = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	if (err != CUDA_SUCCESS) fail_cuda(env,err,"cuModuleLoadData jit compile");

	// use compiled ptx
	err = cuModuleGetFunction(&function, module, kernel_name);
	if (err != CUDA_SUCCESS) fail_cuda(env,err,"cuModuleGetFunction jit compile");

	void *args[size_args];

	ERL_NIF_TERM list_types;
	ERL_NIF_TERM head_types;
	ERL_NIF_TERM tail_types;

	ERL_NIF_TERM list_args;
	ERL_NIF_TERM head_args;
	ERL_NIF_TERM tail_args;

	list_types = argv[5];
	list_args = argv[6];

	for(int i=0; i<size_args; i++) {
		char type_name[1024];
		unsigned int size_type;
		if (!enif_get_list_cell(env,list_types,&head_types,&tail_types)) {
			printf("erro get list cell\n");
			return enif_make_badarg(env);
		}
		if (!enif_get_list_length(env,head_types,&size_type)) {
			printf("erro get list length\n");
			return enif_make_badarg(env);
		}

		enif_get_string(env,head_types,type_name,size_type+1,ERL_NIF_LATIN1);

		if (!enif_get_list_cell(env,list_args,&head_args,&tail_args)) {
			printf("erro get list cell\n");
			return enif_make_badarg(env);
		}

		if (strcmp(type_name, "int") == 0) {
			int iarg;
			if(!enif_get_int(env, head_args, &iarg)) {
				printf("error getting int arg\n");
				return enif_make_badarg(env);
			}
			ints[ints_ptr] = iarg;
			args[i] = (void*)  &ints[ints_ptr];
			ints_ptr++;
		} else if (strcmp(type_name, "float") == 0) {
			double darg;
			if(!enif_get_double(env, head_args, &darg)) {
				printf("error getting float arg\n");
				return enif_make_badarg(env);
			}
			floats[floats_ptr] = (float) darg;
			args[i] = (void*)  &floats[floats_ptr];
			floats_ptr++;
		} else if (strcmp(type_name, "double") == 0) {
			double darg;
			if(!enif_get_double(env, head_args, &darg)) {
				printf("error getting double arg\n");
				return enif_make_badarg(env);
			}
			doubles[doubles_ptr] =  darg;
			args[i] = (void*)  &doubles[doubles_ptr];
			doubles_ptr++;
		} else if (strcmp(type_name, "tint") == 0) {
			CUdeviceptr *array_res;
			enif_get_resource(env, head_args, ARRAY_TYPE, (void **) &array_res);
			arrays[arrays_ptr] = *array_res;
			args[i] = (void*)  &arrays[arrays_ptr];
			arrays_ptr++;
		} else if (strcmp(type_name, "tfloat") == 0) {
			CUdeviceptr *array_res;
			enif_get_resource(env, head_args, ARRAY_TYPE, (void **) &array_res);
			arrays[arrays_ptr] = *array_res;
			args[i] = (void*)  &arrays[arrays_ptr];
			arrays_ptr++;

		} else if (strcmp(type_name, "tdouble") == 0) {
			CUdeviceptr *array_res;
			enif_get_resource(env, head_args, ARRAY_TYPE, (void **) &array_res);
			arrays[arrays_ptr] = *array_res;
			args[i] = (void*)  &arrays[arrays_ptr];
			arrays_ptr++;
		} else {
			printf("Type %s not suported\n", type_name);
			return enif_make_badarg(env);
		}

		list_types = tail_types;
		list_args = tail_args;
	}

	// launch kernel
	init_cuda(env);

	err = cuLaunchKernel(
		function,
		b1, b2, b3, // Nx1x1 blocks
		t1, t2, t3, // 1x1x1 threads
		0, 0, args, 0
	) ;
	if (err != CUDA_SUCCESS) fail_cuda(env,err,"cuLaunchKernel jit compile");

	cuCtxSynchronize();

	if(err != CUDA_SUCCESS)  {
		char message[200];//printf("its ok\n");
		const char *error;
		cuGetErrorString(err, &error);
		strcpy(message,"Error at kernel launch: ");
		strcat(message, error);
		enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
	}

	return enif_make_int(env, 0);
}

///////////////////////
/////////
////////  JIT COMPILATION KERNEL
////////
///////////////////

static ERL_NIF_TERM get_gpu_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  int nrow;
  int ncol;
  char type_name[1024];

  ERL_NIF_TERM result;
  CUdeviceptr dev_array;
  CUresult err;

  init_cuda(env);
  // printf("entrou get array\n");
  CUdeviceptr *array_res;

  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&array_res))
  {
    return enif_make_badarg(env);
  }

  dev_array = *array_res;

  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  if (strcmp(type_name, "float") == 0)
  {
    size_t result_size = sizeof(float) * (nrow * ncol);
    size_t data_size = sizeof(float) * (nrow * ncol);
    float *result_data = (float *)enif_make_new_binary(env, result_size, &result);

    float *ptr_matrix;
    ptr_matrix = result_data;

    //// MAKE CUDA CALL
    err = cuMemcpyDtoH(ptr_matrix, dev_array, data_size);

    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error (get_gpu_array_nif): error compying data from device to host: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    //////// END CUDA CALL
  }
  else if (strcmp(type_name, "int") == 0)
  {

    size_t result_size = sizeof(int) * (nrow * ncol);
    size_t data_size = sizeof(int) * (nrow * ncol);
    int *result_data = (int *)enif_make_new_binary(env, result_size, &result);

    int *ptr_matrix;
    ptr_matrix = result_data;

    //// MAKE CUDA CALL
    // printf("cuda get\n");
    // printf("pointer %p\n",dev_array);
    err = cuMemcpyDtoH(ptr_matrix, dev_array, data_size);

    if (err != CUDA_SUCCESS)
    {
      char message[200]; // printf("cuda get\n");
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error (get_gpu_array_nif) copying data from device to host: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    //////// END CUDA CALL
  }
  else if (strcmp(type_name, "double") == 0)
  {

    size_t result_size = sizeof(double) * (nrow * ncol);
    size_t data_size = sizeof(double) * (nrow * ncol);
    double *result_data = (double *)enif_make_new_binary(env, result_size, &result);

    double *ptr_matrix;
    ptr_matrix = result_data;

    //// MAKE CUDA CALL
    err = cuMemcpyDtoH(ptr_matrix, dev_array, data_size);

    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error (get_gpu_array_nif) copying data from device to host: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    //////// END CUDA CALL
  }
  else /* default: */
  {
    char message[200];
    strcpy(message, "Error (get_gpu_array_nif) copying data from device to host: ");
    strcat(message, type_name);
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }
  return result;
}

static ERL_NIF_TERM create_gpu_array_nx_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  ErlNifBinary array_el;

  int nrow, ncol;

  CUresult err;
  CUdeviceptr dev_array;

  init_cuda(env);

  if (!enif_inspect_binary(env, argv[0], &array_el))
    return enif_make_badarg(env);

  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  char type_name[1024];

  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  // FINAL TERM TO BE RETURNED:
  ERL_NIF_TERM term;

  if (strcmp(type_name, "float") == 0)
  {

    float *array;

    array = (float *)array_el.data;

    size_t data_size = sizeof(float) * ncol * nrow;
    // printf("size float: %d data size: %d\n", sizeof(float),data_size);
    ///// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size);

    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      // printf("nrow %d ncol %d size %d size bytes %d\n", nrow, ncol,nrow*ncol, data_size);
      strcpy(message, "Error (create_gpu_array_nx_nif:) cuMemAlloc size: %d");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    ///// MAKE CUDA CALL
    err = cuMemcpyHtoD(dev_array, array, data_size);
    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error create_gpu_array_nx_nif: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    /////////// END CUDA CALL
  }
  else if (strcmp(type_name, "int") == 0)
  {
    int *array;
    array = (int *)array_el.data;

    size_t data_size = sizeof(int) * ncol * nrow;

    ///// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size);

    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error create_gpu_array_nx_nif1: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    ///// MAKE CUDA CALL
    err = cuMemcpyHtoD(dev_array, array, data_size);
    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error create_gpu_array_nx_nif2: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    /////////// END CUDA CALL
  }
  else if (strcmp(type_name, "double") == 0)
  {
    double *array;

    array = (double *)array_el.data;

    size_t data_size = sizeof(double) * ncol * nrow;

    err = cuMemAlloc(&dev_array, data_size);
    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error create_gpu_array_nx_nif1: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    ///// MAKE CUDA CALL
    err = cuMemcpyHtoD(dev_array, array, data_size);
    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error create_gpu_array_nx_nif2: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    /////////// END CUDA CALL
  }
  /* more else if clauses */
  else /* default: */
  {

    char message[200];
    strcpy(message, "Error create_gpu_array_nx_nif: unknown type: ");
    strcat(message, type_name);
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  CUdeviceptr *gpu_res = (CUdeviceptr *)enif_alloc_resource(ARRAY_TYPE, sizeof(CUdeviceptr));
  *gpu_res = dev_array;
  term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);
  return term;
}

static ERL_NIF_TERM new_gpu_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  size_t data_size;
  int nrow, ncol;
  ERL_NIF_TERM term;

  CUresult err;
  CUdeviceptr dev_array;

  init_cuda(env);

  if (!enif_get_int(env, argv[0], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[1], &ncol))
  {
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[2];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }
  char type_name[1024];
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  if (strcmp(type_name, "float") == 0)
  {

    data_size = sizeof(float) * ncol * nrow;

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size);

    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error new_gpu_array_nif: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    // END CUDA CALL
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * ncol * nrow;

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size);

    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error new_gpu_array_nif: ");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    // END CUDA CALL

    // printf("pointer %p\n",dev_array);
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * ncol * nrow;

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size);

    if (err != CUDA_SUCCESS)
    {
      char message[200];
      const char *error;
      cuGetErrorString(err, &error);
      strcpy(message, "Error new_gpu_array_nif:");
      strcat(message, error);
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    // END CUDA CALL
  }
  else /* default: */
  {
    char message[200];
    strcpy(message, "Error new_gpu_array_nif: unknown type: ");
    strcat(message, type_name);
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  CUdeviceptr *gpu_res = (CUdeviceptr *)enif_alloc_resource(ARRAY_TYPE, sizeof(CUdeviceptr));
  *gpu_res = dev_array;
  term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);
  return term;
}

/////////////////////////////
////////
///////////////////////
//////////////////////////////
//////////////// END CUDA DRIVER API CODE
/////////////////////////////
/////////////////////////////
/////////////////////////////

static ERL_NIF_TERM new_pinned_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{

  float *host_matrix;
  cudaError_t error_gpu;
  int length;
  ERL_NIF_TERM list;
  ERL_NIF_TERM head;
  ERL_NIF_TERM tail;
  if (!enif_get_list_cell(env, argv[0], &head, &tail))
    return enif_make_badarg(env);
  if (!enif_get_int(env, argv[1], &length))
    return enif_make_badarg(env);

  size_t data_size = sizeof(float) * (length + 2);

  ///// MAKE CUDA CALL
  // teste host_matrix = (float*) malloc(data_size);
  cudaMallocHost((void **)&host_matrix, data_size);
  error_gpu = cudaGetLastError();
  if (error_gpu != cudaSuccess)
  {
    char message[200];
    strcpy(message, "Error new_pinned_nif: ");
    strcat(message, cudaGetErrorString(error_gpu));
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }
  MX_SET_ROWS(host_matrix, 1);
  MX_SET_COLS(host_matrix, length);

  list = argv[0];

  for (int i = 2; i < (length + 2); i++)
  {
    enif_get_list_cell(env, list, &head, &tail);
    double dvalue;
    enif_get_double(env, head, &dvalue);
    host_matrix[i] = (float)dvalue;
    list = tail;
  }

  float **pinned_res = (float **)enif_alloc_resource(PINNED_ARRAY, sizeof(float *));
  *pinned_res = host_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, pinned_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(pinned_res);

  return term;
}

static ERL_NIF_TERM new_gmatrex_pinned_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  float *matrix;
  float *dev_matrix;
  cudaError_t error_gpu;
  float **array_res;

  if (!enif_get_resource(env, argv[0], PINNED_ARRAY, (void **)&array_res))
  {
    return enif_make_badarg(env);
  }

  matrix = (float *)*array_res;
  uint64_t data_size = sizeof(float) * (MX_LENGTH(matrix) - 2);

  matrix += 2;

  ///// MAKE CUDA CALL
  cudaMalloc((void **)&dev_matrix, data_size);
  error_gpu = cudaGetLastError();
  if (error_gpu != cudaSuccess)
  {
    char message[200];
    strcpy(message, "Error create_ref_nif: ");
    strcat(message, cudaGetErrorString(error_gpu));
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }
  ///// MAKE CUDA CALL
  cudaMemcpy(dev_matrix, matrix, data_size, cudaMemcpyHostToDevice);
  error_gpu = cudaGetLastError();
  if (error_gpu != cudaSuccess)
  {
    char message[200];
    strcpy(message, "Error create_ref_nif: ");
    strcat(message, cudaGetErrorString(error_gpu));
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  /////////// END CUDA CALL

  float **gpu_res = (float **)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ERL_NIF_TERM create_nx_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  ErlNifBinary matrix_el;
  cudaError_t error_gpu;
  int nrow, ncol;

  if (!enif_inspect_binary(env, argv[0], &matrix_el))
    return enif_make_badarg(env);

  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  char type_name[1024];

  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  // FINAL TERM TO BE RETURNED:
  ERL_NIF_TERM term;

  if (strcmp(type_name, "float") == 0)
  {

    float *matrix;
    float *dev_matrix;
    matrix = (float *)matrix_el.data;

    uint64_t data_size = sizeof(float) * ncol * nrow;

    ///// MAKE CUDA CALL
    cudaMalloc((void **)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error create_nx_ref_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    ///// MAKE CUDA CALL
    cudaMemcpy(dev_matrix, matrix, data_size, cudaMemcpyHostToDevice);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error create_nx_ref_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    /////////// END CUDA CALL

    float **gpu_res = (float **)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
    // ...and release the resource so that it will be freed when Erlang garbage collects
    enif_release_resource(gpu_res);
  }
  else if (strcmp(type_name, "int") == 0)
  {
    int *matrix;
    int *dev_matrix;
    matrix = (int *)matrix_el.data;

    uint64_t data_size = sizeof(int) * ncol * nrow;

    ///// MAKE CUDA CALL
    cudaMalloc((void **)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error create_nx_ref_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    ///// MAKE CUDA CALL
    cudaMemcpy(dev_matrix, matrix, data_size, cudaMemcpyHostToDevice);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error create_nx_ref_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    /////////// END CUDA CALL

    int **gpu_res = (int **)enif_alloc_resource(ARRAY_TYPE, sizeof(int *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
    // ...and release the resource so that it will be freed when Erlang garbage collects
    enif_release_resource(gpu_res);
  }
  else if (strcmp(type_name, "double") == 0)
  {
    double *matrix;
    double *dev_matrix;
    matrix = (double *)matrix_el.data;

    uint64_t data_size = sizeof(double) * ncol * nrow;

    ///// MAKE CUDA CALL
    cudaMalloc((void **)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error create_nx_ref_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    ///// MAKE CUDA CALL
    cudaMemcpy(dev_matrix, matrix, data_size, cudaMemcpyHostToDevice);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error create_nx_ref_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    /////////// END CUDA CALL

    double **gpu_res = (double **)enif_alloc_resource(ARRAY_TYPE, sizeof(double *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
    // ...and release the resource so that it will be freed when Erlang garbage collects
    enif_release_resource(gpu_res);
  }
  /* more else if clauses */
  else /* default: */
  {
    char message[200];
    strcpy(message, "Error create_nx_ref_nif: unknown type: ");
    strcat(message, type_name);
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  return term;
}

static ERL_NIF_TERM create_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  ErlNifBinary matrix_el;
  float *matrix;
  float *dev_matrix;
  cudaError_t error_gpu;

  if (!enif_inspect_binary(env, argv[0], &matrix_el))
    return enif_make_badarg(env);

  matrix = (float *)matrix_el.data;
  uint64_t data_size = sizeof(float) * (MX_LENGTH(matrix) - 2);

  matrix += 2;

  ///// MAKE CUDA CALL
  cudaMalloc((void **)&dev_matrix, data_size);
  error_gpu = cudaGetLastError();
  if (error_gpu != cudaSuccess)
  {
    char message[200];
    strcpy(message, "Error create_ref_nif: ");
    strcat(message, cudaGetErrorString(error_gpu));
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }
  ///// MAKE CUDA CALL
  cudaMemcpy(dev_matrix, matrix, data_size, cudaMemcpyHostToDevice);
  error_gpu = cudaGetLastError();
  if (error_gpu != cudaSuccess)
  {
    char message[200];
    strcpy(message, "Error create_ref_nif: ");
    strcat(message, cudaGetErrorString(error_gpu));
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  /////////// END CUDA CALL

  float **gpu_res = (float **)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ERL_NIF_TERM new_gpu_nx_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  size_t data_size;
  cudaError_t error_gpu;
  int nrow, ncol;
  ERL_NIF_TERM term;

  if (!enif_get_int(env, argv[0], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[1], &ncol))
  {
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[2];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }
  char type_name[1024];
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  if (strcmp(type_name, "float") == 0)
  {

    float *dev_matrix;
    data_size = sizeof(float) * ncol * nrow;

    //// MAKE CUDA CALL
    cudaMalloc((void **)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error new_gpu_nx_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    // END CUDA CALL

    float **gpu_res = (float **)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
    // ...and release the resource so that it will be freed when Erlang garbage collects
    enif_release_resource(gpu_res);
  }
  else if (strcmp(type_name, "int") == 0)
  {
    int *dev_matrix;
    data_size = sizeof(int) * ncol * nrow;

    //// MAKE CUDA CALL
    cudaMalloc((void **)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error new_gpu_nx_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    // END CUDA CALL

    int **gpu_res = (int **)enif_alloc_resource(ARRAY_TYPE, sizeof(int *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
    // ...and release the resource so that it will be freed when Erlang garbage collects
    enif_release_resource(gpu_res);
  }
  else if (strcmp(type_name, "double") == 0)
  {
    double *dev_matrix;
    data_size = sizeof(double) * ncol * nrow;

    //// MAKE CUDA CALL
    cudaMalloc((void **)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error new_gpu_nx_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }

    // END CUDA CALL

    double **gpu_res = (double **)enif_alloc_resource(ARRAY_TYPE, sizeof(double *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
    // ...and release the resource so that it will be freed when Erlang garbage collects
    enif_release_resource(gpu_res);
  }
  else /* default: */
  {
    char message[200];
    strcpy(message, "Error gpu_nx_nif: unknown type: ");
    strcat(message, type_name);
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }
  return term;
}

static ERL_NIF_TERM new_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  float *dev_matrix;
  int data_size;
  cudaError_t error_gpu;

  if (!enif_get_int(env, argv[0], &data_size))
  {
    return enif_make_badarg(env);
  }

  data_size = data_size * sizeof(float);

  //// MAKE CUDA CALL
  cudaMalloc((void **)&dev_matrix, data_size);
  error_gpu = cudaGetLastError();
  if (error_gpu != cudaSuccess)
  {
    char message[200];
    strcpy(message, "Error new_ref_nif: ");
    strcat(message, cudaGetErrorString(error_gpu));
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // END CUDA CALL

  float **gpu_res = (float **)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ERL_NIF_TERM get_nx_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  int nrow;
  int ncol;
  ERL_NIF_TERM result;
  cudaError_t error_gpu;

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  char type_name[1024];
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  if (strcmp(type_name, "float") == 0)
  {
    float **array_res;

    if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&array_res))
    {
      return enif_make_badarg(env);
    }

    float *dev_array = *array_res;

    int result_size = sizeof(float) * (nrow * ncol);
    int data_size = sizeof(float) * (nrow * ncol);
    float *result_data = (float *)enif_make_new_binary(env, result_size, &result);

    float *ptr_matrix;
    ptr_matrix = result_data;

    //// MAKE CUDA CALL
    cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error get_nx_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    //////// END CUDA CALL
  }
  else if (strcmp(type_name, "int") == 0)
  {
    int **array_res;

    if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&array_res))
    {
      return enif_make_badarg(env);
    }

    int *dev_array = *array_res;

    int result_size = sizeof(int) * (nrow * ncol);
    int data_size = sizeof(int) * (nrow * ncol);
    int *result_data = (int *)enif_make_new_binary(env, result_size, &result);

    int *ptr_matrix;
    ptr_matrix = result_data;

    //// MAKE CUDA CALL
    cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error get_nx_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    //////// END CUDA CALL
  }
  else if (strcmp(type_name, "double") == 0)
  {
    double **array_res;

    if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&array_res))
    {
      return enif_make_badarg(env);
    }

    double *dev_array = *array_res;

    double result_size = sizeof(double) * (nrow * ncol);
    double data_size = sizeof(double) * (nrow * ncol);
    double *result_data = (double *)enif_make_new_binary(env, result_size, &result);

    double *ptr_matrix;
    ptr_matrix = result_data;

    //// MAKE CUDA CALL
    cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost);
    error_gpu = cudaGetLastError();
    if (error_gpu != cudaSuccess)
    {
      char message[200];
      strcpy(message, "Error get_nx_nif: ");
      strcat(message, cudaGetErrorString(error_gpu));
      enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    }
    //////// END CUDA CALL
  }
  else /* default: */
  {
    char message[200];
    strcpy(message, "Error get_nx_nif: unknown type: ");
    strcat(message, type_name);
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }
  return result;
}

static ERL_NIF_TERM get_matrex_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  int nrow;
  int ncol;
  ERL_NIF_TERM result;
  float **array_res;
  cudaError_t error_gpu;

  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&array_res))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  float *dev_array = *array_res;

  int result_size = sizeof(float) * (nrow * ncol + 2);
  int data_size = sizeof(float) * (nrow * ncol);
  float *result_data = (float *)enif_make_new_binary(env, result_size, &result);

  float *ptr_matrix;
  ptr_matrix = result_data;
  ptr_matrix += 2;

  //// MAKE CUDA CALL
  cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost);
  error_gpu = cudaGetLastError();
  if (error_gpu != cudaSuccess)
  {
    char message[200];
    strcpy(message, "Error get_matrex_nif: ");
    strcat(message, cudaGetErrorString(error_gpu));
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }
  //////// END CUDA CALL

  MX_SET_ROWS(result_data, nrow);
  MX_SET_COLS(result_data, ncol);

  return result;
}

static ERL_NIF_TERM synchronize_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  cudaError_t error_gpu;

  ////// MAKE CUDA CALL
  cudaDeviceSynchronize();
  error_gpu = cudaGetLastError();
  if (error_gpu != cudaSuccess)
  {
    char message[200];
    strcpy(message, "Error synchronize_nif: ");
    strcat(message, cudaGetErrorString(error_gpu));
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }
  //// END CUDA CALL
  return enif_make_int(env, 0);
}

static ERL_NIF_TERM load_kernel_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{

  ERL_NIF_TERM e_name_module = argv[0];
  ERL_NIF_TERM e_name_fun = argv[1];

  unsigned int size_name_module;
  unsigned int size_name_fun;

  enif_get_list_length(env, e_name_fun, &size_name_fun);
  enif_get_list_length(env, e_name_module, &size_name_module);

  char kernel_name[1024];
  char func_name[1024];
  char lib_name[1024];
  char module_name[1024];

  enif_get_string(env, e_name_fun, kernel_name, size_name_fun + 1, ERL_NIF_LATIN1);
  enif_get_string(env, e_name_module, module_name, size_name_module + 1, ERL_NIF_LATIN1);

  strcpy(func_name, kernel_name);
  strcat(func_name, "_call");
  strcpy(lib_name, "priv/");
  strcat(lib_name, module_name);
  strcat(lib_name, ".so");

  // printf("libname %s\n",lib_name);
  // printf("func name a %s\n",func_name);
  // printf("module name %s\n", module_name);

  void *m_handle = dlopen(lib_name, RTLD_NOW);
  if (m_handle == NULL)
  {
    fprintf(stderr, "dlopen failure: %s\n", dlerror());
    char message[200];
    strcpy(message, "Error opening shared library for the programa. It was not found.\n");
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    return enif_make_int(env, 0);
  }

  // printf("Pointer %p\n",m_handle);

  void (*fn)();
  fn = (void (*)())dlsym(m_handle, func_name);

  if (fn == NULL)
  {
    fprintf(stderr, "dlopen failure: %s\n", dlerror());
    char message[200];
    strcpy(message, "Error opening .so file: ");
    strcat(message, func_name);
    strcat(message, " was not found!");
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    return enif_make_int(env, 0);
  }

  void (**kernel_res)() = (void (**)())enif_alloc_resource(KERNEL_TYPE, sizeof(void *));

  // Let's create conn and let the resource point to it

  *kernel_res = fn;

  // We can now make the Erlang term that holds the resource...
  ERL_NIF_TERM term = enif_make_resource(env, kernel_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(kernel_res);

  return term;
}

static ERL_NIF_TERM load_fun_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{

  ERL_NIF_TERM e_name_module = argv[0];
  ERL_NIF_TERM e_name_fun = argv[1];

  unsigned int size_name_module;
  unsigned int size_name_fun;

  enif_get_list_length(env, e_name_fun, &size_name_fun);
  enif_get_list_length(env, e_name_module, &size_name_module);

  char kernel_name[1024];
  char func_name[1024];
  char lib_name[1024];
  char module_name[1024];

  enif_get_string(env, e_name_fun, kernel_name, size_name_fun + 1, ERL_NIF_LATIN1);
  enif_get_string(env, e_name_module, module_name, size_name_module + 1, ERL_NIF_LATIN1);

  strcpy(func_name, "get_");
  strcat(func_name, kernel_name);
  strcat(func_name, "_ptr");
  strcpy(lib_name, "priv/");
  strcat(lib_name, module_name);
  strcat(lib_name, ".so");

  // printf("libname %s\n",lib_name);

  void *m_handle = dlopen(lib_name, RTLD_NOW);
  if (m_handle == NULL)
  {
    char message[200];
    strcpy(message, "Error opening shared library for the programa. It was not found.");
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    return enif_make_int(env, 0);
  }

  // printf("Pointer %p\n",m_handle);

  // printf("function name %s \nlib name %s pointer %p\n", func_name, lib_name, m_handle);

  void *(*fn)();
  fn = (void *(*)())dlsym(m_handle, func_name);

  // printf("pointer function a %p %li\n",fn, (long int) fn);

  if (fn == NULL)
  {

    char message[200];
    strcpy(message, "Error opening .so file: ");
    strcat(message, kernel_name);
    strcat(message, " was not found!");
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
    return enif_make_int(env, 0);
  }

  void *ptr = fn();

  /// printf("function pointer %p\n",ptr);

  void **kernel_res = (void **)enif_alloc_resource(KERNEL_TYPE, sizeof(void *));

  // Let's create conn and let the resource point to it

  *kernel_res = ptr;

  // printf("kernel resource %p\n", *kernel_res);
  // printf("erlang resource %p\n", kernel_res);

  // We can now make the Erlang term that holds the resource...
  ERL_NIF_TERM term = enif_make_resource(env, kernel_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(kernel_res);
  // printf("saiu\n");
  // printf("term %p\n", term);
  return term;
}

static ERL_NIF_TERM spawn_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  void (**kernel_res)(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType *type, ErlNifResourceType *ftype);
  // void (**kernel_res)();
  // float **array_res;
  // printf("spawn begin\n");
  // fflush(stdout);
  if (!enif_get_resource(env, argv[0], KERNEL_TYPE, (void **)&kernel_res))
  {
    return enif_make_badarg(env);
  }

  void (*fn)(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType *type, ErlNifResourceType *ftype) = *kernel_res;
  // void (*fn)() = *kernel_res;
  // float *array = *array_res;
  // printf("ok nif");
  (*fn)(env, argv, ARRAY_TYPE, KERNEL_TYPE);
  //(*fn)();

  return enif_make_int(env, 0);
}

static ErlNifFunc nif_funcs[] = {
    {"jit_compile_nif", 2, jit_compile_nif},
    {"jit_launch_nif", 7, jit_launch_nif},
    {"new_gpu_array_nif", 3, new_gpu_array_nif},
    {"get_gpu_array_nif", 4, get_gpu_array_nif},
    {"create_gpu_array_nx_nif", 4, create_gpu_array_nx_nif},
    {"load_kernel_nif", 2, load_kernel_nif},
    {"load_fun_nif", 2, load_fun_nif},
    {"new_pinned_nif", 2, new_pinned_nif},
    {"new_gmatrex_pinned_nif", 1, new_gmatrex_pinned_nif},
    {"spawn_nif", 4, spawn_nif},
    {"create_nx_ref_nif", 4, create_nx_ref_nif},
    {"get_nx_nif", 4, get_nx_nif},
    {"new_gpu_nx_nif", 3, new_gpu_nx_nif},
    {"create_ref_nif", 1, create_ref_nif},
    {"new_ref_nif", 1, new_ref_nif},
    {"get_matrex_nif", 3, get_matrex_nif},
    {"synchronize_nif", 0, synchronize_nif}};

ERL_NIF_INIT(Elixir.PolyHok, nif_funcs, &load, NULL, NULL, NULL)
