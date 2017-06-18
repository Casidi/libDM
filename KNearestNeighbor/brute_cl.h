#pragma once

#include <ANN\ANN.h>
#include <cstdlib>
#include <CL\cl.h>
#include <ctime>
#include <cstdio>

class KNNBruteCL {
	int k;
	int dataLength;
	int dataDim;
	float** data;
	float* allDists;
	int *allIndexes;

	cl_context context;
	cl_kernel update_dist_kernel;
	cl_command_queue queue;
	cl_program program;

	cl_mem query_gpu;
	cl_mem all_data_gpu;
	cl_mem all_dists_gpu;
	cl_mem all_index_gpu;

	void check_cl_error(cl_int err, const char *file, int line)
	{
		if (err != CL_SUCCESS) {
			printf("Error with errorcode: %d in file %s in line %d \n", err, file, line);
		}
	}

	void print_build_log(cl_program program, cl_device_id device)
	{
		cl_int err;
		char *build_log;
		size_t build_log_size;
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
		check_cl_error(err, __FILE__, __LINE__);
		build_log = (char *)malloc(build_log_size);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
		check_cl_error(err, __FILE__, __LINE__);
		printf("%s\n", build_log);
		free(build_log);
	}

	cl_program load_program(char* fileName, cl_context context, cl_device_id device) {
		FILE* fp = fopen(fileName, "r");

		fseek(fp, 0, SEEK_END);
		int length = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		char* buffer = new char[length + 1];
		memset(buffer, 0, length + 1);
		fread(buffer, 1, length, fp);

		cl_program program = clCreateProgramWithSource(context, 1, (const char**)&buffer, 0, 0);
		if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
			print_build_log(program, device);
		}

		fclose(fp);
		delete buffer;

		return program;
	}

	void cleanupCL() {
		if (update_dist_kernel) {
			clReleaseKernel(update_dist_kernel);
			update_dist_kernel = 0;
		}

		if (program) {
			clReleaseProgram(program);
			program = 0;
		}

		if (query_gpu) {
			clReleaseMemObject(query_gpu);
			query_gpu = 0;
		}

		if (all_data_gpu) {
			clReleaseMemObject(all_data_gpu);
			all_data_gpu = 0;
		}
		if (all_dists_gpu) {
			clReleaseMemObject(all_dists_gpu);
			all_dists_gpu = 0;
		}
		if (all_index_gpu) {
			clReleaseMemObject(all_index_gpu);
			all_index_gpu = 0;
		}

		if (queue) {
			clReleaseCommandQueue(queue);
			queue = 0;
		}
		if (context) {
			clReleaseContext(context);
			context = 0;
		}
	}

	void initCL() {
		cl_uint num;
		cl_int err;
		clGetPlatformIDs(0, 0, &num);

		cl_platform_id *platforms = new cl_platform_id[num];
		clGetPlatformIDs(num, platforms, &num);
		cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0]), 0 };
		context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, &err);
		if (err != CL_SUCCESS) {
			printf("failed to create context\n");
		}

		size_t cb;
		clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
		cl_device_id* devices = new cl_device_id[cb/sizeof(cl_device_id)];
		clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

		queue = clCreateCommandQueue(context, devices[0], 0, 0);
		program = load_program("knn_kernels.cl", context, devices[0]);

		query_gpu = clCreateBuffer(context, 0, sizeof(cl_float) * dataDim, NULL, NULL);
		all_data_gpu = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * dataDim*dataLength, data[0], NULL);
		all_dists_gpu = clCreateBuffer(context, 0, sizeof(cl_float) * dataLength, NULL, NULL);
		all_index_gpu = clCreateBuffer(context, 0, sizeof(cl_int) * dataLength, NULL, NULL);

		update_dist_kernel = clCreateKernel(program, "update_dist_local", 0);
		clSetKernelArg(update_dist_kernel, 0, sizeof(cl_mem), &query_gpu);
		clSetKernelArg(update_dist_kernel, 1, sizeof(cl_mem), &all_data_gpu);
		clSetKernelArg(update_dist_kernel, 2, sizeof(cl_mem), &all_dists_gpu);
		clSetKernelArg(update_dist_kernel, 3, sizeof(cl_mem), &all_index_gpu);
		clSetKernelArg(update_dist_kernel, 4, sizeof(cl_int), &dataLength);
		clSetKernelArg(update_dist_kernel, 5, sizeof(cl_int), &dataDim);
		clSetKernelArg(update_dist_kernel, 6, sizeof(cl_float)*dataDim, NULL);
	}

	void updateCL(float* query) {
		int reserveNumber = dataLength % 256;
		size_t globalSize = dataLength - reserveNumber;

		cl_int err;
		clEnqueueWriteBuffer(queue, query_gpu, CL_TRUE, 0, sizeof(float)*dataDim, query, 0, 0, 0);
		err = clEnqueueNDRangeKernel(queue, update_dist_kernel, 1, 0, (size_t*)&globalSize, 0, 0, 0, 0);
		err = clEnqueueReadBuffer(queue, all_dists_gpu, CL_TRUE, 0, sizeof(float)*globalSize, allDists, 0, 0, 0);
		err = clEnqueueReadBuffer(queue, all_index_gpu, CL_TRUE, 0, sizeof(int)*globalSize, allIndexes, 0, 0, 0);

		for (int i = dataLength - reserveNumber; i < dataLength; ++i) {
			float sum = 0;
			for (int j = 0; j < dataDim; ++j) {
				sum += (data[i][j] - query[j])*(data[i][j] - query[j]);
			}
			allDists[i] = sum;
			allIndexes[i] = i;
		}
	}

	void update(float* query) {
		for (int i = 0; i < dataLength; ++i) {
			float sum = 0;
			for (int j = 0; j < dataDim; ++j)
				sum += (data[i][j] - query[j])*(data[i][j] - query[j]);
			allDists[i] = sum;

			allIndexes[i] = i;
		}
	}

	void kNearestPQ(int k, int* nn_idx, float* dists) {
		int currentLength = 0;
		int i;

		for (int pointIndex = 0; pointIndex < dataLength; ++pointIndex) {
			for (i = currentLength; i > 0; --i) {
				if (dists[i - 1] > allDists[pointIndex]) {
					if (i == k)
						continue;
					dists[i] = dists[i - 1];
					nn_idx[i] = nn_idx[i - 1];
				}
				else {
					break;
				}
			}

			if (i == k)
				continue;

			dists[i] = allDists[pointIndex];
			nn_idx[i] = allIndexes[pointIndex];

			if (currentLength < k)
				++currentLength;
		}
	}

	void kNearest(int k, int* nn_idx, float* dists) {
		for (int i = 0; i < k; ++i) {
			for (int j = 1; j < dataLength - i; ++j) {
				if (allDists[j - 1] < allDists[j]) {
					float temp = allDists[j - 1];
					allDists[j - 1] = allDists[j];
					allDists[j] = temp;

					int tempIndex = allIndexes[j - 1];
					allIndexes[j - 1] = allIndexes[j];
					allIndexes[j] = tempIndex;
				}
			}

			nn_idx[i] = allIndexes[dataLength - i - 1];
			dists[i] = allDists[dataLength - i - 1];
		}
	}

public:
	KNNBruteCL(int k = 10){
		this->k = k;
		allDists = NULL;
		allIndexes = NULL;

		context = 0;
		update_dist_kernel = 0;
		queue = 0;
		program = 0;

		query_gpu = 0;
		all_data_gpu = 0;
		all_dists_gpu = 0;
		all_index_gpu = 0;
	}

	void fit(float** pa, int n, int dd) {
		data = pa;
		dataLength = n;
		dataDim = dd;

		if (allDists)
			delete[] allDists;
		if (allIndexes)
			delete[] allIndexes;
		allDists = new float[dataLength];
		allIndexes = new int[dataLength];

		cleanupCL();
		initCL();
	}

	~KNNBruteCL() {
		delete[] allDists;
		delete[] allIndexes;

		cleanupCL();
	}

	void knn(float* query, int* nn_idx, float* dists) {
		//update(query);
		updateCL(query);

		//kNearest(k, nn_idx, dists);		
		kNearestPQ(k, nn_idx, dists);
	}
};