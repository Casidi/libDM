//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void update_dist_local(__global float* query, __global float* allData, 
		__global float* allDists, __global int* allIndexes, int length, int dim,
		__local float* query_local) 
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	size_t local_size = get_local_size(0);
	size_t global_size = get_global_size(0);
	
	for (int i = lid; i < dim; i += local_size) {
		query_local[i] = query[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0;
	for (int i = 0; i < dim; ++i)
		sum += (allData[gid*dim + i] - query_local[i])*(allData[gid*dim + i] - query_local[i]);
	allDists[gid] = sum;

	allIndexes[gid] = gid;
}

__kernel void update_dist(__global float* query, __global float* allData,
	__global float* allDists, __global int* allIndexes, int length, int dim,
	__local float* query_local) 
{
	size_t gid = get_global_id(0);

	float sum = 0;
	for (int i = 0; i < dim; ++i)
		sum += (allData[gid*dim + i] - query[i])*(allData[gid*dim + i] - query[i]);
	allDists[gid] = sum;

	allIndexes[gid] = gid;
}
