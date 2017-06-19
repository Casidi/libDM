#include "kmeanslib.h"
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

KMeans::KMeans(int n_clusters = 8)
{
	this->n_clusters = n_clusters;
	cl_uint num;
	clGetPlatformIDs(0, 0, &num);
	if (num > 0) {
		ocl = true;
	}
	else {
		ocl = false;
	}
}

void KMeans::fit(double ** x, int n, int dim)
{
	if (ocl)
		ocl_kmeans(x, dim, n, n_clusters, (double)0.0001);
	else
		seq_kmeans(x, dim, n, n_clusters, (double)0.0001);
}

double KMeans::predict(double * x, int dim)
{
	return find_nearest_cluster(n_clusters, dim, x);
}

void KMeans::predict_multiple(double ** x, int n, int dim, double * label)
{
	for (int i = 0; i < n; i++) {
		label[i] = find_nearest_cluster(n_clusters, dim, x[i]);
	}
}

double KMeans::get_label(int i)
{
	return membership[i];
}

cl_program KMeans::load_program(cl_context context, const char* filename, cl_device_id device)
{
	std::ifstream in(filename, std::ios_base::binary);
	if (!in.good()) {
		return 0;
	}

	// get file length
	in.seekg(0, std::ios_base::end);
	size_t length = in.tellg();
	in.seekg(0, std::ios_base::beg);

	// read program source
	std::vector<char> data(length + 1);
	in.read(&data[0], length);
	data[length] = 0;

	// create and build program 
	const char* source = &data[0];
	cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
	if (program == 0) {
		return 0;
	}
	int t = clBuildProgram(program, 0, 0, 0, 0, 0);
	//printf("%d\n", t);
	if (t != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char *)malloc(log_size);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
		return 0;
	}
	return program;
}


/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
void KMeans::ocl_kmeans(double **objects,      /* in: [numObjs][numCoords] */
	int     numCoords,    /* no. features */
	int     numObjs,      /* no. objects */
	int     numClusters,  /* no. clusters */
	double   threshold)
{
	membership = (int*)malloc(sizeof(int)*numObjs);
	cl_int err;
	cl_uint num;
	err = clGetPlatformIDs(0, 0, &num);
	if (err != CL_SUCCESS) {
		std::cerr << "Unable to get platforms\n";
		exit(0);
	}
	std::vector<cl_platform_id> platforms(num);
	err = clGetPlatformIDs(num, &platforms[0], &num);
	//printf("platform num %d\n", num);
	if (err != CL_SUCCESS) {
		std::cerr << "Unable to get platform ID\n";
		exit(0);
	}

	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, 
		reinterpret_cast<cl_context_properties>(platforms[1]), 0 };

	char *profile = NULL;
	size_t size;
	clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, NULL, profile, &size); // get size of profile char array
	profile = (char*)malloc(size);
	clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, size, profile, NULL); // get profile char array
																			//std::cout << profile << std::endl;

	cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
	if (context == 0) {
		std::cerr << "Can't create OpenCL context\n";
		exit(0);
	}

	size_t cb;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
	std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
	//printf("device num %d\n", devices.size());
	clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
	std::string devname;
	devname.resize(cb);
	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
	//std::cout << "Device: " << devname.c_str() << "\n";

	clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, 0, NULL, &cb);
	std::string devver;
	devver.resize(cb);
	clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, cb, &devver[0], 0);
	//std::cout << "Device: " << devver.c_str() << "\n";

	//int err;
	//cl_queue_properties qprop[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &err);

	if (queue == 0) {
		std::cerr << "Can't create command queue\n";
		clReleaseContext(context);
		exit(0);
	}
	int      i, j, /*index,*/ loop = 0;
	int     *newClusterSize; /* [numClusters]: no. objects assigned in each
							 new cluster */
	double    delta;          /* % of objects change their clusters */
	//double  **clusters;       /* out: [numClusters][numCoords] */
	double  **newClusters;    /* [numClusters][numCoords] */
	double  *dimObjects;
	double  *dimClusters;
	clusters = (double**)malloc(numClusters * sizeof(double*));
	assert(clusters != NULL);
	clusters[0] = (double*)malloc(numClusters * numCoords * sizeof(double));
	assert(clusters[0] != NULL);
	for (i = 1; i < numClusters; i++)
		clusters[i] = clusters[i - 1] + numCoords;


	//malloc2D(dimObjects, numCoords, numObjs, double);
	dimObjects = (double*)malloc(sizeof(double)*numObjs * numCoords);
	for (i = 0; i < numObjs; i++) {
		for (j = 0; j < numCoords; j++) {
			dimObjects[i*numCoords + j] = objects[i][j];
		}
	}


	/* pick first numClusters elements of objects[] as initial cluster centers*/
	//malloc2D(dimClusters, numCoords, numClusters, double);
	dimClusters = (double*)malloc(sizeof(double)*numClusters*numCoords);
	for (i = 0; i < numClusters; i++) {
		for (j = 0; j < numCoords; j++) {
			dimClusters[i*numCoords + j] = dimObjects[i*numCoords + j];
		}
	}


	/* initialize membership[] */
	for (i = 0; i < numObjs; i++) membership[i] = -1;
	int *newmembership = (int*)calloc(numObjs, sizeof(int));

	/* need to initialize newClusterSize and newClusters[0] to all 0 */
	newClusterSize = (int*)calloc(numClusters, sizeof(int));
	assert(newClusterSize != NULL);

	newClusters = (double**)malloc(numClusters * sizeof(double*));
	assert(newClusters != NULL);
	newClusters[0] = (double*)calloc(numClusters * numCoords, sizeof(double));
	assert(newClusters[0] != NULL);
	for (i = 1; i < numClusters; i++)
		newClusters[i] = newClusters[i - 1] + numCoords;

	cl_mem cl_Objects = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double) * numObjs * numCoords, NULL, NULL);
	cl_mem cl_deviceClusters = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * numClusters * numCoords, &dimClusters[0], NULL);
	cl_mem cl_membership = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * numObjs, NULL, NULL);

	if (cl_Objects == 0 || cl_deviceClusters == 0 || cl_membership == 0) {
		std::cerr << "Can't create OpenCL buffer\n";
	}
	cl_program program = load_program(context, "kmeans_kernel.cl", devices[0]);

	if (program == 0) {
		std::cerr << "Can't load or build program\n";
	}
	cl_kernel kernel = clCreateKernel(program, "find_nearest_cluster", 0);
	if (kernel == 0) {
		std::cerr << "Can't load kernel\n";
	}
	clSetKernelArg(kernel, 0, sizeof(int), &numClusters);
	clSetKernelArg(kernel, 1, sizeof(int), &numCoords);
	clSetKernelArg(kernel, 2, sizeof(int), &numObjs);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_Objects);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_deviceClusters);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_membership);
	clSetKernelArg(kernel, 6, numClusters * numCoords * sizeof(cl_double), NULL);

	size_t work_size = numObjs;

	cl_ulong iotime = 0;
	//cl_event event;

	clEnqueueWriteBuffer(queue, cl_Objects, CL_TRUE, 0, sizeof(double)*numObjs*numCoords, dimObjects, 0, NULL, NULL);

	size_t global = 0, local = 256;
	while (global < work_size)
		global += local;

	do {

		//cl_deviceClusters = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * numClusters * numCoords, &dimClusters[0], NULL);
		clEnqueueWriteBuffer(queue, cl_deviceClusters, CL_TRUE, 0, sizeof(double)*numClusters*numCoords, dimClusters, 0, NULL, NULL);
		//size_t global = 131072, local = 512;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local, 0, 0, 0);
		//err = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &work_size, 0, 0, 0, 0);
		if (err == CL_SUCCESS) {
			//event.push_back(t);
			err = clEnqueueReadBuffer(queue, cl_membership, CL_TRUE, 0, sizeof(cl_int) * numObjs, &newmembership[0], 0, 0, NULL);
		}

		delta = 0;
		for (i = 0; i < numObjs; i++) {
			if (membership[i] != newmembership[i]) {
				delta += 1;
			}
			newClusterSize[newmembership[i]]++;
			for (j = 0; j < numCoords; j++)
				newClusters[newmembership[i]][j] += objects[i][j];
		}
		for (i = 0; i < numObjs; i++) {
			membership[i] = newmembership[i];
		}
		for (i = 0; i < numClusters; i++) {
			for (j = 0; j < numCoords; j++) {
				if (newClusterSize[i] > 0)
					dimClusters[i*numCoords + j] = newClusters[i][j] / newClusterSize[i];
				newClusters[i][j] = 0.0;   /* set back to 0 */
			}
			newClusterSize[i] = 0;   /* set back to 0 */
		}

		delta /= numObjs;


	} while (delta > threshold && loop++ < 500);

	//*loop_iterations = loop + 1;
	std::cout << loop << std::endl;

	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
	for (i = 0; i < numClusters; i++)
		for (int j = 0; j < numCoords; j++)
			clusters[i][j] = dimClusters[i*numCoords + j];

	int ret = clFlush(queue);
	ret = clFinish(queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(cl_Objects);
	ret = clReleaseMemObject(cl_deviceClusters);
	ret = clReleaseMemObject(cl_membership);
	ret = clReleaseCommandQueue(queue);
	ret = clReleaseContext(context);
}


double euclid_dist_2(int    numdims,  /* no. dimensions */
	double *coord1,   /* [numdims] */
	double *coord2)   /* [numdims] */
{
	int i;
	double ans = 0.0;

	for (i = 0; i < numdims; i++)
		ans += (coord1[i] - coord2[i]) * (coord1[i] - coord2[i]);

	return(ans);
}

int KMeans::find_nearest_cluster(int     numClusters, /* no. clusters */
	int     numCoords,   /* no. coordinates */
	double  *object      /* [numCoords] */)
{
	int   index, i;
	double dist, min_dist;

	/* find the cluster id that has min distance to object */
	index = 0;
	min_dist = euclid_dist_2(numCoords, object, clusters[0]);

	for (i = 1; i < numClusters; i++) {
		dist = euclid_dist_2(numCoords, object, clusters[i]);
		/* no need square root */
		if (dist < min_dist) { /* find the min and its array index */
			min_dist = dist;
			index = i;
		}
	}
	return(index);
}

// sequentail version
/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */

double KMeans::seq_euclid_dist_2(int    numdims,  /* no. dimensions */
	double *coord1,   /* [numdims] */
	double *coord2)   /* [numdims] */
{
	int i;
	double ans = 0.0;

	for (i = 0; i < numdims; i++)
		ans += (coord1[i] - coord2[i]) * (coord1[i] - coord2[i]);

	return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/

int KMeans::seq_find_nearest_cluster(int     numClusters, /* no. clusters */
	int     numCoords,   /* no. coordinates */
	double  *object,      /* [numCoords] */
	double **clusters)    /* [numClusters][numCoords] */
{
	int   index, i;
	double dist, min_dist;

	/* find the cluster id that has min distance to object */
	index = 0;
	min_dist = seq_euclid_dist_2(numCoords, object, clusters[0]);

	for (i = 1; i < numClusters; i++) {
		dist = seq_euclid_dist_2(numCoords, object, clusters[i]);
		/* no need square root */
		if (dist < min_dist) { /* find the min and its array index */
			min_dist = dist;
			index = i;
		}
	}
	return(index);
}

/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
void KMeans::seq_kmeans(double **objects,      /* in: [numObjs][numCoords] */
	int     numCoords,    /* no. features */
	int     numObjs,      /* no. objects */
	int     numClusters,  /* no. clusters */
	double   threshold    /* % objects change membership */
/* out: [numObjs] */)
{
	membership = (int*)malloc(sizeof(int)*numCoords);
	int      i, j, index, loop = 0;
	int     *newClusterSize; /* [numClusters]: no. objects assigned in each
							 new cluster */
	double    delta;          /* % of objects change their clusters */
	//double  **clusters;       /* out: [numClusters][numCoords] */
	double  **newClusters;    /* [numClusters][numCoords] */

							  /* allocate a 2D space for returning variable clusters[] (coordinates
							  of cluster centers) */
	clusters = (double**)malloc(numClusters * sizeof(double*));
	assert(clusters != NULL);
	clusters[0] = (double*)malloc(numClusters * numCoords * sizeof(double));
	assert(clusters[0] != NULL);
	for (i = 1; i < numClusters; i++)
		clusters[i] = clusters[i - 1] + numCoords;

	/* pick first numClusters elements of objects[] as initial cluster centers*/
	for (i = 0; i < numClusters; i++)
		for (j = 0; j < numCoords; j++)
			clusters[i][j] = objects[i][j];

	/* initialize membership[] */
	for (i = 0; i < numObjs; i++) membership[i] = -1;

	/* need to initialize newClusterSize and newClusters[0] to all 0 */
	newClusterSize = (int*)calloc(numClusters, sizeof(int));
	assert(newClusterSize != NULL);

	newClusters = (double**)malloc(numClusters * sizeof(double*));
	assert(newClusters != NULL);
	newClusters[0] = (double*)calloc(numClusters * numCoords, sizeof(double));
	assert(newClusters[0] != NULL);
	for (i = 1; i < numClusters; i++)
		newClusters[i] = newClusters[i - 1] + numCoords;

	do {
		delta = 0.0;
		for (i = 0; i < numObjs; i++) {
			/* find the array index of nestest cluster center */
			index = seq_find_nearest_cluster(numClusters, numCoords, objects[i],
				clusters);

			/* if membership changes, increase delta by 1 */
			if (membership[i] != index) delta += 1.0;

			/* assign the membership to object i */
			membership[i] = index;

			/* update new cluster centers : sum of objects located within */
			newClusterSize[index]++;
			for (j = 0; j < numCoords; j++)
				newClusters[index][j] += objects[i][j];
		}

		/* average the sum and replace old cluster centers with newClusters */
		for (i = 0; i < numClusters; i++) {
			for (j = 0; j < numCoords; j++) {
				if (newClusterSize[i] > 0)
					clusters[i][j] = newClusters[i][j] / newClusterSize[i];
				newClusters[i][j] = 0.0;   /* set back to 0 */
			}
			newClusterSize[i] = 0;   /* set back to 0 */
		}

		delta /= numObjs;
	} while (delta > threshold && loop++ < 500);
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
}