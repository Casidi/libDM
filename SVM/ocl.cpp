#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include "svm.h"

using namespace std;

cl_program load_program(cl_context context, const char* filename, cl_device_id device)
{
	std::ifstream in(filename, std::ios_base::binary);
	if (!in.good()) {
		return 0;
	}

	// get file length
	in.seekg(0, ios_base::end);
	size_t length = in.tellg();
	in.seekg(0, ios_base::beg);

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
	int t = clBuildProgram(program, 0, 0, NULL, 0, 0);
	//printf("%d\n", t);
	if (t != CL_SUCCESS) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		//while (1);
		system("pause");
		return 0;
	}
	/*if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
	return 0;
	}*/

	return program;
}
cl_context context;
cl_program program;
cl_kernel cl_kernel_rbf = 0;
cl_kernel cl_kernel_predict = 0;
cl_command_queue queue;
void ocl_init(int kerneltype)
{
	//printf("%d\n", kerneltype);
	cl_int err;
	cl_uint num = 0;
	err = clGetPlatformIDs(0, NULL, &num);
	//printf("err %d num %d\n", err, num);
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

	cl_platform_id selectedPlatform = platforms[0];
	for (int i = 0; i < platforms.size(); ++i) {
		size_t length;
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, 0, &length);
		string infoBuffer;
		infoBuffer.resize(length);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, length, &infoBuffer[0], 0);
		cout << infoBuffer << endl;
		
		if (infoBuffer.find("NVIDIA") != string::npos || infoBuffer.find("AMD") != string::npos) {
			selectedPlatform = platforms[i];
			break;
		}
	}

	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(selectedPlatform), 0 };

	char *profile = NULL;
	size_t size;
	clGetPlatformInfo(selectedPlatform, CL_PLATFORM_NAME, NULL, profile, &size); // get size of profile char array
	profile = (char*)malloc(size);
	clGetPlatformInfo(selectedPlatform, CL_PLATFORM_NAME, size, profile, NULL); // get profile char array
																			//std::cout << profile << std::endl;

	context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
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
	string devname;
	devname.resize(cb);
	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
	std::cout << "Device: " << devname.c_str() << "\n";

	clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, 0, NULL, &cb);
	std::string devver;
	devver.resize(cb);
	clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, cb, &devver[0], 0);
	std::cout << "Device: " << devver.c_str() << "\n";



	/*cl_ulong sizez;
	clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &sizez, 0);
	std::cout << sizez << std::endl;*/


	//int err;
	//cl_command_queue_properties qprop[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	//queue = clCreateCommandQueueWithProperties(context, devices[0], qprop, &err);
	queue = clCreateCommandQueue(context, devices[0], NULL, &err);


	if (queue == 0) {
		std::cerr << "Can't create command queue\n";
		clReleaseContext(context);
		exit(0);
	}
	program = load_program(context, "svm_kernel.cl", devices[0]);

	if (program == 0) {
		std::cerr << "Can't load or build program\n";
		//system("PAUSE");
		exit(0);
	}
	if (kerneltype == LINEAR) {
		cl_kernel_rbf = clCreateKernel(program, "kernel_linear", 0);
	}
	else if (kerneltype == POLY) {
#ifdef LOCAL
		cl_kernel_rbf = clCreateKernel(program, "kernel_poly_local", 0);
#else
		cl_kernel_rbf = clCreateKernel(program, "kernel_poly", 0);
#endif
	}
	else if (kerneltype == RBF) {
//#define LOCAL
#ifdef LOCAL
		cl_kernel_rbf = clCreateKernel(program, "kernel_rbf_local", 0);
#else
		cl_kernel_rbf = clCreateKernel(program, "kernel_rbf", 0);
#endif
	}
	else if (kerneltype == SIGMOID) {
#ifdef LOCAL
		cl_kernel_rbf = clCreateKernel(program, "kernel_sigmoid_local", 0);
#else
		cl_kernel_rbf = clCreateKernel(program, "kernel_sigmoid", 0);
#endif
	}
	else if(kerneltype == 100) {
		cl_kernel_predict = clCreateKernel(program, "predict_kernel", 0);
	}
	else {
		// parallel with points
		cl_kernel_predict = clCreateKernel(program, "predict", 0);
	}
	cl_kernel_predict = clCreateKernel(program, "predict", 0);

	if (cl_kernel_rbf == 0 && cl_kernel_predict == 0) {
		std::cerr << "Can't load kernel\n";
		exit(0);
	}
}