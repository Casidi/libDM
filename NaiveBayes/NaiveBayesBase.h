#pragma once
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <CL\cl.hpp>

using namespace std;

class NaiveBayesBase {
	int nClass, dim, nTrain;
	float* piHatLog;
	float** thetaHatLog;
	float** oneMinusThetaHatLog;
	int* attribThresh;

	int** alloc2D(int d1, int d2) {
		int* block = new int[d1*d2];
		int** result = new int*[d1];
		for (int i = 0; i < d1; ++i)
			result[i] = block + (i*d2);
		return result;
	}

	void free2D(int** ptr) {
		delete[] ptr[0];
		delete[] ptr;
	}

	float** alloc2Df(int d1, int d2) {
		float* block = new float[d1*d2];
		float** result = new float*[d1];
		for (int i = 0; i < d1; ++i)
			result[i] = block + (i*d2);
		return result;
	}

	void free2Df(float** ptr) {
		delete[] ptr[0];
		delete[] ptr;
	}


	float* calcPIHat(int* classifierFreq, int nTrain) {
		float* result = new float[nClass];
		for (int i = 0; i < nClass; ++i)
			result[i] = (float)classifierFreq[i] / (float)nTrain;
		return result;
	}

	float* calcPIHatLog(float *piHat) {
		float* result = new float[nClass];
		for (int i = 0; i < nClass; ++i)
			result[i] = log10f(piHat[i]);
		return result;
	}

	float** calcThetaHat(int** pixelFreq, int* classifierFreq) {
		float** result = alloc2Df(dim, nClass);
		for (int i = 0; i < dim; ++i)
			for (int j = 0; j < nClass; ++j)
				result[i][j] = 0;

		for (int c = 0; c < nClass; ++c) {
			for (int i = 0; i < dim; ++i) {
				result[i][c] += (float)pixelFreq[i][c] / (float)classifierFreq[c];
			}
		}

		return result;
	}

	float** calcThetaHatLog(float** thetaHat) {
		float** result = alloc2Df(dim, nClass);
		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < nClass; ++j) {
				result[i][j] = log10f(thetaHat[i][j]);
			}
		}
		return result;
	}

	float** calcOneMinusThetaHatLog(float** thetaHat) {
		float** result = alloc2Df(dim, nClass);
		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < nClass; ++j) {
				result[i][j] = log10f(1 - thetaHat[i][j]);
			}
		}
		return result;
	}

	void posterior(int* point, float* result) {
		for (int i = 0; i < nClass; ++i)
			result[i] = 0;

		for (int c = 0; c < nClass; ++c) {
			result[c] += piHatLog[c];
			for (int j = 0; j < dim; ++j) {
				if (point[j] > attribThresh[j])
					result[c] += thetaHatLog[j][c];
				else
					result[c] += oneMinusThetaHatLog[j][c];
			}
		}
	}

	int** calcPixelFreq(int* label, int* data, int* classifierFreq) {
		int** result = alloc2D(dim, nClass);
		for (int i = 0; i < dim; ++i)
			for (int j = 0; j < nClass; ++j)
				result[i][j] = 1;
		for (int i = 0; i < nTrain; ++i) {
			int c = label[i];
			for (int j = 0; j < dim; ++j) {
				if (data[i*dim + j] > attribThresh[j]) {
					result[j][c] += 1;
				}
			}
		}

		return result;
	}

	int* calcClassifierFreq(int* label) {
		int* result = new int[nClass];
		for (int i = 0; i < nClass; ++i)
			result[i] = 1;
		for (int i = 0; i < nTrain; ++i) {
			int c = label[i];
			result[c] += 1;
		}
		return result;
	}

	int* calcAttribThresh(int* data) {
		int* result = new int[dim];
		for (int i = 0; i < dim; ++i) {
			int minVal = data[i];
			int maxVal = data[i];
			for (int j = 0; j < nTrain; ++j) {
				minVal = min(minVal, data[j*dim + i]);
				maxVal = max(maxVal, data[j*dim + i]);
			}
			result[i] = (minVal + maxVal) / 2;
		}

		return result;
	}
	
public:
	NaiveBayesBase(int* data, int* label, int n, int dim, int nClass)
		: nClass(nClass), dim(dim), nTrain(n) {

		attribThresh = calcAttribThresh(data);		
		int* classifierFreq = calcClassifierFreq(label);
		int** pixelFreq = calcPixelFreq(label, data, classifierFreq);		

		float* piHat = calcPIHat(classifierFreq, n);
		piHatLog = calcPIHatLog(piHat);
		float** thetaHat = calcThetaHat(pixelFreq, classifierFreq);
		thetaHatLog = calcThetaHatLog(thetaHat);
		oneMinusThetaHatLog = calcOneMinusThetaHatLog(thetaHat);

		delete[] piHat;
		free2Df(thetaHat);
		delete[] classifierFreq;
		free2D(pixelFreq);
	}

	~NaiveBayesBase() {
		delete[] piHatLog;
		free2Df(thetaHatLog);
		free2Df(oneMinusThetaHatLog);
		delete[] attribThresh;
	}

	int predict(int* point) {
		float* probs = new float[nClass];
		posterior(point, probs);

		float maxProb = probs[0];
		int result = 0;
		for (int i = 0; i < nClass; ++i)
			if (probs[i] > maxProb) {
				maxProb = probs[i];
				result = i;
			}

		delete[] probs;
		return result;
	}

	void predictBatch(int* points, int n, int* result) {
		for (int i = 0; i < n; ++i)
			result[i] = predict(points + i*dim);
	}

	void predictBatchCL(int* points, int n, int* result) {
		cl_int err = CL_SUCCESS;
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		cl_context_properties properties[] =
			{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[1])(), 0 };
		cl::Context context(CL_DEVICE_TYPE_GPU, properties);
		vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		//cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;

		cl::Buffer pointsCL(context, CL_MEM_READ_ONLY, sizeof(int)*dim*n, &err);
		cl::Buffer resultCL(context, CL_MEM_READ_WRITE, sizeof(int)*n, &err);
		cl::Buffer piHatLogCL(context, CL_MEM_READ_ONLY, sizeof(float)*nClass, NULL, &err);
		cl::Buffer thetaHatLogCL(context, CL_MEM_READ_ONLY, sizeof(float)*dim*nClass, NULL, &err);
		cl::Buffer oneMinuxThetaHatLogCL(context, CL_MEM_READ_ONLY, sizeof(float)*dim*nClass, NULL, &err);
		cl::Buffer attribThreshCL(context, CL_MEM_READ_ONLY, sizeof(int)*dim, NULL, &err);

		cl::CommandQueue queue(context, devices[0], 0, &err);
		err = queue.enqueueWriteBuffer(pointsCL, CL_TRUE, 0, sizeof(int)*dim*n, points, NULL);
		err = queue.enqueueWriteBuffer(piHatLogCL, CL_TRUE, 0, sizeof(float)*nClass, piHatLog, NULL);
		err = queue.enqueueWriteBuffer(thetaHatLogCL, CL_TRUE, 0, sizeof(float)*dim*nClass, thetaHatLog[0], NULL);
		err = queue.enqueueWriteBuffer(oneMinuxThetaHatLogCL, CL_TRUE, 0, sizeof(float)*dim*nClass, oneMinusThetaHatLog[0], NULL);

		ifstream sourceFile("nb_kernel.cl");
		stringstream ss;
		ss << sourceFile.rdbuf();
		sourceFile.close();

		string sourceStr = ss.str();
		cl::Program::Sources source(1,
			make_pair(sourceStr.data(), sourceStr.length()));
		cl::Program program = cl::Program(context, source);
		err = program.build(devices);
		if (err != CL_SUCCESS) {
			cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << endl;
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << endl;
		}

		cl::Kernel kernel(program, "predictBatch", &err);
		err = kernel.setArg(0, pointsCL);
		err = kernel.setArg(1, n);
		err = kernel.setArg(2, resultCL);
		err = kernel.setArg(3, piHatLogCL);
		err = kernel.setArg(4, thetaHatLogCL);
		err = kernel.setArg(5, oneMinuxThetaHatLogCL);
		err = kernel.setArg(6, attribThreshCL);
		err = kernel.setArg(7, dim);
		err = kernel.setArg(8, nClass);
		queue.finish();

		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange, NULL, NULL);
		err = queue.enqueueReadBuffer(resultCL, CL_TRUE, 0, sizeof(int)*n, result);
	}
};
