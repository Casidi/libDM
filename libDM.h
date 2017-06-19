#pragma once
#include "Classify.h"
#include "NaiveBayes\NaiveBayesBase.h"
#include "KNearestNeighbor\brute_cl.h"
#include "KMeans\kmeanslib.h"
#include "SVM\svmlib.h"

#include <vector>
#include <map>
#include <algorithm>
#include <CL\cl.h>

using namespace std;

class KNearestNeighbor : public Classify {
	bool isValidCL;
	KNNBruteCL *knnbcl;
	ANNbruteForce *knnbf;
	float** trainData;
	double* trainLabel; //use the data from the outside of class
	int k;
	int nClass;

	float** allocFloat2D(int d1, int d2) {
		float* block = new float[d1*d2];
		float** result = new float*[d1];
		for (int i = 0; i < d1; ++i)
			result[i] = block + (i*d2);
		return result;
	}

public:
	KNearestNeighbor(int k = 10) {
		knnbcl = NULL;
		knnbf = NULL;
		trainData = NULL;
		this->k = k;

		cl_uint num;
		clGetPlatformIDs(0, 0, &num);
		if (num > 0)
			isValidCL = true;
		else
			isValidCL = false;

		//for TEST!!!!!
		//isValidCL = false;
	}

	~KNearestNeighbor() {
		if (knnbcl)
			delete knnbcl;
		if (knnbf)
			delete knnbf;
		if (trainData) {
			delete[] trainData[0];
			delete trainData;
		}
	}

	virtual void fit(double **x, double *y, int n, int dim) {
		vector<double> classCount;
		for (int i = 0; i < n; ++i)
			if (find(classCount.begin(), classCount.end(), y[i]) == classCount.end())
				classCount.push_back(y[i]);
		nClass = classCount.size();

		if (trainData) {
			delete[] trainData[0];
			delete[] trainData;
			trainData = NULL;
		}

		trainData = allocFloat2D(n, dim);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < dim; ++j)
				trainData[i][j] = x[i][j];

		trainLabel = y;

		if (isValidCL) {
			if (knnbcl) {
				delete knnbcl;
				knnbcl = NULL;
			}
			knnbcl = new KNNBruteCL(k);
			knnbcl->fit(trainData, n, dim);
		}
		else {
			if (knnbf) {
				delete knnbf;
				knnbf = NULL;
			}
			knnbf = new ANNbruteForce(trainData, n, dim);
		}
	}

	virtual double predict(double *x, int dim) {
		float* tempData = new float[dim];
		float* allDists = new float[k];
		int* allIndexes = new int[k];

		for (int i = 0; i < dim; ++i)
			tempData[i] = x[i];

		if (isValidCL) {
			knnbcl->knn(tempData, allIndexes, allDists);
		}
		else {
			knnbf->annkSearch(tempData, k, allIndexes, allDists);
		}

		map<double, int> freqCount;
		for (int i = 0; i < k; ++i) {
			if (freqCount.find(trainLabel[allIndexes[i]]) == freqCount.end())
				freqCount[trainLabel[allIndexes[i]]] = 1;
			else
				freqCount[trainLabel[allIndexes[i]]] += 1;
		}

		double maxKey = freqCount.begin()->first;
		int maxValue = freqCount.begin()->second;
		for (map<double, int>::iterator iter = freqCount.begin(); iter != freqCount.end(); ++iter) {
			if (iter->second > maxValue) {
				maxValue = iter->second;
				maxKey = iter->first;
			}
		}

		delete[] tempData;
		delete[] allDists;
		delete[] allIndexes;

		return maxKey;
	}

	virtual void predict_multiple(double **x, int n, int dim, double *label) {
		for (int i = 0; i < n; ++i) {
			if (i % 200 == 0)
				cout << ".";
			label[i] = predict(x[i], dim);
		}
	}
};

class NaiveBayes : public Classify {
	NaiveBayesBase *nbb;
	int *data;
	int *trainLabel;

public:
	NaiveBayes() {
		nbb = NULL;
		data = NULL;
		trainLabel = NULL;
	}

	~NaiveBayes() {
		if (nbb != NULL)
			delete nbb;
		if (data != NULL)
			delete[] data;
		if (trainLabel != NULL)
			delete[] trainLabel;
	}

	virtual void fit(double **x, double *y, int n, int dim) {
		if (nbb) {
			delete nbb;
			nbb = NULL;
		}
		if (data) {
			delete[] data;
			data = NULL;
		}
		if (trainLabel) {
			delete[] trainLabel;
			trainLabel = NULL;
		}

		data = new int[n*dim];
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < dim; ++j)
				data[i*dim + j] = (int) x[i][j];

		trainLabel = new int[n];
		for (int i = 0; i < n; ++i)
			trainLabel[i] = (int)y[i];

		vector<int> classCount;
		for (int i = 0; i < n; ++i) {
			if (find(classCount.begin(), classCount.end(), trainLabel[i]) == classCount.end())
				classCount.push_back(trainLabel[i]);
		}

		nbb = new NaiveBayesBase(data, trainLabel, n, dim, classCount.size());
	}

	virtual double predict(double *x, int dim) {
		int *tempData = new int[dim];
		for (int i = 0; i < dim; ++i)
			tempData[i] = x[i];

		int result = nbb->predict(tempData);
		delete[] tempData;
		return result;
	}

	virtual void predict_multiple(double **x, int n, int dim, double *label) {
		cl_uint nPlatforms = 0;
		clGetPlatformIDs(0, 0, &nPlatforms);

		int *tempData = new int[n*dim];
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < dim; ++j)
				tempData[i*dim + j] = x[i][j];

		int *tempLabel = new int[n];

		//for test!!!!
		//nPlatforms = 0;

		if (nPlatforms > 0) {
			nbb->predictBatchCL(tempData, n, tempLabel);
		}
		else {
			nbb->predictBatch(tempData, n, tempLabel);
		}

		for (int i = 0; i < n; ++i)
			label[i] = tempLabel[i];

		delete[] tempData;
		delete[] tempLabel;
	}
};