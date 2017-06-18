#include "libDM.h"
#include <fstream>

using namespace std;

double* loadTrainLabel() {
	ifstream inputFile("train-labels.idx1-ubyte", ios::in | ios::binary);

	unsigned char* rawData = new unsigned char[60000];
	inputFile.seekg(8);
	inputFile.read((char*)rawData, 60000);

	double* dData = new double[60000];
	for (int i = 0; i < 60000; ++i)
		dData[i] = rawData[i];

	inputFile.close();
	delete[] rawData;
	return dData;
}

double** allocDouble2D(int d1, int d2) {
	double* block = new double[d1*d2];
	double** result = new double*[d1];
	for (int i = 0; i < d1; ++i)
		result[i] = block + (i*d2);
	return result;
}

void freeDouble2D(double** ptr) {
	delete[] ptr[0];
	delete[] ptr;
}

double** loadTrainData() {
	ifstream inputFile("train-images.idx3-ubyte", ios::in | ios::binary);

	int nPixels = 28 * 28 * 60000;
	unsigned char* rawData = new unsigned char[nPixels];
	inputFile.seekg(16);
	inputFile.read((char*)rawData, nPixels);
	inputFile.close();

	double** result = allocDouble2D(60000, 28 * 28);
	for (int i = 0; i < 60000; ++i)
		for (int j = 0; j < 28 * 28; ++j)
			result[i][j] = rawData[i * 28 * 28 + j];

	delete[] rawData;
	return result;
}

double** loadTestData() {
	ifstream inputFile("t10k-images.idx3-ubyte", ios::in | ios::binary);

	int nPixels = 28 * 28 * 10000;
	unsigned char* rawData = new unsigned char[nPixels];
	inputFile.seekg(16);
	inputFile.read((char*)rawData, nPixels);
	inputFile.close();

	double** result = allocDouble2D(10000, 28 * 28);
	for (int i = 0; i < 10000; ++i)
		for (int j = 0; j < 28 * 28; ++j)
			result[i][j] = rawData[i * 28 * 28 + j];

	delete[] rawData;
	return result;
}

double* loadTestLabel() {
	ifstream inputFile("t10k-labels.idx1-ubyte", ios::in | ios::binary);

	unsigned char* rawData = new unsigned char[10000];
	inputFile.seekg(8);
	inputFile.read((char*)rawData, 10000);
	inputFile.close();

	double* dData = new double[10000];
	for (int i = 0; i < 10000; ++i)
		dData[i] = rawData[i];

	delete[] rawData;
	return dData;
}

void testNB(double** trainData, double* trainLabel, double** testData, double* testLabel) {
	NaiveBayes nb;
	nb.fit(trainData, trainLabel, 60000, 28 * 28);
}

void testKNN(double** trainData, double* trainLabel, double** testData, double* testLabel) {
	KNearestNeighbor knn;
	knn.fit(trainData, trainLabel, 60000, 28 * 28);
}

void testSVM(double** trainData, double* trainLabel, double** testData, double* testLabel) {
	SVM svm;
	svm.fit(trainData, trainLabel, 60000, 28 * 28);
}

int main() {
	double** trainData = loadTrainData();
	double* trainLabel = loadTrainLabel();
	double** testData = loadTestData();
	double* testLabel = loadTestLabel();
	
	//testNB(trainData, trainLabel, testData, testLabel);
	//testKNN(trainData, trainLabel, testData, testLabel);
	testSVM(trainData, trainLabel, testData, testLabel);

	freeDouble2D(trainData);
	freeDouble2D(testData);
	delete[] trainLabel;
	delete[] testLabel;

	return 0;
}