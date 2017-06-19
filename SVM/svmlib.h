#ifndef SVMLIB
#define SVMLIB
#include "svm.h"
#include "Classify.h"
class SVM: public Classify
{
public:
	SVM();
	/**
	x: train data<br>
	y: label<br>
	n: number of training datas<br>
	dim: number of features<br>
	*/
	void fit(double **x, double *y, int n, int dim);
	/**
	x: predict data
	dim: number of features
	*/
	double predict(double *x, int dim);
	void predict_multiple(double **x, int n, int dim, double *label);
	void set_rbf(double gamma=0);
	void set_linear();
	void set_polynomial(int degree = 3, double gamma = 0, double coef0 = 0);
	void set_sigmoid(double gamma = 0, double coef0 = 0);
private:
	struct svm_parameter param;
	struct svm_problem prob;
	struct svm_model *model;
};
#endif