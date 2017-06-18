#ifndef CLASSIFY
#define CLASSIFY

class Classify
{
public:
	virtual void fit(double **x, double *y, int n, int dim) = 0;
	virtual double predict(double *x, int dim) = 0;
	virtual void predict_multiple(double **x, int n, int dim, double *label) = 0;
};

#endif