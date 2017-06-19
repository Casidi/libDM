#include "svm.h"
#include "svmlib.h"
#include <stddef.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <CL/cl.h>

using std::vector;

extern cl_context context;
extern cl_program program;
extern cl_kernel cl_kernel_predict;
extern cl_command_queue queue;

void print_null(const char *s) {}

SVM::SVM()
{
	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	svm_set_print_string_function(&print_null);
}

void SVM::fit(double **x, double *y, int n, int dim)
{
	if (param.gamma == 0 && dim > 0)
		param.gamma = 1.0 / dim;
	prob.l = n;
	prob.y = (double*)malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++) {
		prob.y[i] = y[i];
	}
	prob.x = (struct svm_node**) malloc(sizeof(svm_node*) * n);

	for (int i = 0; i < n; i++) {
		int cnt = 0;
		for (int j = 0; j < dim; j++) {
			if (x[i][j] != 0) {
				cnt++;
			}
		}
		prob.x[i] = (struct svm_node*) malloc(sizeof(svm_node) * (cnt + 1));
		int k = 0;
		for (int j = 0; j < dim; j++) {
			if (x[i][j] != 0) {
				prob.x[i][k].index = j + 1;
				prob.x[i][k].value = x[i][j];
				k++;
			}
		}
		prob.x[i][k].index = -1;
	}
	const char *error_msg;


	error_msg = svm_check_parameter(&prob, &param);

	if (error_msg)
	{
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}
	model = svm_train(&prob, &param);
	for (int i = 0; i < n; i++) {
		free(prob.x[i]);
	}
	free(prob.x);
}

double SVM::predict(double *x, int n)
{
	int cnt = 0;
	for (int i = 0; i < n; i++) {
		if (x[i] != 0) {
			cnt++;
		}
	}
	struct svm_node *node = (struct svm_node*) malloc(sizeof(struct svm_node) * (cnt + 1));
	int k = 0;
	for (int i = 0; i < n; i++) {
		if (x[i] != 0) {
			node[k].index = i;
			node[k].value = x[i];
			k++;
		}
	}
	node[k].index = -1;
	free(node);
	return svm_predict(model, node);
}

void SVM::predict_multiple(double ** x, int n, int dim, double * label)
{
	ocl_load_model2(model, false);
	size_t num_predict = n;
	vector<int> x_index;
	vector<double> x_value;
	vector<int> head_index;
	vector<double> vtarget_label;

	for (int i = 0; i < n; i++) {
		head_index.push_back(x_index.size());
		for (int j = 0; j < dim; j++) {
			if (x[i][j] != 0) {
				x_index.push_back(j + 1);
				x_value.push_back(x[i][j]);
			}
		}
		x_index.push_back(-1);
		x_value.push_back(0);
	}

	cl_mem cl_x_index = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * x_index.size(), &x_index[0], NULL);
	cl_mem cl_x_value = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * x_value.size(), &x_value[0], NULL);
	cl_mem cl_head_index = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * head_index.size(), &head_index[0], NULL);
	cl_mem cl_predict_label = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * num_predict, NULL, NULL);
	cl_mem cl_kvalue = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * model->l*num_predict, NULL, NULL);
	cl_mem cl_start = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * model->nr_class*num_predict, NULL, NULL);
	cl_mem cl_vote = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * model->nr_class*num_predict, NULL, NULL);
	cl_mem cl_dec_values = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * model->nr_class*(model->nr_class - 1) / 2 * num_predict, NULL, NULL);

	cl_int err;
	err = clSetKernelArg(cl_kernel_predict, 10, sizeof(cl_mem), &cl_x_index);
	err |= clSetKernelArg(cl_kernel_predict, 11, sizeof(cl_mem), &cl_x_value);
	err |= clSetKernelArg(cl_kernel_predict, 12, sizeof(cl_mem), &cl_head_index);
	err |= clSetKernelArg(cl_kernel_predict, 13, sizeof(cl_mem), &cl_predict_label);
	err |= clSetKernelArg(cl_kernel_predict, 14, sizeof(cl_mem), &cl_kvalue);
	err |= clSetKernelArg(cl_kernel_predict, 15, sizeof(cl_mem), &cl_start);
	err |= clSetKernelArg(cl_kernel_predict, 16, sizeof(cl_mem), &cl_vote);
	err |= clSetKernelArg(cl_kernel_predict, 17, sizeof(cl_mem), &cl_dec_values);
	if (err != CL_SUCCESS) {
		puts("kernel argument error");
		exit(0);
	}
	err = clEnqueueNDRangeKernel(queue, cl_kernel_predict, 1, 0, &num_predict, 0, 0, 0, 0);
	if (err == CL_SUCCESS) {
		err = clEnqueueReadBuffer(queue, cl_predict_label, CL_TRUE, 0, sizeof(cl_double) * num_predict, label, 0, 0, NULL);
	}
	else {
		printf("%d\n", err);
		exit(0);
	}
	clReleaseMemObject(cl_x_index);
	clReleaseMemObject(cl_x_value);
	clReleaseMemObject(cl_head_index);
	clReleaseMemObject(cl_predict_label);
	clReleaseMemObject(cl_kvalue);
	clReleaseMemObject(cl_start);
	clReleaseMemObject(cl_vote);
	clReleaseMemObject(cl_dec_values);
}

void SVM::set_rbf(double gamma)
{
	param.kernel_type = RBF;
	param.gamma = gamma;
}

void SVM::set_linear()
{
	param.kernel_type = LINEAR;
}

void SVM::set_polynomial(int degree, double gamma, double coef0)
{
	param.kernel_type = POLY;
	param.degree = degree;
	param.gamma = gamma;
	param.coef0 = coef0;
}

void SVM::set_sigmoid(double gamma, double coef0)
{
	param.kernel_type = SIGMOID;
	param.gamma = gamma;
	param.coef0 = coef0;
}