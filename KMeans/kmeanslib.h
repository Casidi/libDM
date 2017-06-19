#ifndef KMEANSLIB
#define KMEANSLIB
#include <CL/cl.h>
class KMeans
{
public:
	KMeans(int n_clusters);
	void fit(double **x, int n, int dim);
	double predict(double *x, int dim);
	void predict_multiple(double **x, int n, int dim, double *label);
	double get_label(int i);
private:
	int find_nearest_cluster(int, int, double*);
	void ocl_kmeans(double**, int, int, int, double);
	cl_program load_program(cl_context context, const char* filename, cl_device_id device);
	double seq_euclid_dist_2(int, double*, double*);
	int seq_find_nearest_cluster(int, int, double*, double**);
	void seq_kmeans(double**, int, int, int, double);
	
	double **clusters;
	int n_clusters;
	int *membership;
	bool ocl;
};
#endif
