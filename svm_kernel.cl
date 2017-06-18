double dot2(int i, int j, int px, int py, __global int *index, __global double *value)
{
	double sum = 0;
	while (index[px] != -1 && index[py] != -1)
	{
		if (index[px] == index[py])
		{
			sum += value[px] * value[py];
			++px;
			++py;
		}
		else
		{
			if (index[px] > index[py])
				++py;
			else
				++px;
		}
	}
	return sum;
}
double dot3(int i, int j, int px, int py, __global int *index, __global double *value,
		__local int *lindex, __local double *lvalue)
{
	double sum = 0;
	while (lindex[px] != -1 && index[py] != -1)
	{
		if (lindex[px] == index[py])
		{
			sum += lvalue[px] * value[py];
			++px;
			++py;
		}
		else
		{
			if (lindex[px] > index[py])
				++py;
			else
				++px;
		}
	}
	return sum;
}
double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

__kernel void kernel_linear(int i, __global char *y, 
			__global int *head_index, __global int *x_index, __global double *x_value,
			__global float *data, int offset)
{
	int j = get_global_id(0);
	if (j < offset) return;
	data[j] = y[i] * y[j] * dot2(i, j, head_index[i], head_index[j], x_index, x_value);
	return;
}
__kernel void kernel_rbf(int i, __global char *y,
				__global int *head_index, __global int *x_index, __global double *x_value, 
				__global float *data, int offset, __global double *x_square, double gamma)
{
	int j = get_global_id(0);
	if (j < offset) return;	
	double ker = exp(-gamma*((double)x_square[i]+(double)x_square[j]-2.0*dot2(i, j, head_index[i], head_index[j], x_index, x_value)));
	data[j] = (float)(y[i] * y[j] * ker);
}

__kernel void kernel_rbf_local(int i, __global char *y,
				__global int *head_index, __global int *x_index, __global double *x_value, 
				__global float *data, int offset, __global double *x_square, double gamma,
				__local int *lindex, __local double *lvalue, int len)
{
	int j = get_global_id(0);
	int tid = get_local_id(0);
	int local_size = get_local_size(0);
	int k2=head_index[i];
	for(int k=tid;k<len;k+=local_size){
		lindex[k] = x_index[k+k2];
		lvalue[k] = x_value[k+k2];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	//if (j < offset || j >= work_size) return;	
	if (j < offset) return;	
	
	//double ker = exp(-gamma*((double)x_square[i]+(double)x_square[j]-2.0*dot2(i, j, head_index[i], head_index[j], x_index, x_value)));
	double ker = exp(-gamma*((double)x_square[i]+(double)x_square[j]-2.0*dot3(i, j, 0, head_index[j], x_index, x_value, lindex, lvalue)));
	data[j] = (float)(y[i] * y[j] * ker);
}


__kernel void kernel_poly(int i, __global char *y,
				__global int *head_index, __global int *x_index, __global double *x_value, 
				__global float *data, int offset, double gamma, double coef0, int degree)
{
	int j = get_global_id(0);
	if (j < offset) return;
	data[j] = y[i] * y[j] * powi(gamma*dot2(i, j, head_index[i], head_index[j], x_index, x_value)+coef0,degree);
}

__kernel void kernel_poly_local(int i, __global char *y,
				__global int *head_index, __global int *x_index, __global double *x_value, 
				__global float *data, int offset, double gamma, double coef0, int degree,
				__local int *lindex, __local double *lvalue, int len)
{
	int j = get_global_id(0);
	int tid = get_local_id(0);
	int local_size = get_local_size(0);
	int k2=head_index[i];
	for(int k=tid;k<len;k+=local_size){
		lindex[k] = x_index[k+k2];
		lvalue[k] = x_value[k+k2];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (j < offset) return;
	data[j] = y[i] * y[j] * powi(gamma*dot3(i, j, 0, head_index[j], x_index, x_value, lindex, lvalue)+coef0,degree);
	//data[j] = y[i] * y[j] * powi(gamma*dot2(i, j, head_index[i], head_index[j], x_index, x_value)+coef0,degree);
}
__kernel void kernel_sigmoid(int i, __global char *y,
				__global int *head_index, __global int *x_index, __global double *x_value, 
				__global float *data, int offset, double gamma, double coef0)
{
	int j = get_global_id(0);
	if (j < offset) return;
	data[j] = y[i] * y[j] * tanh(gamma*dot2(i, j, head_index[i], head_index[j], x_index, x_value)+coef0);
}
__kernel void kernel_sigmoid_local(int i, __global char *y,
				__global int *head_index, __global int *x_index, __global double *x_value, 
				__global float *data, int offset, double gamma, double coef0,
				__local int *lindex, __local double *lvalue, int len)
{
	int j = get_global_id(0);
	int tid = get_local_id(0);
	int local_size = get_local_size(0);
	int k2=head_index[i];
	for(int k=tid;k<len;k+=local_size){
		lindex[k] = x_index[k+k2];
		lvalue[k] = x_value[k+k2];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (j < offset) return;
	data[j] = y[i] * y[j] * tanh(gamma*dot3(i, j, 0, head_index[j], x_index, x_value, lindex, lvalue)+coef0);
	//data[j] = y[i] * y[j] * tanh(gamma*dot2(i, j, head_index[i], head_index[j], x_index, x_value)+coef0);
}


__kernel void predict_kernel(__global int *x_index, __global double *x_value,
			__global int *head_index, __global int *sv_index, __global double *sv_value,
			__global double *data, double gamma)
{
	int i=get_global_id(0);
	double sum = 0;
	int xi=0,yi=head_index[i];
	while (x_index[xi] != -1 && sv_index[yi] != -1)
	{
		if(x_index[xi]==sv_index[yi])
		{
			double d = x_value[xi] - sv_value[yi];
			sum += d * d;
			++xi;
			++yi;
		}
		else
		{
			if(x_index[xi]>sv_index[yi])
			{
				sum+=sv_value[yi]*sv_value[yi];
				++yi;
			}
			else
			{
				sum+=x_value[xi]*x_value[xi];
				++xi;
			}
		}
	}
	while(x_index[xi]!=-1)
	{
		sum+=x_value[xi]*x_value[xi];
		++xi;
	}
	while(sv_index[yi]!=-1)
	{
		sum+=sv_value[yi]*sv_value[yi];
		++yi;
	}
	data[i]=exp(-gamma*sum);
}

double predict_one_kernel(__global int *x_index, __global double *x_value,
			__global int *sv_index, __global double *sv_value,
			double gamma)
{
	//int i=get_global_id(0);
	double sum = 0;
	int xi=0,yi=0;
	while (x_index[xi] != -1 && sv_index[yi] != -1)
	{
		if(x_index[xi]==sv_index[yi])
		{
			double d = x_value[xi] - sv_value[yi];
			sum += d * d;
			++xi;
			++yi;
		}
		else
		{
			if(x_index[xi]>sv_index[yi])
			{
				sum+=sv_value[yi]*sv_value[yi];
				++yi;
			}
			else
			{
				sum+=x_value[xi]*x_value[xi];
				++xi;
			}
		}
	}
	while(x_index[xi]!=-1)
	{
		sum+=x_value[xi]*x_value[xi];
		++xi;
	}
	while(sv_index[yi]!=-1)
	{
		sum+=sv_value[yi]*sv_value[yi];
		++yi;
	}
	return exp(-gamma*sum);
	//data[i]=exp(-gamma*sum);
}


/* sv_coef[nr_class-1][l] */
int svm_predict(int id, int nr_class, int l, __global double *sv_coef, __global int *nSV,
			__global int *svhead_index, __global int *sv_index, __global double *sv_value,
			__global double *rho, double gamma,
			__global int *x_index, __global double *x_value, __global int *head_index,
			__global double *kvalue, __global int *start, __global int *vote, __global double *dec_values)
{
	int i;
	//if(id==10)printf("%d\n",*(x_index+head_index[id]+1));
	
	/* nested parallel */
	/*size_t ll = l;
	printf("%u\n",ll);
	ndrange_t ndrange = ndrange_1D(ll);

	void (^block)(void) = ^{predict_kernel(x_index,x_value,head_index,sv_index,sv_value,kvalue+id*l,gamma);};
	int err = enqueue_kernel(get_default_queue(), 
				CLK_ENQUEUE_FLAGS_NO_WAIT,
				ndrange,
				block);
				
	printf("err %d\n",err);*/
	/* nested parallel */
	
	for(i=0;i<l;i++){
		
		kvalue[id*l+i] = predict_one_kernel(x_index + head_index[id], x_value + head_index[id], sv_index + svhead_index[i], sv_value + svhead_index[i], gamma);
	}
	
	//int *start = Malloc(int,nr_class);
	/*start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+nSV[i-1];*/
	int base = id*nr_class;
	start[base] = 0;
	for(i=1;i<nr_class;i++)
		start[base+i] = start[base+i-1]+nSV[i-1];
	//int *vote = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		vote[base+i] = 0;
	int decbase = id*(nr_class*(nr_class-1)/2);
	int p=0;
	for(i=0;i<nr_class;i++)
		for(int j=i+1;j<nr_class;j++)
		{
			double sum = 0;
			int si = start[base+i];
			int sj = start[base+j];
			//int ci = model->nSV[i];
			//int cj = model->nSV[j];
			int ci = nSV[i];
			int cj = nSV[j];
			int k;
			/*double *coef1 = model->sv_coef[j-1];
			double *coef2 = model->sv_coef[i];
			for(k=0;k<ci;k++)
				sum += coef1[si+k] * kvalue[si+k];
			for(k=0;k<cj;k++)
				sum += coef2[sj+k] * kvalue[sj+k];*/
			int coef1 = (j-1) * l;
			int coef2 = i * l;
			for(k=0;k<ci;k++){
				sum += sv_coef[coef1+si+k] * kvalue[id*l+si+k];
				//if(id==0)printf("%f %f\n",sv_coef[coef1+si+k],kvalue[id*l+si+k]);
			}
			
				
			for(k=0;k<cj;k++)
				sum += sv_coef[coef2+sj+k] * kvalue[id*l+sj+k];
			
			sum -= rho[p];
			
			//if(id<=10)printf("%d %f\n",id,sum);
			
			dec_values[decbase+p] = sum;
			//if(id==0)printf("%f\n",sum);

			if(dec_values[decbase+p] > 0)
				++vote[base+i];
			else
				++vote[base+j];
			p++;
		}

	int vote_max_idx = 0;
	for(i=1;i<nr_class;i++)
		if(vote[base+i] > vote[base+vote_max_idx]){
			//printf("%d %d\n",vote[base+i],vote[base+vote_max_idx]);
			vote_max_idx = i;
		}
	return vote_max_idx;
}
__kernel void predict(int nr_class, int l, __global double *sv_coef, __global int *nSV,
				__global int *svhead_index, __global int *sv_index, __global double *sv_value,
				__global double *rho, __global int *label, double gamma, 
				__global int *x_index, __global double *x_value, __global int *head_index, __global double *predict_label,
				__global double *kvalue, __global int *start, __global int *vote, __global double *dec_values)
{
	int i = get_global_id(0);
	predict_label[i] = label[svm_predict(i,nr_class,l,sv_coef,nSV,svhead_index,sv_index,sv_value,rho,gamma,x_index,x_value,head_index,kvalue,start,vote,dec_values)];
}