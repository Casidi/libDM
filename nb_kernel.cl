void posterior(__global int* point, float* result, int nClass, __global float* piHatLog,
	__global float* thetaHatLog, __global float* oneMinusThetaHatLog, __global int* attribThresh, int dim) {
	for (int i = 0; i < nClass; ++i)
		result[i] = 0;

	for (int c = 0; c < nClass; ++c) {
		result[c] += piHatLog[c];
		for (int j = 0; j < dim; ++j) {
			if (point[j] > attribThresh[j])
				result[c] += thetaHatLog[j*nClass + c];
			else
				result[c] += oneMinusThetaHatLog[j*nClass + c];
		}
	}
}

__kernel void predictBatch(__global int* points, int n, __global int* allResults,
	__global float* piHatLog, __global float* thetaHatLog, __global float* oneMinusThetaHatLog,
	__global int* attribThresh, int dim, int nClass) {
	size_t gid = get_global_id(0);

	//TODO: the 10 is the max number of class this kernel can handle
	float probs[10];
	posterior(points + gid*dim, probs, nClass, piHatLog,
		thetaHatLog, oneMinusThetaHatLog, attribThresh, dim);

	float maxProb = probs[0];
	int result = 0;
	for (int i = 0; i < nClass; ++i) {
		if (probs[i] > maxProb) {
			maxProb = probs[i];
			result = i;
		}
	}

	allResults[gid] = result;
}