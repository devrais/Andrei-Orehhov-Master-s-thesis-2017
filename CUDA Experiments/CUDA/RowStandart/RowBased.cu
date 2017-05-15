#include "RowBased.h"
#include <stdio.h>

__device__ int incrementor = 0;

// Do a search in matrix using cuda
__global__
void scaleCudaRow(int (*d_tableRow)[C], int *d_count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < R) {
		d_tableRow += atomicAdd(&incrementor, 1);
		int *row = (int *) d_tableRow;
		if (row[TargetField] == SearchTarget) {
			atomicAdd(d_count, 1);
		}
	}
}

__global__ void resetIncrementor() {
	incrementor = 0;
}

__global__ void substituteFieldCudaValue(int (*d_tableRow)[C]) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < R) {
		d_tableRow += atomicAdd(&incrementor, 1);
		int *row = (int *) d_tableRow;
		if (row[TargetField] == SubstituteValue) {
			row[TargetField] = SearchTarget;
		}
	}

}

/* Every row will have its second field with data
 fillField will add numbers from 1 to 10
 */
void fillField(int (*tableRow)[C]) {

	int i = 0;
	int current = StartValue;
	int *row;

	while (i < R) {
		row = (int *) tableRow;
		tableRow++;
		if (current > EndValue)
			current = StartValue;
		row[TargetField] = current;
		i++;
		current++;
	}
}

