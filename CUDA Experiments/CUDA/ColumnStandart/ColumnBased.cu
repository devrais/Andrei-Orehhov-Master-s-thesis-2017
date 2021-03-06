#include "ColumnBased.h"
#include <stdio.h>

// Do a search in table using cuda
__global__
void scaleCudaColumn(int *d_column, int *d_count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < C) {
		if (d_column[id] == SearchTarget) {
			atomicAdd(d_count, 1);
		}
	}
}

// Function will substitute chosen value by SearchTarget
__global__ void substituteCudaColumnValue(int *d_column) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < C) {
			if (d_column[id] == SubstituteValue) {
				d_column[id] = SearchTarget;
			}
		}
}

/* Every row will have its second field with data
 fillColumn will add numbers from 1 to 10
 */
void fillColumn(int (*tableColumn)[C]) {

    int i = 0;
    int current = StartValue;
    int *column = (int *) tableColumn;

    while (i < C) {
        if (current > EndValue)
            current = StartValue;

        column[i] = current;
        i++;
        current++;
    }
}

