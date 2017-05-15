#include <stdio.h>

#include "RowBased.h"

/*__global__ void test(){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < 15){
		int stride = blockDim.x * gridDim.x;
		printf("\n %d = %d * %d", stride,blockDim.x ,gridDim.x);
		printf("\n gridDim.x = %d", gridDim.x);
	}
}*/

int main(void) {

// timeFillColumn, timeFirstScale, timeSecondScale;
	clock_t start, finish;
	clock_t timeInitCuda, timeMallocTable, timeMallocCount, timeRefactor,
			timeFillRows;
	clock_t timeFirstScale, timeSecondScale, timeSetIncrementorToZero1,
			timeSetIncrementorToZero2;

	int (*d_tableRow)[C]; // declare pointer for table column on host
	int *d_count; // declare pointer to variable that stores counting result

	//***************Initialize cuda *********************

	start = clock();
	cudaFree(0);
	finish = clock();
	timeInitCuda = finish = start;

	/*test<<<R / 1024 + 1, 1024>>>();
	cudaDeviceSynchronize();
	return 0;*/


	//*****************Malloc data in unified cuda memory*****************************

	// malloc memory for column on device

	start = clock();
	cudaMallocManaged(&d_tableRow, R * C * sizeof(int));
	finish = clock();
	timeMallocTable = finish = start;

	start = clock();
	cudaMallocManaged(&d_count, sizeof(int));
	finish = clock();
	timeMallocCount = finish = start;

	*d_count = 0;

	// We fill chosen column with data on host using unified memory and filled it with data
	start = clock();
	fillField(d_tableRow);
	finish = clock();
	timeFillRows = finish - start;

	// Scale rows first time
	start = clock();
	scaleCudaRows<<<R / 1024 + 1, 1024>>>(d_tableRow, d_count);
	cudaDeviceSynchronize();
	finish = clock();
	timeFirstScale = finish - start;

	// Show first result
	printf("\n First Result = %d", *d_count);

	// Set incrementor to zero
	start = clock();
	resetIncrementor<<<1, 1>>>();
	cudaDeviceSynchronize();
	finish = clock();
	timeSetIncrementorToZero1 = finish - start;

	// Substitute number 7 with 10 on cuda
	start = clock();
	substituteFieldCudaValue<<<R / 1024 + 1, 1024>>>(d_tableRow);
	cudaDeviceSynchronize();
	finish = clock();
	timeRefactor = finish - start;

	printf("\nAFTER SUBSTITUTION");

	*d_count = 0;

	// Set incrementor to zero
	start = clock();
	resetIncrementor<<<1, 1>>>();
	cudaDeviceSynchronize();
	finish = clock();
	timeSetIncrementorToZero2 = finish - start;

	// Scale rows second time
	start = clock();
	scaleCudaRows<<<R / 1024 + 1, 1024>>>(d_tableRow, d_count);
	cudaDeviceSynchronize();
	finish = clock();
	timeSecondScale = finish - start;

	//Show second result
	printf("\n Second Result = %d", *d_count);

	printf("\nIt took me %ld clicks (%f seconds) to initialize cuda.\n",
			timeInitCuda, ((float) timeInitCuda) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) to malloc table to device.\n",
			timeMallocTable, ((float) timeMallocTable) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to malloc result variale d_count.\n",
			timeMallocCount, ((float) timeMallocCount) / CLOCKS_PER_SEC);

	printf("\nIt took me %ld clicks (%f seconds) to fill selected rows.\n",
			timeFillRows, ((float) timeFillRows) / CLOCKS_PER_SEC);

	printf("\nIt took me %ld clicks (%f seconds) to scan rows .\n",
			timeFirstScale, ((float) timeFirstScale) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to set global incrementor value to zero.\n",
			timeSetIncrementorToZero1,
			((float) timeSetIncrementorToZero1) / CLOCKS_PER_SEC);

	printf(
			"\nIt took me %ld clicks (%f seconds) to refactor number %d with number %d.\n",
			timeRefactor, ((float) timeRefactor) / CLOCKS_PER_SEC,
			SubstituteValue, SearchTarget);
	printf(
			"\nIt took me %ld clicks (%f seconds) to set global incrementor value to zero.\n",
			timeSetIncrementorToZero2,
			((float) timeSetIncrementorToZero2) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) for second scan rows .\n",
			timeSecondScale, ((float) timeSecondScale) / CLOCKS_PER_SEC);

	printf("\nTotal execution time = %f seconds",
			+(float) timeInitCuda / CLOCKS_PER_SEC
					+ (float) timeMallocTable / CLOCKS_PER_SEC
					+ (float) timeMallocCount / CLOCKS_PER_SEC
					+ (float) timeFillRows / CLOCKS_PER_SEC
					+ (float) timeFirstScale / CLOCKS_PER_SEC
					+ (float) timeSetIncrementorToZero1 / CLOCKS_PER_SEC
					+ (float) timeRefactor / CLOCKS_PER_SEC
					+ (float) timeSetIncrementorToZero2 / CLOCKS_PER_SEC
					+ (float) timeSecondScale / CLOCKS_PER_SEC);

	return 0;
}

