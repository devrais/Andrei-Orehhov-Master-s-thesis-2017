#include <stdio.h>
#include <time.h>

#include "RowBased.h"

int main(void) {

	clock_t timeRefactor, timeFillRows;
	clock_t start, finish;

	clock_t timeInitCuda, timeFirstMallocCudaRows, timeFirstMallocCudaCount,
			timeFirstMemcpRowsToDevice, timeFirstMemcpCountToDevice,
			timeFirstMemcpCountToHost, timeSecondMemcpRowsToDevice,
			timeSecondMemcpCountToDevice, timeSetIncrementorToZero1,
			timeSetIncrementorToZero2, timeSecondMemcpCountToHost;

	clock_t timeFirstKernel, timeSecondKernel;

	// We create row based table. It will have 10 million rows and 5 fields
	int (*h_tableRow)[C] = new int[R][C];

	int count = 0;

	int *h_count = &count; // initialize host variable for result

	int (*d_tableRow)[C]; // initialize cuda pointer for every row
	int *d_count; // initialize cuda variable for result

	//***************** Start Measurements *****************************

	start = clock();
	// Fill selected column with data
	fillField(h_tableRow);
	finish = clock();
	timeFillRows = finish - start;

//*************** Initialize cuda *******
	start = clock();
	cudaFree(0);
	finish = clock();
	timeInitCuda = finish - start;

	// Malloc cuda memory for rows
	start = clock();
	if (cudaMalloc((void**) &d_tableRow, R * C * sizeof(int)) != cudaSuccess) {
		printf("\nCan't allocate memory for d_tableRow on device !!!");
		return 0;
	}
	finish = clock();
	timeFirstMallocCudaRows = finish - start;

	// Malloc cuda memory for d_count
	start = clock();
	if (cudaMalloc((void**) &d_count, sizeof(int)) != cudaSuccess) {
		printf("\nCan't allocate memory for d_count on device !!!");
		return 0;
	}
	finish = clock();
	timeFirstMallocCudaCount = finish - start;

//**************** Copy cuda memory to device ***********

	start = clock();
	if (cudaMemcpy(d_tableRow, h_tableRow, R * C * sizeof(int),
			cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("\nFailed to copy d_tableRow to device");
		return 0;
	}
	finish = clock();
	timeFirstMemcpRowsToDevice = finish - start;

	start = clock();
	if (cudaMemcpy(d_count, h_count, sizeof(int), cudaMemcpyHostToDevice)
			!= cudaSuccess) {
		printf("\nFailed to copy d_count to device");
		return 0;
	}
	finish = clock();
	timeFirstMemcpCountToDevice = finish - start;

	start = clock();
	scaleCudaRow<<<R / 1024 + 1, 1024>>>(d_tableRow, d_count);
	cudaDeviceSynchronize();
	finish = clock();
	timeFirstKernel = finish - start;

	//**************** Copy cuda memory back to host ***********

	start = clock();
	if (cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost)
			!= cudaSuccess) {
		printf("\nFailed to copy d_count to host");
		cudaFree(d_tableRow);
		cudaFree(d_count);
		return 0;
	}
	finish = clock();
	timeFirstMemcpCountToHost = finish - start;

	printf("\nFirst Result = %d", *h_count);

	//**************** Substitute value in column ***********

	start = clock();
	resetIncrementor<<<1, 1>>>();
	cudaDeviceSynchronize();
	finish = clock();
	timeSetIncrementorToZero2 = finish - start;

	start = clock();
	substituteFieldCudaValue<<<R / 1024 + 1, 1024>>>(d_tableRow);
	cudaDeviceSynchronize();
	finish = clock();
	timeRefactor = finish - start;

	*h_count = 0;

	start = clock();
	resetIncrementor<<<1, 1>>>();
	cudaDeviceSynchronize();
	finish = clock();
	timeSetIncrementorToZero1 = finish - start;

	//**************** Copy cuda memory to device after substitute ***********

	printf("\nAFTER SUBSTITUTION");

	start = clock();
	if (cudaMemcpy(d_count, h_count, sizeof(int), cudaMemcpyHostToDevice)
			!= cudaSuccess) {
		printf("\nFailed to copy d_count to device after substitute///////");
		return 0;
	}
	finish = clock();
	timeSecondMemcpCountToDevice = finish - start;

	start = clock();
	scaleCudaRow<<<R / 1024 + 1, 1024>>>(d_tableRow, d_count);
	cudaDeviceSynchronize();
	finish = clock();
	timeSecondKernel = finish - start;

	if (cudaMemcpy(h_tableRow, d_tableRow, R * C * sizeof(int),
			cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("\nFailed to copy d_count to device after substitute");
		return 0;
	}

	//**************** Copy cuda memory back to host ***********
	start = clock();
	if (cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost)
			!= cudaSuccess) {
		printf("\nFailed to copy d_count to host/////////////////////");
		return 0;
	}
	finish = clock();
	timeSecondMemcpCountToHost = finish - start;

	printf("\nSecond Result = %d", *h_count);

	printf("\nIt took me %ld clicks (%f seconds) to fill selected rows.\n",
			timeFillRows, ((float) timeFillRows) / CLOCKS_PER_SEC);

	printf("\nIt took me %ld clicks (%f seconds) to initialize cuda.\n",
			timeInitCuda, ((float) timeInitCuda) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) to malloc rows to device.\n",
			timeFirstMallocCudaRows,
			((float) timeFirstMallocCudaRows) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to malloc result variale d_count.\n",
			timeFirstMallocCudaCount,
			((float) timeFirstMallocCudaCount) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to copy d_tableRows to device.\n",
			timeFirstMemcpRowsToDevice,
			((float) timeFirstMemcpRowsToDevice) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) to copy d_count to device.\n",
			timeFirstMemcpCountToDevice,
			((float) timeFirstMemcpCountToDevice) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) to scan rows in cuda.\n",
			timeFirstKernel, ((float) timeFirstKernel) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to copy d_count result back to host .\n",
			timeFirstMemcpCountToHost,
			((float) timeFirstMemcpCountToHost) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to set global incrementor value to zero.\n",
			timeSetIncrementorToZero2,
			((float) timeSetIncrementorToZero2) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to re factor number %d with number %d.\n",
			timeRefactor, ((float) timeRefactor) / CLOCKS_PER_SEC,
			SubstituteValue, SearchTarget);
	printf(
			"\nIt took me %ld clicks (%f seconds) to set global incrementor value to zero.\n",
			timeSetIncrementorToZero1,
			((float) timeSetIncrementorToZero1) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds)copy nulled d_count to device.\n",
			timeSecondMemcpCountToDevice,
			((float) timeSecondMemcpCountToDevice) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to scan rows after re factor .\n",
			timeSecondKernel, ((float) timeSecondKernel) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to copy re factored result d_count back to host.\n",
			timeSecondMemcpCountToHost,
			((float) timeSecondMemcpCountToHost) / CLOCKS_PER_SEC);

	printf("\nTotal execution time = %f seconds",
			((float) timeFillRows / CLOCKS_PER_SEC
					+ (float) timeInitCuda / CLOCKS_PER_SEC
					+ (float) timeFirstMallocCudaRows / CLOCKS_PER_SEC
					+ (float) timeFirstMallocCudaCount / CLOCKS_PER_SEC
					+ (float) timeFirstMemcpRowsToDevice / CLOCKS_PER_SEC
					+ (float) timeFirstMemcpCountToDevice / CLOCKS_PER_SEC
					+ (float) timeFirstKernel / CLOCKS_PER_SEC
					+ (float) timeFirstMemcpCountToHost / CLOCKS_PER_SEC
					+ (float) timeSetIncrementorToZero2 / CLOCKS_PER_SEC
					+ (float) timeRefactor / CLOCKS_PER_SEC
					+ (float) timeSetIncrementorToZero1 / CLOCKS_PER_SEC
					+ (float) timeSecondMemcpCountToDevice / CLOCKS_PER_SEC
					+ (float) timeSecondKernel / CLOCKS_PER_SEC
					+ (float) timeSecondMemcpCountToHost / CLOCKS_PER_SEC));

	return 0;
}
