#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ColumnBased.h"

int main(void) {
	clock_t start, finish;

	clock_t timeRefactor, timeFillColumn, timeInitCuda,
			timeFirstMallocCudaColumn, timeFirstMallocCudaCount,
			timeFirstMemcpColumnToDevice, timeFirstMemcpCountToDevice,
			timeFirstMemcpCountToHost, //timeSecondMemcpColumnToDevice,
			timeSecondMemcpCountToHost, timeSecondMemcpCountToDevice, timeFirstKernel, timeSecondKernel;

	// We create column based table. It will have 5 rows and 10 million fields
	int (*tableColumn)[C] = new int[R][C];

	// We choose 3rd column
	tableColumn += TargetColumn;

	int count = 0; // initialize host variable for result
	int *h_count = &count;

	int *d_column; // initialize column for cuda
	int *d_count; // initialize cuda variable for result

	//***************** Start Measurements *****************************

	start = clock();
	// Fill selected column with data
	fillColumn(tableColumn);
	finish = clock();
	timeFillColumn = finish - start;

//*************** Initialize cuda *******
	start = clock();
	cudaFree(0);
	finish = clock();
	timeInitCuda = finish - start;

	// Malloc cuda memory for d_column
	start = clock();
	if (cudaMalloc(&d_column, C * sizeof(int)) != cudaSuccess) {
		printf("\nCan't allocate memory for d_column on device !!!");
		return 0;
	}
	finish = clock();
	timeFirstMallocCudaColumn = finish - start;

	// Malloc cuda memory for d_count
	start = clock();
	if (cudaMalloc(&d_count, sizeof(int)) != cudaSuccess) {
		printf("\nCan't allocate memory for d_count on device !!!");
		return 0;
	}
	finish = clock();
	timeFirstMallocCudaCount = finish - start;

//**************** Copy cuda memory to device ***********

	start = clock();
	if (cudaMemcpy(d_column, tableColumn, C * sizeof(int),
			cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("\nFailed to copy d_column to device");
		return 0;
	}
	finish = clock();
	timeFirstMemcpColumnToDevice = finish - start;

	start = clock();
	if (cudaMemcpy(d_count, h_count, sizeof(int), cudaMemcpyHostToDevice)
			!= cudaSuccess) {
		printf("\nFailed to copy d_count to device");
		return 0;
	}
	finish = clock();
	timeFirstMemcpCountToDevice = finish - start;

	// Scale column
	start = clock();
	scaleCudaColumn<<<C / 1024 + 1, 1024>>>(d_column, d_count);
	cudaDeviceSynchronize();
	finish = clock();
	timeFirstKernel = finish - start;

	//**************** Copy cuda memory to host ***********

	start = clock();
	if (cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost)
			!= cudaSuccess) {
		printf("\nFailed to copy d_count to host");
		return 0;
	}
	finish = clock();
	timeFirstMemcpCountToHost = finish - start;
	printf("\nResult = %d", *h_count);

	//**************** Substitute value in column ***********

	//Substitute value in column
	start = clock();
	substituteCudaColumnValue<<<C / 1024 + 1, 1024>>>(d_column);
	finish = clock();
	timeRefactor = finish - start;

	*h_count = 0;

	//**************** Copy cuda memory to device after substitute ***********

	printf("\nAFTER SUBSTITUTION");

	start = clock();
	if (cudaMemcpy(d_count, h_count, sizeof(int), cudaMemcpyHostToDevice)
			!= cudaSuccess) {
		printf("\nFailed to copy d_count to device after substitute");
		return 0;
	}
	finish = clock();
	timeSecondMemcpCountToDevice = finish - start;

	// Scale column second time
	start = clock();
	scaleCudaColumn<<<C / 1024 + 1, 1024>>>(d_column, d_count);
	cudaDeviceSynchronize();
	finish = clock();
	timeSecondKernel = finish - start;

	start = clock();
	if (cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost)
			!= cudaSuccess) {
		printf("\nFailed to copy d_count to host");
		return 0;
	}
	finish = clock();
	timeSecondMemcpCountToHost = finish - start;

	printf("\nSecond Result = %d", *h_count);

	printf("\nIt took me %ld clicks (%f seconds) to fill selected column.\n",
			timeFillColumn, ((float) timeFillColumn) / CLOCKS_PER_SEC);

	printf("\nIt took me %ld clicks (%f seconds) to initialize cuda.\n",
			timeInitCuda, ((float) timeInitCuda) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) to malloc column to device.\n",
			timeFirstMallocCudaColumn,
			((float) timeFirstMallocCudaColumn) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to malloc result variale d_count.\n",
			timeFirstMallocCudaCount,
			((float) timeFirstMallocCudaCount) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) to copy d_column to device.\n",
			timeFirstMemcpColumnToDevice,
			((float) timeFirstMemcpColumnToDevice) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) to copy d_count to device.\n",
			timeFirstMemcpCountToDevice, ((float) timeFirstMemcpCountToDevice) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) to scan column .\n",
			timeFirstKernel, ((float) timeFirstKernel) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to copy d_count result back to host .\n",
			timeFirstMemcpCountToHost,
			((float) timeFirstMemcpCountToHost) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to re factor number %d with number %d.\n",
			timeRefactor, ((float) timeRefactor) / CLOCKS_PER_SEC,
			SubstituteValue, SearchTarget);
	/*printf(
			"\nIt took me %ld clicks (%f seconds) to copy refactored d_column to device.\n",
			timeSecondMemcpColumnToDevice, ((float) timeSecondMemcpColumnToDevice) / CLOCKS_PER_SEC);*/
	printf("\nIt took me %ld clicks (%f seconds) nulled d_count to device.\n",
			timeSecondMemcpCountToDevice, ((float) timeSecondMemcpCountToDevice) / CLOCKS_PER_SEC);
	printf("\nIt took me %ld clicks (%f seconds) for second scan column .\n",
			timeSecondKernel, ((float) timeSecondKernel) / CLOCKS_PER_SEC);
	printf(
			"\nIt took me %ld clicks (%f seconds) to copy refactored result d_count back to host.\n",
			timeSecondMemcpCountToHost, ((float) timeSecondMemcpCountToHost) / CLOCKS_PER_SEC);

	printf("\nTotal execution time = %f seconds",
			((float) timeFillColumn / CLOCKS_PER_SEC
					+ (float) timeInitCuda / CLOCKS_PER_SEC
					+ (float) timeFirstMallocCudaColumn / CLOCKS_PER_SEC
					+ (float) timeFirstMallocCudaCount / CLOCKS_PER_SEC
					+ (float) timeFirstMemcpColumnToDevice / CLOCKS_PER_SEC
					+ (float) timeFirstMemcpCountToDevice / CLOCKS_PER_SEC
					+ (float) timeFirstKernel / CLOCKS_PER_SEC
					+ (float) timeFirstMemcpCountToHost / CLOCKS_PER_SEC
					+ (float) timeRefactor / CLOCKS_PER_SEC
					//+ (float) timeSecondMemcpColumnToDevice / CLOCKS_PER_SEC
					+ (float) timeSecondMemcpCountToDevice / CLOCKS_PER_SEC
					+ (float) timeSecondKernel / CLOCKS_PER_SEC
					+ (float) timeSecondMemcpCountToHost / CLOCKS_PER_SEC));
	return 0;
}
