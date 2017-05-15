#include <stdio.h>

#include "ColumnBased.h"

int main(void) {

// timeFillColumn, timeFirstScale, timeSecondScale;
	clock_t start, finish;
	clock_t timeInitCuda, timeMallocTable, timeMallocCount, timeRefactor,
			timeFillColumn;
	clock_t timeFirstScale, timeSecondScale;

	int (*d_tableColumn)[C]; // declare pointer for table column on host
	int *d_count; // declare pointer to variable that stores counting result

	//***************Initialize cuda *********************

	start = clock();
	cudaFree(0);
	finish = clock();
	timeInitCuda = finish = start;

	//*****************Malloc data in unified cuda memory*****************************

	// malloc memory for column on device

	start = clock();
	cudaMallocManaged(&d_tableColumn, R * C * sizeof(int));
	finish = clock();
	timeMallocTable = finish = start;

	start = clock();
	cudaMallocManaged(&d_count, sizeof(int));
	finish = clock();
	timeMallocCount = finish = start;

	*d_count = 0;

	// We fill chosen column with data on host using unified memory and filled it with data
	start = clock();
	fillColumn(d_tableColumn);
	finish = clock();
	timeFillColumn = finish - start;

	// Scale column first time
	start = clock();
	scaleCudaColumn<<<C / 1024 + 1, 1024>>>(d_tableColumn, d_count);
	cudaDeviceSynchronize();
	finish = clock();
	timeFirstScale = finish - start;

	// Show first result
	printf("\n First Result = %d", *d_count);

	// Substitute number 7 with 10 on cuda
	start = clock();
	substituteCudaColumnValue<<<C / 1024 + 1, 1024>>>(d_tableColumn);
	cudaDeviceSynchronize();
	finish = clock();
	timeRefactor = finish - start;

	// Null result variable
	*d_count = 0;

	// Scale column second time
	start = clock();
	scaleCudaColumn<<<C / 1024 + 1, 1024>>>(d_tableColumn, d_count);
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

	printf("\nIt took me %ld clicks (%f seconds) to fill selected column.\n",
			timeFillColumn, ((float) timeFillColumn) / CLOCKS_PER_SEC);

	printf("\nIt took me %ld clicks (%f seconds) to scan column .\n",
			timeFirstScale, ((float) timeFirstScale) / CLOCKS_PER_SEC);

	printf(
			"\nIt took me %ld clicks (%f seconds) to refactor number %d with number %d.\n",
			timeRefactor, ((float) timeRefactor) / CLOCKS_PER_SEC,
			SubstituteValue, SearchTarget);
	printf("\nIt took me %ld clicks (%f seconds) for second scan column .\n",
			timeSecondScale, ((float) timeSecondScale) / CLOCKS_PER_SEC);

	printf("\nTotal execution time = %f seconds",
			+(float) timeInitCuda / CLOCKS_PER_SEC
					+ (float) timeMallocTable / CLOCKS_PER_SEC
					+ (float) timeMallocCount / CLOCKS_PER_SEC
					+ (float) timeFillColumn / CLOCKS_PER_SEC
					+ (float) timeFirstScale / CLOCKS_PER_SEC
					+ (float) timeRefactor / CLOCKS_PER_SEC
					+ (float) timeSecondScale / CLOCKS_PER_SEC);

	return 0;
}

