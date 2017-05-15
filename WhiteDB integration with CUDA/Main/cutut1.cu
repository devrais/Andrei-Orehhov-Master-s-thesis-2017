#include "../../../C/dballoc.h"
#include "../../../C/dbmem.h"
#include "../../../C/dbquery.h"
#include "../../../C/dbdata.h"

#include "stdio.h"

#define MAX 10
#define R 1000
#define C 5
#define FIELD 2

__global__
void initialize_db_on_cuda(void *d_db, gint *d_size) {

	printf("\nInit");
	if (threadIdx.x == 0) {
		d_db = wg_attach_local_cuda_database(*d_size);
	}
}

__global__
void delete_db_on_cuda(void *d_db) {

	if (threadIdx.x == 0) {
		wg_delete_local_cuda_database(d_db);
	}
}

__global__
void GpuCount(void *d_db, int *d_count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	wg_int target = 10;

	if (id < R) {
		void *rec = wg_cuda_find_record_int(d_db, 0, WG_COND_EQUAL, id, NULL);
		wg_int value = wg_cuda_decode_int(d_db,
				wg_cuda_get_field(d_db, rec, 2));
		if (value == target) {
			atomicAdd(d_count, 1);
		}
	}
}

__global__ void createTable(void *d_db, int *d_value) {
	wg_int enc;
	void *rec;

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < R) {
		rec = wg_cuda_create_record(d_db, C);
		if (!rec) {
			printf("\nFailed to create database - not enough memory");
			wg_delete_local_cuda_database(d_db);
		}
		enc = wg_cuda_encode_int(d_db, *d_value);
		wg_cuda_set_field(d_db, rec, 2, enc);
	}
}

int main() {

	void *d_db;
	gint size = 2000000;
	gint *h_size = &size;
	void *h_db = malloc(sizeof(db_handle));

	int value = 1;
	int count = 0;

	int *h_count = &count;
	int *h_value = &value;

	int *d_value;
	gint *d_size;
	int *d_count;

	if (cudaMalloc((void **) &d_db, sizeof(db_handle)) != cudaSuccess) {
		printf("Failed to allocate memory for db_handle to the GPU");
		return 0;
	}


	if (cudaMalloc((void **) &d_size, sizeof(gint)) != cudaSuccess) {
		printf("Failed to allocate memory for d_size to the GPU");
		return 0;
	}


	if (cudaMemcpy(d_size, h_size, sizeof(gint), cudaMemcpyHostToDevice)
			!= cudaSuccess) {
		printf("\nFailed to copy size pointer to GPU");
		return 0;
	}

	printf("\nWe are here before init");
	initialize_db_on_cuda<<<1, 32>>>(d_db, d_size);
	cudaDeviceSynchronize();

	if (cudaMalloc((void **) &d_value, sizeof(int)) != cudaSuccess) {
		printf("Failed to allocate memory for d_value to the GPU");
		return 0;
	}

	if (cudaMemcpy(d_value, h_value, sizeof(int), cudaMemcpyHostToDevice)
			!= cudaSuccess) {
		printf("\nFailed to copy d_value pointer to GPU");
		return 0;
	}

	createTable<<<1, 32>>>(d_db, d_value);
	cudaDeviceSynchronize();

	GpuCount<<<1,32>>>(d_db, d_count);
        cudaDeviceSynchronize();

	printf("\nThe end");
	return 0;

}

