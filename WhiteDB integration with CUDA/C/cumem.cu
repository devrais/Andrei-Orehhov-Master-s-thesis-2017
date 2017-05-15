#include "../config.h"
#include "stdio.h"
#include "dballoc.h"
#include "dbfeatures.h"
#include "dbmem.h"
#include "dblog.h"

/* ======= Private protos ================ */
__device__ static gint show_memory_error(char *errmsg);

/* ====== Functions ============== */

#ifdef USE_DATABASE_HANDLE
__device__ static void *init_dbhandle(void);
__device__ static void free_dbhandle(void *dbhandle);
#endif

__device__ static void *init_dbhandle() {
	void *dbhandle = malloc(sizeof(db_handle));
	if (!dbhandle) {
		show_memory_error("Failed to allocate the db handle");
		return NULL;
	} else {
		memset(dbhandle, 0, sizeof(db_handle));
	}
#ifdef USE_DBLOG
	if(wg_init_handle_logdata(dbhandle)) {
		free(dbhandle);
		return NULL;
	}
#endif
	return dbhandle;
}

__device__ void* wg_attach_local_cuda_database(gint size) {
	void* shm;
#ifdef USE_DATABASE_HANDLE
	void *dbhandle = init_dbhandle();
	if (!dbhandle)
		return NULL;
#endif
	if (size <= 0)
		size = DEFAULT_MEMDBASE_SIZE;

	shm = (void *) malloc(size);
	if (shm == NULL) {
		show_memory_error("cuda malloc failed (shm)");
		return NULL;
	} else {
		/* key=0 - no shared memory associated */;
#ifdef USE_DATABASE_HANDLE
		((db_handle *) dbhandle)->db = (db_memsegment_header *)shm;
		if (wg_init_db_memsegment(dbhandle, 0, size)) {
#else
			if(wg_init_db_memsegment(shm, 0, size)) {
#endif
			show_memory_error("Cuda Database initialization failed");
			free(shm);
#ifdef USE_DATABASE_HANDLE
			free_dbhandle(dbhandle);
#endif
			return NULL;
		}
	}
#ifdef USE_DATABASE_HANDLE
	return dbhandle;
#else
	return shm;
#endif
}

__device__ static gint show_memory_error(char *errmsg) {
#ifdef WG_NO_ERRPRINT
#else
	printf("wg cuda memory error: %s.\n", errmsg);
#endif
	return -1;
}

__device__  static void free_dbhandle(void *dbhandle) {
#ifdef USE_DBLOG
  wg_cleanup_handle_logdata(dbhandle);
#endif
  free(dbhandle);
}

__device__ void wg_delete_local_cuda_database(void* dbase) {
  if(dbase) {
    void *localmem = dbmemseg(dbase);
    if(localmem)
      free(localmem);
#ifdef USE_DATABASE_HANDLE
    free_dbhandle(dbase);
#endif
  }
}
