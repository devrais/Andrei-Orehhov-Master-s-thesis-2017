#include "../config.h"
#include "dballoc.h"
#include "dblock.h"
#include "stdio.h"

#if (LOCK_PROTO==TFQUEUE)
#ifdef __linux__
#include <linux/futex.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/errno.h>
#endif
#endif

/* ======= Private protos ================ */

__device__ static gint show_lock_error(void *db, char *errmsg);

__device__
gint wg_init_locks(void * db) {
#if (LOCK_PROTO==TFQUEUE)
  gint i, chunk_wall;
  lock_queue_node *tmp = NULL;
#endif
  db_memsegment_header* dbh;

#ifdef CHECK
  if (!dbcheck(db) && !dbcheckinit(db)) {
    show_lock_error(db, "Invalid database pointer in wg_init_locks");
    return -1;
  }
#endif
  dbh = dbmemsegh(db);

#if (LOCK_PROTO==TFQUEUE)
  chunk_wall = dbh->locks.storage + dbh->locks.max_nodes*SYN_VAR_PADDING;

  for(i=dbh->locks.storage; i<chunk_wall; ) {
    tmp = (lock_queue_node *) offsettoptr(db, i);
    i+=SYN_VAR_PADDING;
    tmp->next_cell = i; /* offset of next cell */
  }
  tmp->next_cell=0; /* last node */

  /* top of the stack points to first cell in chunk */
  dbh->locks.freelist = dbh->locks.storage;

  /* reset the state */
  dbh->locks.tail = 0; /* 0 is considered invalid offset==>no value */
  dbstore(db, dbh->locks.queue_lock, 0);
#else
  dbstore(db, dbh->locks.global_lock, 0);
  dbstore(db, dbh->locks.writers, 0);
#endif
  return 0;
}

__device__ static gint show_lock_error(void *db, char *errmsg) {
#ifdef WG_NO_ERRPRINT
#else
  printf("wg locking error: %s.\n", errmsg);
#endif
  return -1;
}
