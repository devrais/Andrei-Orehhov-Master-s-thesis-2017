#include "../config.h"
#include "dballoc.h"
#include "dbdata.h"
#include "dbhash.h"
#include "dblog.h"

__device__ void wg_cleanup_handle_logdata(void *db) {
#ifdef USE_DBLOG
  db_handle_logdata *ld = \
    (db_handle_logdata *) (((db_handle *) db)->logdata);
  if(ld) {
    if(ld->fd >= 0) {
#ifndef _WIN32
      close(ld->fd);
#else
      _close(ld->fd);
#endif
      ld->fd = -1;
    }
    free(ld);
    ((db_handle *) db)->logdata = NULL;
  }
#endif
}
