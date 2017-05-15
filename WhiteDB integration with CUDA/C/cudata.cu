#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <sys/timeb.h>
//#include <math.h>

#include "../config.h"
#include "dballoc.h"
#include "dbindex.h"
#include "dbcompare.h"
#include "dbapi.h"
#include "dbhash.h"
#include "custring.h"

/** Get the next record from the database
 *
 */

#ifdef USE_BACKLINKING
__device__ static gint remove_backlink_index_entries(void *db, gint *record,
  gint value, gint depth);
__device__ static gint restore_backlink_index_entries(void *db, gint *record,
  gint value, gint depth);
#endif

__device__
static gint show_data_error(void* db, char* errmsg);
__device__
static gint show_data_error_nr(void* db, char* errmsg, gint nr);
__device__
void* wg_get_first_record(void* db);              ///< returns NULL when error or no recs
__device__
void* wg_get_next_record(void* db, void* record); ///< returns NULL when error or no more recs
__device__
void* wg_get_first_raw_record(void* db);
__device__
void* wg_get_next_raw_record(void* db, void* record);

__device__ static gint free_field_encoffset(void* db,gint encoffset);

__device__ static gint show_data_error_str(void* db, char* errmsg, char* str);

/* ====== Functions ============== */

__device__
void* wg_get_first_record(void* db) {  // 293
  void *res = wg_get_first_raw_record(db);
  if(res && is_special_record(res))
    return wg_get_next_record(db, res); /* find first data record */
  return res;
}

#ifdef USE_BACKLINKING

/** Remove index entries in backlink chain recursively.
 *  Needed for index maintenance when records are compared by their
 *  contens, as change in contents also changes the value of the entire
 *  record and thus affects it's placement in the index.
 *  Returns 0 for success
 *  Returns -1 in case of errors.
 */
__device__ static gint remove_backlink_index_entries(void *db, gint *record,
  gint value, gint depth) {
  gint col, length, err = 0;
  db_memsegment_header *dbh = dbmemsegh(db);

  if(!is_special_record(record)) {
    /* Find all fields in the record that match value (which is actually
     * a reference to a child record in encoded form) and remove it from
     * indexes. It will be recreated in the indexes by wg_set_field() later.
     */
    length = getusedobjectwantedgintsnr(*record) - RECORD_HEADER_GINTS;
    if(length > MAX_INDEXED_FIELDNR)
      length = MAX_INDEXED_FIELDNR + 1;

    for(col=0; col<length; col++) {
      if(*(record + RECORD_HEADER_GINTS + col) == value) {
        /* Changed value is always a WG_RECORDDTYPE field, therefore
         * we don't need to deal with index templates here
         * (record links are not allowed in templates).
         */
        if(dbh->index_control_area_header.index_table[col]) {
          if(wg_index_del_field(db, record, col) < -1)
            return -1;
        }
      }
    }
  }

  /* If recursive depth is not exchausted, continue with the parents
   * of this record.
   */
  if(depth > 0) {
    gint backlink_list = *(record + RECORD_BACKLINKS_POS);
    if(backlink_list) {
      gcell *next = (gcell *) offsettoptr(db, backlink_list);
      for(;;) {
        err = remove_backlink_index_entries(db,
          (gint *) offsettoptr(db, next->car),
          wg_encode_record(db, record), depth-1);
        if(err)
          return err;
        if(!next->cdr)
          break;
        next = (gcell *) offsettoptr(db, next->cdr);
      }
    }
  }

  return 0;
}

__device__ char* wg_decode_unistr_lang(void* db, gint data, gint type) {
  gint* objptr;
  gint* fldptr;
  gint fldval;
  char* res;

#ifdef USETINYSTR
  if (type==WG_STRTYPE && istinystr(data)) {
    return NULL;
  }
#endif
  if (type==WG_STRTYPE && isshortstr(data)) {
    return NULL;
  }
  if (islongstr(data)) {
    objptr = (gint *) offsettoptr(db,decode_longstr_offset(data));
    fldptr=((gint*)objptr)+LONGSTR_EXTRASTR_POS;
    fldval=*fldptr;
    if (fldval==0) return NULL;
    res=wg_decode_unistr(db,fldval,type);
    return res;
  }
  show_data_error(db,"data given to wg_decode_unistr_lang is not an encoded string");
  return NULL;
}


/** Add index entries in backlink chain recursively.
 *  Called after doing remove_backling_index_entries() and updating
 *  data in the record that originated the call. This recreates the
 *  entries in the indexes for all the records that were affected.
 *  Returns 0 for success
 *  Returns -1 in case of errors.
 */
__device__ static gint restore_backlink_index_entries(void *db, gint *record,
  gint value, gint depth) {
  gint col, length, err = 0;
  db_memsegment_header *dbh = dbmemsegh(db);

  if(!is_special_record(record)) {
    /* Find all fields in the record that match value (which is actually
     * a reference to a child record in encoded form) and add it back to
     * indexes.
     */
    length = getusedobjectwantedgintsnr(*record) - RECORD_HEADER_GINTS;
    if(length > MAX_INDEXED_FIELDNR)
      length = MAX_INDEXED_FIELDNR + 1;

    for(col=0; col<length; col++) {
      if(*(record + RECORD_HEADER_GINTS + col) == value) {
        if(dbh->index_control_area_header.index_table[col]) {
          if(wg_index_add_field(db, record, col) < -1)
            return -1;
        }
      }
    }
  }

  /* Continue to the parents until depth==0 */
  if(depth > 0) {
    gint backlink_list = *(record + RECORD_BACKLINKS_POS);
    if(backlink_list) {
      gcell *next = (gcell *) offsettoptr(db, backlink_list);
      for(;;) {
        err = restore_backlink_index_entries(db,
          (gint *) offsettoptr(db, next->car),
          wg_encode_record(db, record), depth-1);
        if(err)
          return err;
        if(!next->cdr)
          break;
        next = (gcell *) offsettoptr(db, next->cdr);
      }
    }
  }

  return 0;
}

#endif

/** Get the next data record from the database
 *  Uses header meta bits to filter out special records
 */
__device__
void* wg_get_next_record(void* db, void* record) {  //  303
  void *res = record;
  do {
    res = wg_get_next_raw_record(db, res);
  } while(res && is_special_record(res));
  return res;
}

__device__ wg_int wg_encode_int(void* db, wg_int data) {
  gint offset;
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_encode_int");
    return WG_ILLEGAL;
  }
#endif
  if (fits_smallint(data)) {
    return encode_smallint(data);
  } else {
#ifdef USE_DBLOG
    /* Log before allocating. Note this call is skipped when
     * we have a small int.
     */
    if(dbmemsegh(db)->logging.active) {
      if(wg_log_encode(db, WG_INTTYPE, &data, 0, NULL, 0))
        return WG_ILLEGAL;
    }
#endif
    offset=alloc_word(db);
    if (!offset) {
      show_data_error_nr(db,"cannot store an integer in wg_set_int_field: ",data);
#ifdef USE_DBLOG
      if(dbmemsegh(db)->logging.active) {
        wg_log_encval(db, WG_ILLEGAL);
      }
#endif
      return WG_ILLEGAL;
    }
    dbstore(db,offset,data);
#ifdef USE_DBLOG
    if(dbmemsegh(db)->logging.active) {
      if(wg_log_encval(db, encode_fullint_offset(offset)))
        return WG_ILLEGAL; /* journal error */
    }
#endif
    return encode_fullint_offset(offset);
  }
}

__device__ wg_int wg_cuda_set_field(void* db, void* record, wg_int fieldnr, wg_int data) {
  gint* fieldadr;
  gint fielddata;
  gint* strptr;
#ifdef USE_BACKLINKING
  gint backlink_list;           /** start of backlinks for this record */
  gint rec_enc = WG_ILLEGAL;    /** this record as encoded value. */
#endif
  db_memsegment_header *dbh = dbmemsegh(db);
#ifdef USE_CHILD_DB
  void *offset_owner = dbmemseg(db);
#endif

#ifdef CHECK
  recordcheck(db,record,fieldnr,"wg_set_field");
#endif

#ifdef USE_DBLOG
  /* Do not proceed before we've logged the operation */
  if(dbh->logging.active) {
    if(wg_log_set_field(db,record,fieldnr,data))
      return -6; /* journal error, cannot write */
  }
#endif

  /* Read the old encoded value */
  fieldadr=((gint*)record)+RECORD_HEADER_GINTS+fieldnr;
  fielddata=*fieldadr;

  /* Update index(es) while the old value is still in the db */
#ifdef USE_INDEX_TEMPLATE
  if(!is_special_record(record) && fieldnr<=MAX_INDEXED_FIELDNR &&\
    (dbh->index_control_area_header.index_table[fieldnr] ||\
     dbh->index_control_area_header.index_template_table[fieldnr])) {
#else
  if(!is_special_record(record) && fieldnr<=MAX_INDEXED_FIELDNR &&\
    dbh->index_control_area_header.index_table[fieldnr]) {
#endif
    if(wg_index_del_field(db, record, fieldnr) < -1)
      return -3; /* index error */
  }

  /* If there are backlinks, go up the chain and remove the reference
   * to this record from all indexes (updating a field in the record
   * causes the value of the record to change). Note that we only go
   * as far as the recursive comparison depth - records higher in the
   * hierarchy are not affected.
   */
#if defined(USE_BACKLINKING) && (WG_COMPARE_REC_DEPTH > 0)
  backlink_list = *((gint *) record + RECORD_BACKLINKS_POS);
  if(backlink_list) {
    gint err;
    gcell *next = (gcell *) offsettoptr(db, backlink_list);
    rec_enc = wg_encode_record(db, record);
    for(;;) {
      err = remove_backlink_index_entries(db,
        (gint *) offsettoptr(db, next->car),
        rec_enc, WG_COMPARE_REC_DEPTH-1);
      if(err) {
        return -4; /* override the error code, for now. */
      }
      if(!next->cdr)
        break;
      next = (gcell *) offsettoptr(db, next->cdr);
    }
  }
#endif

#ifdef USE_CHILD_DB
  /* Get the offset owner */
  if(isptr(data)) {
    offset_owner = get_ptr_owner(db, data);
    if(!offset_owner) {
      show_data_error(db, "External reference not recognized");
      return -5;
    }
  }
#endif

#ifdef USE_BACKLINKING
  /* Is the old field value a record pointer? If so, remove the backlink.
   * XXX: this can be optimized to use a custom macro instead of
   * wg_get_encoded_type().
   */
#ifdef USE_CHILD_DB
  /* Only touch local records */
  if(wg_get_encoded_type(db, fielddata) == WG_RECORDTYPE &&
    offset_owner == dbmemseg(db)) {
#else
  if(wg_get_encoded_type(db, fielddata) == WG_RECORDTYPE) {
#endif
    gint *rec = (gint *) wg_decode_record(db, fielddata);
    gint *next_offset = rec + RECORD_BACKLINKS_POS;
    gint parent_offset = ptrtooffset(db, record);
    gcell *old = NULL;

    while(*next_offset) {
      old = (gcell *) offsettoptr(db, *next_offset);
      if(old->car == parent_offset) {
        gint old_offset = *next_offset;
        *next_offset = old->cdr; /* remove from list chain */
        wg_free_listcell(db, old_offset); /* free storage */
        goto setfld_backlink_removed;
      }
      next_offset = &(old->cdr);
    }
    show_data_error(db, "Corrupt backlink chain");
    return -4; /* backlink error */
  }
setfld_backlink_removed:
#endif

  //printf("wg_set_field adr %d offset %d\n",fieldadr,ptrtooffset(db,fieldadr));
  if (isptr(fielddata)) {
    //printf("wg_set_field freeing old data\n");
    free_field_encoffset(db,fielddata);
  }
  (*fieldadr)=data; // store data to field
#ifdef USE_CHILD_DB
  if (islongstr(data) && offset_owner == dbmemseg(db)) {
#else
  if (islongstr(data)) {
#endif
    // increase data refcount for longstr-s
    strptr = (gint *) offsettoptr(db,decode_longstr_offset(data));
    ++(*(strptr+LONGSTR_REFCOUNT_POS));
  }

  /* Update index after new value is written */
#ifdef USE_INDEX_TEMPLATE
  if(!is_special_record(record) && fieldnr<=MAX_INDEXED_FIELDNR &&\
    (dbh->index_control_area_header.index_table[fieldnr] ||\
     dbh->index_control_area_header.index_template_table[fieldnr])) {
#else
  if(!is_special_record(record) && fieldnr<=MAX_INDEXED_FIELDNR &&\
    dbh->index_control_area_header.index_table[fieldnr]) {
#endif
    if(wg_index_add_field(db, record, fieldnr) < -1)
      return -3;
  }

#ifdef USE_BACKLINKING
  /* Is the new field value a record pointer? If so, add a backlink */
#ifdef USE_CHILD_DB
  if(wg_get_encoded_type(db, data) == WG_RECORDTYPE &&
    offset_owner == dbmemseg(db)) {
#else
  if(wg_get_encoded_type(db, data) == WG_RECORDTYPE) {
#endif
    gint *rec = (gint *) wg_decode_record(db, data);
    gint *next_offset = rec + RECORD_BACKLINKS_POS;
    gint new_offset = wg_alloc_fixlen_object(db,
      &(dbmemsegh(db)->listcell_area_header));
    gcell *new_cell = (gcell *) offsettoptr(db, new_offset);

    while(*next_offset)
      next_offset = &(((gcell *) offsettoptr(db, *next_offset))->cdr);
    new_cell->car = ptrtooffset(db, record);
    new_cell->cdr = 0;
    *next_offset = new_offset;
  }
#endif

#if defined(USE_BACKLINKING) && (WG_COMPARE_REC_DEPTH > 0)
  /* Create new entries in indexes in all referring records */
  if(backlink_list) {
    gint err;
    gcell *next = (gcell *) offsettoptr(db, backlink_list);
    for(;;) {
      err = restore_backlink_index_entries(db,
        (gint *) offsettoptr(db, next->car),
        rec_enc, WG_COMPARE_REC_DEPTH-1);
      if(err) {
        return -4;
      }
      if(!next->cdr)
        break;
      next = (gcell *) offsettoptr(db, next->cdr);
    }
  }
#endif

  return 0;
}

__device__ wg_int wg_set_field(void* db, void* record, wg_int fieldnr, wg_int data) {
  gint* fieldadr;
  gint fielddata;
  gint* strptr;
#ifdef USE_BACKLINKING
  gint backlink_list;           /** start of backlinks for this record */
  gint rec_enc = WG_ILLEGAL;    /** this record as encoded value. */
#endif
  db_memsegment_header *dbh = dbmemsegh(db);
#ifdef USE_CHILD_DB
  void *offset_owner = dbmemseg(db);
#endif

#ifdef CHECK
  recordcheck(db,record,fieldnr,"wg_set_field");
#endif

#ifdef USE_DBLOG
  /* Do not proceed before we've logged the operation */
  if(dbh->logging.active) {
    if(wg_log_set_field(db,record,fieldnr,data))
      return -6; /* journal error, cannot write */
  }
#endif

  /* Read the old encoded value */
  fieldadr=((gint*)record)+RECORD_HEADER_GINTS+fieldnr;
  fielddata=*fieldadr;

  /* Update index(es) while the old value is still in the db */
#ifdef USE_INDEX_TEMPLATE
  if(!is_special_record(record) && fieldnr<=MAX_INDEXED_FIELDNR &&\
    (dbh->index_control_area_header.index_table[fieldnr] ||\
     dbh->index_control_area_header.index_template_table[fieldnr])) {
#else
  if(!is_special_record(record) && fieldnr<=MAX_INDEXED_FIELDNR &&\
    dbh->index_control_area_header.index_table[fieldnr]) {
#endif
    if(wg_index_del_field(db, record, fieldnr) < -1)
      return -3; /* index error */
  }

  /* If there are backlinks, go up the chain and remove the reference
   * to this record from all indexes (updating a field in the record
   * causes the value of the record to change). Note that we only go
   * as far as the recursive comparison depth - records higher in the
   * hierarchy are not affected.
   */
#if defined(USE_BACKLINKING) && (WG_COMPARE_REC_DEPTH > 0)
  backlink_list = *((gint *) record + RECORD_BACKLINKS_POS);
  if(backlink_list) {
    gint err;
    gcell *next = (gcell *) offsettoptr(db, backlink_list);
    rec_enc = wg_encode_record(db, record);
    for(;;) {
      err = remove_backlink_index_entries(db,
        (gint *) offsettoptr(db, next->car),
        rec_enc, WG_COMPARE_REC_DEPTH-1);
      if(err) {
        return -4; /* override the error code, for now. */
      }
      if(!next->cdr)
        break;
      next = (gcell *) offsettoptr(db, next->cdr);
    }
  }
#endif

#ifdef USE_CHILD_DB
  /* Get the offset owner */
  if(isptr(data)) {
    offset_owner = get_ptr_owner(db, data);
    if(!offset_owner) {
      show_data_error(db, "External reference not recognized");
      return -5;
    }
  }
#endif

#ifdef USE_BACKLINKING
  /* Is the old field value a record pointer? If so, remove the backlink.
   * XXX: this can be optimized to use a custom macro instead of
   * wg_get_encoded_type().
   */
#ifdef USE_CHILD_DB
  /* Only touch local records */
  if(wg_get_encoded_type(db, fielddata) == WG_RECORDTYPE &&
    offset_owner == dbmemseg(db)) {
#else
  if(wg_get_encoded_type(db, fielddata) == WG_RECORDTYPE) {
#endif
    gint *rec = (gint *) wg_decode_record(db, fielddata);
    gint *next_offset = rec + RECORD_BACKLINKS_POS;
    gint parent_offset = ptrtooffset(db, record);
    gcell *old = NULL;

    while(*next_offset) {
      old = (gcell *) offsettoptr(db, *next_offset);
      if(old->car == parent_offset) {
        gint old_offset = *next_offset;
        *next_offset = old->cdr; /* remove from list chain */
        wg_free_listcell(db, old_offset); /* free storage */
        goto setfld_backlink_removed;
      }
      next_offset = &(old->cdr);
    }
    show_data_error(db, "Corrupt backlink chain");
    return -4; /* backlink error */
  }
setfld_backlink_removed:
#endif

  //printf("wg_set_field adr %d offset %d\n",fieldadr,ptrtooffset(db,fieldadr));
  if (isptr(fielddata)) {
    //printf("wg_set_field freeing old data\n");
    free_field_encoffset(db,fielddata);
  }
  (*fieldadr)=data; // store data to field
#ifdef USE_CHILD_DB
  if (islongstr(data) && offset_owner == dbmemseg(db)) {
#else
  if (islongstr(data)) {
#endif
    // increase data refcount for longstr-s
    strptr = (gint *) offsettoptr(db,decode_longstr_offset(data));
    ++(*(strptr+LONGSTR_REFCOUNT_POS));
  }

  /* Update index after new value is written */
#ifdef USE_INDEX_TEMPLATE
  if(!is_special_record(record) && fieldnr<=MAX_INDEXED_FIELDNR &&\
    (dbh->index_control_area_header.index_table[fieldnr] ||\
     dbh->index_control_area_header.index_template_table[fieldnr])) {
#else
  if(!is_special_record(record) && fieldnr<=MAX_INDEXED_FIELDNR &&\
    dbh->index_control_area_header.index_table[fieldnr]) {
#endif
    if(wg_index_add_field(db, record, fieldnr) < -1)
      return -3;
  }

#ifdef USE_BACKLINKING
  /* Is the new field value a record pointer? If so, add a backlink */
#ifdef USE_CHILD_DB
  if(wg_get_encoded_type(db, data) == WG_RECORDTYPE &&
    offset_owner == dbmemseg(db)) {
#else
  if(wg_get_encoded_type(db, data) == WG_RECORDTYPE) {
#endif
    gint *rec = (gint *) wg_decode_record(db, data);
    gint *next_offset = rec + RECORD_BACKLINKS_POS;
    gint new_offset = wg_alloc_fixlen_object(db,
      &(dbmemsegh(db)->listcell_area_header));
    gcell *new_cell = (gcell *) offsettoptr(db, new_offset);

    while(*next_offset)
      next_offset = &(((gcell *) offsettoptr(db, *next_offset))->cdr);
    new_cell->car = ptrtooffset(db, record);
    new_cell->cdr = 0;
    *next_offset = new_offset;
  }
#endif

#if defined(USE_BACKLINKING) && (WG_COMPARE_REC_DEPTH > 0)
  /* Create new entries in indexes in all referring records */
  if(backlink_list) {
    gint err;
    gcell *next = (gcell *) offsettoptr(db, backlink_list);
    for(;;) {
      err = restore_backlink_index_entries(db,
        (gint *) offsettoptr(db, next->car),
        rec_enc, WG_COMPARE_REC_DEPTH-1);
      if(err) {
        return -4;
      }
      if(!next->cdr)
        break;
      next = (gcell *) offsettoptr(db, next->cdr);
    }
  }
#endif

  return 0;
}

__device__ void* wg_decode_record(void* db, wg_int data) {
  #ifdef CHECK
    if (!dbcheck(db)) {
      show_data_error(db,"wrong database pointer given to wg_encode_char");
      return 0;
    }
  #endif
    return (void*)(offsettoptr(db,decode_datarec_offset(data)));
  }

__device__ static gint free_field_encoffset(void* db,gint encoffset) {
    gint offset;
  #if 0
    gint* dptr;
    gint* dendptr;
    gint data;
    gint i;
  #endif
    gint tmp;
    gint* objptr;
    gint* extrastr;

    // takes last three bits to decide the type
    // fullint is represented by two options: 001 and 101
    switch(encoffset&NORMALPTRMASK) {
      case DATARECBITS:
  #if 0
  /* This section of code in quarantine */
        // remove from list
        // refcount check
        offset=decode_datarec_offset(encoffset);
        tmp=dbfetch(db,offset+sizeof(gint)*LONGSTR_REFCOUNT_POS);
        tmp--;
        if (tmp>0) {
          dbstore(db,offset+LONGSTR_REFCOUNT_POS,tmp);
        } else {
          // free frompointers structure
          // loop over fields, freeing them
          dptr=offsettoptr(db,offset);
          dendptr=(gint*)(((char*)dptr)+datarec_size_bytes(*dptr));
          for(i=0,dptr=dptr+RECORD_HEADER_GINTS;dptr<dendptr;dptr++,i++) {
            data=*dptr;
            if (isptr(data)) free_field_encoffset(db,data);
          }
          // really free object from area
          wg_free_object(db,&(dbmemsegh(db)->datarec_area_header),offset);
        }
  #endif
        break;
      case LONGSTRBITS:
        offset=decode_longstr_offset(encoffset);
  #ifdef USE_CHILD_DB
        if(!is_local_offset(db, offset))
          break; /* Non-local reference, ignore it */
  #endif
        // refcount check
        tmp=dbfetch(db,offset+sizeof(gint)*LONGSTR_REFCOUNT_POS);
        tmp--;
        if (tmp>0) {
          dbstore(db,offset+sizeof(gint)*LONGSTR_REFCOUNT_POS,tmp);
        } else {
          objptr = (gint *) offsettoptr(db,offset);
          extrastr=(gint*)(((char*)(objptr))+(sizeof(gint)*LONGSTR_EXTRASTR_POS));
          tmp=*extrastr;
          // remove from hash
          wg_remove_from_strhash(db,encoffset);
          // remove extrastr
          if (tmp!=0) free_field_encoffset(db,tmp);
          *extrastr=0;
          // really free object from area
          wg_free_object(db,&(dbmemsegh(db)->longstr_area_header),offset);
        }
        break;
      case SHORTSTRBITS:
  #ifdef USE_CHILD_DB
        offset = decode_shortstr_offset(encoffset);
        if(!is_local_offset(db, offset))
          break; /* Non-local reference, ignore it */
        wg_free_shortstr(db, offset);
  #else
        wg_free_shortstr(db,decode_shortstr_offset(encoffset));
  #endif
        break;
      case FULLDOUBLEBITS:
  #ifdef USE_CHILD_DB
        offset = decode_fulldouble_offset(encoffset);
        if(!is_local_offset(db, offset))
          break; /* Non-local reference, ignore it */
        wg_free_doubleword(db, offset);
  #else
        wg_free_doubleword(db,decode_fulldouble_offset(encoffset));
  #endif
        break;
      case FULLINTBITSV0:
  #ifdef USE_CHILD_DB
        offset = decode_fullint_offset(encoffset);
        if(!is_local_offset(db, offset))
          break; /* Non-local reference, ignore it */
        wg_free_word(db, offset);
  #else
        wg_free_word(db,decode_fullint_offset(encoffset));
  #endif
        break;
      case FULLINTBITSV1:
  #ifdef USE_CHILD_DB
        offset = decode_fullint_offset(encoffset);
        if(!is_local_offset(db, offset))
          break; /* Non-local reference, ignore it */
        wg_free_word(db, offset);
  #else
        wg_free_word(db,decode_fullint_offset(encoffset));
  #endif
        break;

    }
    return 0;
  }

/** Get the first record from the database
 *
 */
__device__
void* wg_get_first_raw_record(void* db) {  //314
  db_subarea_header* arrayadr;
  gint firstoffset;
  void* res;

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_get_first_record");
    return NULL;
  }
#endif
  arrayadr=&((dbmemsegh(db)->datarec_area_header).subarea_array[0]);
  firstoffset=((arrayadr[0]).alignedoffset); // do NOT skip initial "used" marker
  //printf("arrayadr %x firstoffset %d \n",(uint)arrayadr,firstoffset);
  res=wg_get_next_raw_record(db,offsettoptr(db,firstoffset));
  return res;
}

__device__
void* wg_get_next_raw_record(void* db, void* record) {  //335
  gint curoffset;
  gint head;
  db_subarea_header* arrayadr;
  gint last_subarea_index;
  gint i;
  gint found;
  gint subareastart;
  gint subareaend;
  gint freemarker;

  curoffset=ptrtooffset(db,record);
  //printf("curroffset %d record %x\n",curoffset,(uint)record);
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_get_first_record");
    return NULL;
  }
  head=dbfetch(db,curoffset);
  if (isfreeobject(head)) {
    show_data_error(db,"wrong record pointer (free) given to wg_get_next_record");
    return NULL;
  }
#endif
  freemarker=0; //assume input pointer to used object
  head=dbfetch(db,curoffset);
  while(1) {
    // increase offset to next memory block
    curoffset=curoffset+(freemarker ? getfreeobjectsize(head) : getusedobjectsize(head));
    head=dbfetch(db,curoffset);
    //printf("new curoffset %d head %d isnormaluseobject %d isfreeobject %d \n",
    //       curoffset,head,isnormalusedobject(head),isfreeobject(head));
    // check if found a normal used object
    if (isnormalusedobject(head)) return offsettoptr(db,curoffset); //return ptr to normal used object
    if (isfreeobject(head)) {
      freemarker=1;
      // loop start leads us to next object
    } else {
      // found a special object (dv or end marker)
      freemarker=0;
      if (dbfetch(db,curoffset+sizeof(gint))==SPECIALGINT1DV) {
        // we have reached a dv object
        continue; // loop start leads us to next object
      } else {
        // we have reached an end marker, have to find the next subarea
        // first locate subarea for this offset
        arrayadr=&((dbmemsegh(db)->datarec_area_header).subarea_array[0]);
        last_subarea_index=(dbmemsegh(db)->datarec_area_header).last_subarea_index;
        found=0;
        for(i=0;(i<=last_subarea_index)&&(i<SUBAREA_ARRAY_SIZE);i++) {
          subareastart=((arrayadr[i]).alignedoffset);
          subareaend=((arrayadr[i]).offset)+((arrayadr[i]).size);
          if (curoffset>=subareastart && curoffset<subareaend) {
            found=1;
            break;
          }
        }
        if (!found) {
          show_data_error(db,"wrong record pointer (out of area) given to wg_get_next_record");
          return NULL;
        }
        // take next subarea, while possible
        i++;
        if (i>last_subarea_index || i>=SUBAREA_ARRAY_SIZE) {
          //printf("next used object not found: i %d curoffset %d \n",i,curoffset);
          return NULL;
        }
        //printf("taking next subarea i %d\n",i);
        curoffset=((arrayadr[i]).alignedoffset);  // curoffset is now the special start marker
        head=dbfetch(db,curoffset);
        // loop start will lead us to next object from special marker
      }
    }
  }
}

__device__
wg_int wg_get_record_len(void* db, void* record) {  // 580

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_get_record_len");
    return -1;
  }
#endif
  return ((gint)(getusedobjectwantedgintsnr(*((gint*)record))))-RECORD_HEADER_GINTS;
}

__device__
wg_int wg_cuda_get_field(void* db, void* record, wg_int fieldnr) { // 1141

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error_nr(db,"wrong database pointer given to wg_get_field",fieldnr);
    return WG_ILLEGAL;
  }
  if (fieldnr<0 || (getusedobjectwantedgintsnr(*((gint*)record))<=fieldnr+RECORD_HEADER_GINTS)) {
    show_data_error_nr(db,"wrong field number given to wg_get_field",fieldnr);\
    return WG_ILLEGAL;
  }
#endif
  //printf("wg_get_field adr %d offset %d\n",
  //       (((gint*)record)+RECORD_HEADER_GINTS+fieldnr),
  //       ptrtooffset(db,(((gint*)record)+RECORD_HEADER_GINTS+fieldnr)));
  return *(((gint*)record)+RECORD_HEADER_GINTS+fieldnr);
}

__device__
wg_int wg_get_field(void* db, void* record, wg_int fieldnr) { // 1141

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error_nr(db,"wrong database pointer given to wg_get_field",fieldnr);
    return WG_ILLEGAL;
  }
  if (fieldnr<0 || (getusedobjectwantedgintsnr(*((gint*)record))<=fieldnr+RECORD_HEADER_GINTS)) {
    show_data_error_nr(db,"wrong field number given to wg_get_field",fieldnr);\
    return WG_ILLEGAL;
  }
#endif
  //printf("wg_get_field adr %d offset %d\n",
  //       (((gint*)record)+RECORD_HEADER_GINTS+fieldnr),
  //       ptrtooffset(db,(((gint*)record)+RECORD_HEADER_GINTS+fieldnr)));
  return *(((gint*)record)+RECORD_HEADER_GINTS+fieldnr);
}

__device__
wg_int wg_get_encoded_type(void* db, wg_int data) {  // 1350
  gint fieldoffset;
  gint tmp;

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_get_encoded_type");
    return 0;
  }
#endif
  if (!data) return WG_NULLTYPE;
  if (((data)&NONPTRBITS)==NONPTRBITS) {
    // data is one of the non-pointer types
    if (isvar(data)) return (gint)WG_VARTYPE;
    if (issmallint(data)) return (gint)WG_INTTYPE;
    switch(data&LASTBYTEMASK) {
      case CHARBITS: return WG_CHARTYPE;
      case FIXPOINTBITS: return WG_FIXPOINTTYPE;
      case DATEBITS: return WG_DATETYPE;
      case TIMEBITS: return WG_TIMETYPE;
      case TINYSTRBITS: return WG_STRTYPE;
      case VARBITS: return WG_VARTYPE;
      case ANONCONSTBITS: return WG_ANONCONSTTYPE;
      default: return -1;
    }
  }
  // here we know data must be of ptr type
  // takes last three bits to decide the type
  // fullint is represented by two options: 001 and 101
  //printf("cp0\n");
  switch(data&NORMALPTRMASK) {
    case DATARECBITS: return (gint)WG_RECORDTYPE;
    case LONGSTRBITS:
      //printf("cp1\n");
      fieldoffset=decode_longstr_offset(data)+LONGSTR_META_POS*sizeof(gint);
      //printf("fieldoffset %d\n",fieldoffset);
      tmp=dbfetch(db,fieldoffset);
      //printf("str meta %d lendiff %d subtype %d\n",
      //  tmp,(tmp&LONGSTR_META_LENDIFMASK)>>LONGSTR_META_LENDIFSHFT,tmp&LONGSTR_META_TYPEMASK);
      return tmp&LONGSTR_META_TYPEMASK; // WG_STRTYPE, WG_URITYPE, WG_XMLLITERALTYPE
    case SHORTSTRBITS:   return (gint)WG_STRTYPE;
    case FULLDOUBLEBITS: return (gint)WG_DOUBLETYPE;
    case FULLINTBITSV0:  return (gint)WG_INTTYPE;
    case FULLINTBITSV1:  return (gint)WG_INTTYPE;
    default: return -1;
  }
  return 0;
}

__device__
wg_int wg_cuda_decode_int(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_int");
    return 0;
  }
#endif
  if (issmallint(data)) return decode_smallint(data);
  if (isfullint(data)) return dbfetch(db,decode_fullint_offset(data));
  show_data_error_nr(db,"data given to wg_decode_int is not an encoded int: ",data);
  return 0;
}

__device__
wg_int wg_decode_int(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_int");
    return 0;
  }
#endif
  if (issmallint(data)) return decode_smallint(data);
  if (isfullint(data)) return dbfetch(db,decode_fullint_offset(data));
  show_data_error_nr(db,"data given to wg_decode_int is not an encoded int: ",data);
  return 0;
}

__device__
static gint show_data_error(void* db, char* errmsg) { // 3165
#ifdef WG_NO_ERRPRINT
#else
  //fprintf(stderr,"wg data handling error: %s\n",errmsg);
	printf("wg data handling error: %s\n",errmsg);
#endif
  return -1;

}

__device__
static gint show_data_error_nr(void* db, char* errmsg, gint nr) { // 3174
#ifdef WG_NO_ERRPRINT
#else
 // fprintf(stderr,"wg data handling error: %s %d\n", errmsg, (int) nr);
	printf("wg data handling error: %s %d\n", errmsg, (int) nr);
#endif
  return -1;

}

__device__
void* wg_create_raw_record(void* db, wg_int length) {
  gint offset;
  gint i;

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error_nr(db,"wrong database pointer given to wg_create_record with length ",length);
    return 0;
  }
  if(length < 0) {
    show_data_error_nr(db, "invalid record length:",length);
    return 0;
  }
#endif

#ifdef USE_DBLOG
  /* Log first, modify shared memory next */
  if(dbmemsegh(db)->logging.active) {
    if(wg_log_create_record(db, length))
      return 0;
  }
#endif

  offset=wg_alloc_gints(db,
                     &(dbmemsegh(db)->datarec_area_header),
                    length+RECORD_HEADER_GINTS);
  if (!offset) {
    show_data_error_nr(db,"cannot create a record of size ",length);
#ifdef USE_DBLOG
    if(dbmemsegh(db)->logging.active) {
      wg_log_encval(db, 0);
    }
#endif
    return 0;
  }

  /* Init header */
  dbstore(db, offset+RECORD_META_POS*sizeof(gint), 0);
  dbstore(db, offset+RECORD_BACKLINKS_POS*sizeof(gint), 0);
  for(i=RECORD_HEADER_GINTS;i<length+RECORD_HEADER_GINTS;i++) {
    dbstore(db,offset+(i*(sizeof(gint))),0);
  }

#ifdef USE_DBLOG
  /* Append the created offset to log */
  if(dbmemsegh(db)->logging.active) {
    if(wg_log_encval(db, offset))
      return 0; /* journal error */
  }
#endif

  return offsettoptr(db,offset);
}

__device__
void* wg_cuda_create_record(void* db, wg_int length) {
  void *rec = wg_create_raw_record(db, length);
  /* Index all the created NULL fields to ensure index consistency */
  if(rec) {
    if(wg_index_add_rec(db, rec) < -1)
      return NULL; /* index error */
  }
  return rec;
}

__device__
void* wg_create_record(void* db, wg_int length) {
  void *rec = wg_create_raw_record(db, length);
  /* Index all the created NULL fields to ensure index consistency */
  if(rec) {
    if(wg_index_add_rec(db, rec) < -1)
      return NULL; /* index error */
  }
  return rec;
}

__device__ wg_int wg_encode_record(void* db, void* data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_encode_char");
    return WG_ILLEGAL;
  }
#endif
#ifdef USE_DBLOG
/* Skip logging values that do not cause storage allocation.
  if(dbh->logging.active) {
    if(wg_log_encode(db, WG_RECORDTYPE, &data, 0, NULL, 0))
      return WG_ILLEGAL;
  }
*/
#endif
  return (wg_int)(encode_datarec_offset(ptrtooffset(db,data)));
}

__device__ wg_int wg_cuda_encode_int(void* db, wg_int data) {
  gint offset;
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_encode_int");
    return WG_ILLEGAL;
  }
#endif
  if (fits_smallint(data)) {
    return encode_smallint(data);
  } else {
#ifdef USE_DBLOG
    /* Log before allocating. Note this call is skipped when
     * we have a small int.
     */
    if(dbmemsegh(db)->logging.active) {
      if(wg_log_encode(db, WG_INTTYPE, &data, 0, NULL, 0))
        return WG_ILLEGAL;
    }
#endif
    offset=alloc_word(db);
    if (!offset) {
      show_data_error_nr(db,"cannot store an integer in wg_set_int_field: ",data);
#ifdef USE_DBLOG
      if(dbmemsegh(db)->logging.active) {
        wg_log_encval(db, WG_ILLEGAL);
      }
#endif
      return WG_ILLEGAL;
    }
    dbstore(db,offset,data);
#ifdef USE_DBLOG
    if(dbmemsegh(db)->logging.active) {
      if(wg_log_encval(db, encode_fullint_offset(offset)))
        return WG_ILLEGAL; /* journal error */
    }
#endif
    return encode_fullint_offset(offset);
  }
}

__device__ int wg_decode_time(void* db, wg_int data) {

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_time");
    return 0;
  }
#endif
  if (istime(data)) return decode_time(data);
  show_data_error_nr(db,"data given to wg_decode_time is not an encoded time: ",data);
  return 0;
}

__device__ char* wg_decode_uri_prefix(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_uri_prefix");
    return NULL;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_uri_prefix is 0, not an encoded uri");
    return NULL;
  }
#endif
  return wg_decode_unistr_lang(db,data,WG_URITYPE);
}

__device__ wg_int wg_decode_xmlliteral_xsdtype_len(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_xmlliteral_xsdtype_len");
    return -1;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_xmlliteral_lang_xsdtype is 0, not an encoded xmlliteral");
    return -1;
  }
#endif
  return wg_decode_unistr_lang_len(db,data,WG_XMLLITERALTYPE);
}

__device__ gint wg_decode_unistr_lang_len(void* db, gint data, gint type) {
  char* langptr;
  gint len;

  langptr=wg_decode_unistr_lang(db,data,type);
  if (langptr==NULL) {
    return 0;
  }
  len=strlen(langptr);
  return len;
}
__device__ char* wg_decode_xmlliteral(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_xmlliteral");
    return NULL;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_xmlliteral is 0, not an encoded xmlliteral");
    return NULL;
  }
#endif
  return wg_decode_unistr(db,data,WG_XMLLITERALTYPE);
}

__device__ char* wg_decode_unistr(void* db, gint data, gint type) {
  gint* objptr;
  char* dataptr;
#ifdef USETINYSTR
  if (type==WG_STRTYPE && istinystr(data)) {
    if (LITTLEENDIAN) {
      dataptr=((char*)(&data))+1; // type bits stored in lowest addressed byte
    } else {
      dataptr=((char*)(&data));  // type bits stored in highest addressed byte
    }
    return dataptr;
  }
#endif
  if (isshortstr(data)) {
    dataptr=(char*)(offsettoptr(db,decode_shortstr_offset(data)));
    return dataptr;
  }
  if (islongstr(data)) {
    objptr = (gint *) offsettoptr(db,decode_longstr_offset(data));
    dataptr=((char*)(objptr))+(LONGSTR_HEADER_GINTS*sizeof(gint));
    return dataptr;
  }
  show_data_error(db,"data given to wg_decode_unistr is not an encoded string");
  return NULL;
}

__device__
char* wg_decode_str(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_str");
    return NULL;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_str is 0, not an encoded string");
    return NULL;
  }
#endif
  return wg_decode_unistr(db,data,WG_STRTYPE);
}

__device__ char* wg_decode_xmlliteral_xsdtype(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_xmlliteral");
    return NULL;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_xmlliteral_xsdtype is 0, not an encoded xmlliteral");
    return NULL;
  }
#endif
  return wg_decode_unistr_lang(db,data,WG_XMLLITERALTYPE);
}

__device__ int wg_decode_date(void* db, wg_int data) {

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_date");
    return 0;
  }
#endif
  if (isdate(data)) return decode_date(data);
  show_data_error_nr(db,"data given to wg_decode_date is not an encoded date: ",data);
  return 0;
}

__device__ wg_int wg_decode_var(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_var");
    return -1;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_var is 0, not an encoded var");
    return -1;
  }
#endif
  return decode_var(data);
}

__device__ char wg_decode_char(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_char");
    return 0;
  }
#endif
  return (char)(decode_char(data));
}

__device__ wg_int wg_decode_uri_len(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_uri_len");
    return -1;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_uri_len is 0, not an encoded string");
    return -1;
  }
#endif
  return wg_decode_unistr_len(db,data,WG_URITYPE);
}

__device__ char* wg_decode_uri(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_uri");
    return NULL;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_uri is 0, not an encoded string");
    return NULL;
  }
#endif
  return wg_decode_unistr(db,data,WG_URITYPE);
}

__device__ double wg_decode_fixpoint(void* db, wg_int data) {

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_double");
    return 0;
  }
#endif
  if (isfixpoint(data)) return decode_fixpoint(data);
  show_data_error_nr(db,"data given to wg_decode_fixpoint is not an encoded fixpoint: ",data);
  return 0;
}

__device__ double wg_decode_double(void* db, wg_int data) {

#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_double");
    return 0;
  }
#endif
  if (isfulldouble(data)) return *((double*)(offsettoptr(db,decode_fulldouble_offset(data))));
  show_data_error_nr(db,"data given to wg_decode_double is not an encoded double: ",data);
  return 0;
}
__device__ wg_int wg_decode_xmlliteral_len(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_xmlliteral_len");
    return -1;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_xmlliteral_len is 0, not an encoded xmlliteral");
    return -1;
  }
#endif
  return wg_decode_unistr_len(db,data,WG_XMLLITERALTYPE);
}

__device__ char* wg_decode_blob(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_blob");
    return NULL;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_blob is 0, not an encoded string");
    return NULL;
  }
#endif
  return wg_decode_unistr(db,data,WG_BLOBTYPE);
}

__device__ gint wg_decode_unistr_len(void* db, gint data, gint type) {
  char* dataptr;
  gint* objptr;
  gint objsize;
  gint strsize;

#ifdef USETINYSTR
  if (type==WG_STRTYPE && istinystr(data)) {
    if (LITTLEENDIAN) {
      dataptr=((char*)(&data))+1; // type bits stored in lowest addressed byte
    } else {
      dataptr=((char*)(&data));  // type bits stored in highest addressed byte
    }
    strsize=strlen(dataptr);
    return strsize;
  }
#endif
  if (isshortstr(data)) {
    dataptr=(char*)(offsettoptr(db,decode_shortstr_offset(data)));
    strsize=strlen(dataptr);
    return strsize;
  }
  if (islongstr(data)) {
    objptr = (gint *) offsettoptr(db,decode_longstr_offset(data));
    objsize=getusedobjectsize(*objptr);
    dataptr=((char*)(objptr))+(LONGSTR_HEADER_GINTS*sizeof(gint));
    //printf("dataptr to read from %d str '%s' of len %d\n",dataptr,dataptr,strlen(dataptr));
    strsize=objsize-(((*(objptr+LONGSTR_META_POS))&LONGSTR_META_LENDIFMASK)>>LONGSTR_META_LENDIFSHFT);
    return strsize-1;
  }
  show_data_error(db,"data given to wg_decode_unistr_len is not an encoded string");
  return 0;
}

__device__ wg_int wg_decode_str_len(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_str_len");
    return -1;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_str_len is 0, not an encoded string");
    return -1;
  }
#endif
  return wg_decode_unistr_len(db,data,WG_STRTYPE);
}

__device__ wg_int wg_decode_blob_len(void* db, wg_int data) {
#ifdef CHECK
  if (!dbcheck(db)) {
    show_data_error(db,"wrong database pointer given to wg_decode_blob_len");
    return -1;
  }
  if (!data) {
    show_data_error(db,"data given to wg_decode_blob_len is 0, not an encoded string");
    return -1;
  }
#endif
  return wg_decode_unistr_len(db,data,WG_BLOBTYPE)+1;
}

__device__
static gint show_data_error_str(void* db, char* errmsg, char* str) {
#ifdef WG_NO_ERRPRINT
#else
  //printf("wg data handling error: %s %s\n",errmsg,str);
#endif
  return -1;
}
