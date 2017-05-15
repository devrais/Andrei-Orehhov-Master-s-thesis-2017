#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "../config.h"
#include "dbhash.h"
#include "dbdata.h"
#include "dbmpool.h"
#include "custring.h"

#define CONCAT_FOR_HASHING(d, b, e, l, bb, en) \
  if(e) { \
    gint xl = wg_decode_xmlliteral_xsdtype_len(d, en); \
    bb = (char *)malloc(xl + l + 1); \
    if(!bb) \
      return 0; \
    memcpy(bb, e, xl); \
    bb[xl] = '\0'; \
    memcpy(bb + xl + 1, b, l); \
    b = bb; \
    l += xl + 1; \
  }

/* Remove longstr from strhash
*
*  Internal langstr etc are not removed by this op.
*
*/

#if __GNUC_PREREQ (3,3)
# define cu__nonnull(params) __attribute__ ((__nonnull__ params))
#else
# define cu__nonnull(params)
#endif

/* ======= Private protos ================ */
__device__ static gint show_consistency_error_nr(void* db, char* errmsg, gint nr);

__device__ static wg_uint hash_bytes(void *db, char *data, gint length, gint hashsz);

__device__ static gint find_idxhash_bucket(void *db, char *data, gint length, gint *chainoffset);

__device__ static gint show_hash_error(void* db, char* errmsg);


/* ====== Functions ============== */

__device__ gint wg_remove_from_strhash(void* db, gint longstr) {
  db_memsegment_header* dbh = dbmemsegh(db);
  gint type;
  gint* extrastrptr;
  char* extrastr;
  char* data;
  gint length;
  gint hash;
  gint chainoffset;
  gint hashchain;
  gint nextchain;
  gint offset;
  gint* objptr;
  gint fldval;
  gint objsize;
  gint strsize;
  gint* typeptr;

  //printf("wg_remove_from_strhash called on %d\n",longstr);
  //wg_debug_print_value(db,longstr);
  //printf("\n\n");
  offset=decode_longstr_offset(longstr);
  objptr=(gint*) offsettoptr(db,offset);
  // get string data elements
  //type=objptr=offsettoptr(db,decode_longstr_offset(data));
  extrastrptr=(gint *) (((char*)(objptr))+(LONGSTR_EXTRASTR_POS*sizeof(gint)));
  fldval=*extrastrptr;
  if (fldval==0) extrastr=NULL;
  else extrastr=wg_decode_str(db,fldval);
  data=((char*)(objptr))+(LONGSTR_HEADER_GINTS*sizeof(gint));
  objsize=getusedobjectsize(*objptr);
  strsize=objsize-(((*(objptr+LONGSTR_META_POS))&LONGSTR_META_LENDIFMASK)>>LONGSTR_META_LENDIFSHFT);
  length=strsize;
  typeptr=(gint*)(((char*)(objptr))+(+LONGSTR_META_POS*sizeof(gint)));
  type=(*typeptr)&LONGSTR_META_TYPEMASK;
  //type=wg_get_encoded_type(db,longstr);
  // get hash of data elements and find the location in hashtable/chains
  hash=wg_hash_typedstr(db,data,extrastr,type,length);
  chainoffset=((dbh->strhash_area_header).arraystart)+(sizeof(gint)*hash);
  hashchain=dbfetch(db,chainoffset);
  while(hashchain!=0) {
    if (hashchain==longstr) {
      nextchain=dbfetch(db,decode_longstr_offset(hashchain)+(LONGSTR_HASHCHAIN_POS*sizeof(gint)));
      dbstore(db,chainoffset,nextchain);
      return 0;
    }
    chainoffset=decode_longstr_offset(hashchain)+(LONGSTR_HASHCHAIN_POS*sizeof(gint));
    hashchain=dbfetch(db,chainoffset);
  }
  show_consistency_error_nr(db,"string not found in hash during deletion, offset",offset);
  return -1;
}

__device__ gint wg_decode_for_hashing(void *db, gint enc, char **decbytes) {
  gint len;
  gint type;
  gint ptrdata;
  int intdata;
  double doubledata;
  char *bytedata;
  char *exdata, *buf = NULL, *outbuf;

  type = wg_get_encoded_type(db, enc);
  switch(type) {
    case WG_NULLTYPE:
      len = sizeof(gint);
      ptrdata = 0;
      bytedata = (char *) &ptrdata;
      break;
    case WG_RECORDTYPE:
      len = sizeof(gint);
      ptrdata = enc;
      bytedata = (char *) &ptrdata;
      break;
    case WG_INTTYPE:
      len = sizeof(int);
      intdata = wg_decode_int(db, enc);
      bytedata = (char *) &intdata;
      break;
    case WG_DOUBLETYPE:
      len = sizeof(double);
      doubledata = wg_decode_double(db, enc);
      bytedata = (char *) &doubledata;
      break;
    case WG_FIXPOINTTYPE:
      len = sizeof(double);
      doubledata = wg_decode_fixpoint(db, enc);
      bytedata = (char *) &doubledata;
      break;
    case WG_STRTYPE:
      len = wg_decode_str_len(db, enc);
      bytedata = wg_decode_str(db, enc);
      break;
    case WG_URITYPE:
      len = wg_decode_uri_len(db, enc);
      bytedata = wg_decode_uri(db, enc);
      exdata = wg_decode_uri_prefix(db, enc);
      CONCAT_FOR_HASHING(db, bytedata, exdata, len, buf, enc)
      break;
    case WG_XMLLITERALTYPE:
      len = wg_decode_xmlliteral_len(db, enc);
      bytedata = wg_decode_xmlliteral(db, enc);
      exdata = wg_decode_xmlliteral_xsdtype(db, enc);
      CONCAT_FOR_HASHING(db, bytedata, exdata, len, buf, enc)
      break;
    case WG_CHARTYPE:
      len = sizeof(int);
      intdata = wg_decode_char(db, enc);
      bytedata = (char *) &intdata;
      break;
    case WG_DATETYPE:
      len = sizeof(int);
      intdata = wg_decode_date(db, enc);
      bytedata = (char *) &intdata;
      break;
    case WG_TIMETYPE:
      len = sizeof(int);
      intdata = wg_decode_time(db, enc);
      bytedata = (char *) &intdata;
      break;
    case WG_VARTYPE:
      len = sizeof(int);
      intdata = wg_decode_var(db, enc);
      bytedata = (char *) &intdata;
      break;
    case WG_ANONCONSTTYPE:
      /* Ignore anonconst */
    default:
      return 0;
  }

  /* Form the hashable buffer. It is not 0-terminated */
  outbuf = (char *)malloc(len + 1);
  if(outbuf) {
    outbuf[0] = (char) type;
    memcpy(outbuf + 1, bytedata, len++);
    *decbytes = outbuf;
  } else {
    /* Indicate failure */
    len = 0;
  }

  if(buf)
    free(buf);
  return len;
}

__device__ gint wg_idxhash_store(void* db, db_hash_area_header *ha,
  char* data, gint length, gint offset)
{
  db_memsegment_header* dbh = dbmemsegh(db);
  wg_uint hash;
  gint head_offset, head, bucket;
  gint rec_head, rec_offset;
  gcell *rec_cell;

  hash = hash_bytes(db, data, length, ha->arraylength);
  head_offset = (ha->arraystart)+(sizeof(gint) * hash);
  head = dbfetch(db, head_offset);

  /* Traverse the hash chain to check if there is a matching
   * hash string already
   */
  bucket = find_idxhash_bucket(db, data, length, &head_offset);
  if(!bucket) {
    size_t i;
    gint lengints, lenrest;
    char* dptr;

    /* Make a new bucket */
    lengints = length / sizeof(gint);
    lenrest = length % sizeof(gint);
    if(lenrest) lengints++;
    bucket = wg_alloc_gints(db,
         &(dbh->indexhash_area_header),
        lengints + HASHIDX_HEADER_SIZE);
    if(!bucket) {
      return -1;
    }

    /* Copy the byte data */
    dptr = (char *) (offsettoptr(db,
      bucket + HASHIDX_HEADER_SIZE*sizeof(gint)));
    memcpy(dptr, data, length);
    for(i=0;lenrest && i<sizeof(gint)-lenrest;i++) {
      *(dptr + length + i)=0; /* XXX: since we have the length, in meta,
                               * this is possibly unnecessary. */
    }

    /* Metadata */
    dbstore(db, bucket + HASHIDX_META_POS*sizeof(gint), length);
    dbstore(db, bucket + HASHIDX_RECLIST_POS*sizeof(gint), 0);

    /* Prepend to hash chain */
    dbstore(db, ((ha->arraystart)+(sizeof(gint) * hash)), bucket);
    dbstore(db, bucket + HASHIDX_HASHCHAIN_POS*sizeof(gint), head);
  }

  /* Add the record offset to the list. */
  rec_head = dbfetch(db, bucket + HASHIDX_RECLIST_POS*sizeof(gint));
  rec_offset = wg_alloc_fixlen_object(db, &(dbh->listcell_area_header));
  rec_cell = (gcell *) offsettoptr(db, rec_offset);
  rec_cell->car = offset;
  rec_cell->cdr = rec_head;
  dbstore(db, bucket + HASHIDX_RECLIST_POS*sizeof(gint), rec_offset);

  return 0;
}

__device__ gint wg_idxhash_remove(void* db, db_hash_area_header *ha,
  char* data, gint length, gint offset)
{
  wg_uint hash;
  gint bucket_offset, bucket;
  gint *next_offset, *reclist_offset;

  hash = hash_bytes(db, data, length, ha->arraylength);
  bucket_offset = (ha->arraystart)+(sizeof(gint) * hash); /* points to head */

  /* Find the correct bucket. */
  bucket = find_idxhash_bucket(db, data, length, &bucket_offset);
  if(!bucket) {
    return show_hash_error(db, "wg_idxhash_remove: Hash value not found.");
  }

  /* Remove the record offset from the list. */
  reclist_offset = (gint *)offsettoptr(db, bucket + HASHIDX_RECLIST_POS*sizeof(gint));
  next_offset = reclist_offset;
  while(*next_offset) {
    gcell *rec_cell = (gcell *) offsettoptr(db, *next_offset);
    if(rec_cell->car == offset) {
      gint rec_offset = *next_offset;
      *next_offset = rec_cell->cdr; /* remove from list chain */
      wg_free_listcell(db, rec_offset); /* free storage */
      goto is_bucket_empty;
    }
    next_offset = &(rec_cell->cdr);
  }
  return show_hash_error(db, "wg_idxhash_remove: Offset not found");

is_bucket_empty:
  if(!(*reclist_offset)) {
    gint nextchain = dbfetch(db, bucket + HASHIDX_HASHCHAIN_POS*sizeof(gint));
    dbstore(db, bucket_offset, nextchain);
    wg_free_object(db, &(dbmemsegh(db)->indexhash_area_header), bucket);
  }

  return 0;
}

__device__ gint wg_idxhash_find(void* db, db_hash_area_header *ha,
  char* data, gint length)
{
  wg_uint hash;
  gint head_offset, bucket;

  hash = hash_bytes(db, data, length, ha->arraylength);
  head_offset = (ha->arraystart)+(sizeof(gint) * hash); /* points to head */

  /* Find the correct bucket. */
  bucket = find_idxhash_bucket(db, data, length, &head_offset);
  if(!bucket)
    return 0;

  return dbfetch(db, bucket + HASHIDX_RECLIST_POS*sizeof(gint));
}

__device__ static gint show_consistency_error_nr(void* db, char* errmsg, gint nr) {
#ifdef WG_NO_ERRPRINT
#else
 // printf("wg consistency error: %s %d\n", errmsg, (int) nr);
  return -1;
#endif
}

__device__ static wg_uint hash_bytes(void *db, char *data, gint length, gint hashsz) {
  char* endp;
  wg_uint hash = 0;

  if (data!=NULL) {
    for(endp=data+length; data<endp; data++) {
      hash = *data + (hash << 6) + (hash << 16) - hash;
    }
  }
  return hash % hashsz;
}

__device__ static gint find_idxhash_bucket(void *db, char *data, gint length,
  gint *chainoffset)
{
  gint bucket = dbfetch(db, *chainoffset);
  while(bucket) {
    gint meta = dbfetch(db, bucket + HASHIDX_META_POS*sizeof(gint));
    if(meta == length) {
      /* Currently, meta stores just size */
      char *bucket_data = (char *)offsettoptr(db, bucket + \
        HASHIDX_HEADER_SIZE*sizeof(gint));
     // if(!memcmp(bucket_data, data, length))
      if(!memcmp(bucket_data, data, length))
        return bucket;
    }
    *chainoffset = bucket + HASHIDX_HASHCHAIN_POS*sizeof(gint);
    bucket = dbfetch(db, *chainoffset);
  }
  return 0;
}

__device__ int wg_hash_typedstr(void* db, char* data, char* extrastr, gint type, gint length) {
  char* endp;
  unsigned long hash = 0;
  int c;

  //printf("in wg_hash_typedstr %s %s %d %d \n",data,extrastr,type,length);
  if (data!=NULL) {
    for(endp=data+length; data<endp; data++) {
      c = (int)(*data);
      hash = c + (hash << 6) + (hash << 16) - hash;
    }
  }
  if (extrastr!=NULL) {
    while ((c = *extrastr++))
      hash = c + (hash << 6) + (hash << 16) - hash;
  }

  return (int)(hash % (dbmemsegh(db)->strhash_area_header).arraylength);
}

__device__ static gint show_hash_error(void* db, char* errmsg) {
#ifdef WG_NO_ERRPRINT
#else
  printf("wg hash error: %s\n",errmsg);
#endif
  return -1;
}
