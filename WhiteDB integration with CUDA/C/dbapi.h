/*
* $Id:  $
* $Version: $
*
* Copyright (c) Tanel Tammet 2004,2005,2006,2007,2008,2009
* Copyright (c) Priit Järv 2009,2010,2011,2013,2014
*
* Contact: tanel.tammet@gmail.com
*
* This file is part of WhiteDB
*
* WhiteDB is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* WhiteDB is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with WhiteDB.  If not, see <http://www.gnu.org/licenses/>.
*
*/

 /** @file dbapi.h
 *
 * Wg database api for public use.
 *
 */

#ifndef DEFINED_DBAPI_H
#define DEFINED_DBAPI_H

/* For gint/wg_int types */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---  built-in data type numbers ----- */

/* the built-in data types are primarily for api purposes.
   internally, some of these types like int, str etc have several
   different ways to encode along with different bit masks
*/


#define WG_NULLTYPE 1
#define WG_RECORDTYPE 2
#define WG_INTTYPE 3
#define WG_DOUBLETYPE 4
#define WG_STRTYPE 5
#define WG_XMLLITERALTYPE 6
#define WG_URITYPE 7
#define WG_BLOBTYPE 8
#define WG_CHARTYPE 9
#define WG_FIXPOINTTYPE 10
#define WG_DATETYPE 11
#define WG_TIMETYPE 12
#define WG_ANONCONSTTYPE 13
#define WG_VARTYPE 14

/* Illegal encoded data indicator */
#define WG_ILLEGAL 0xff

/* Query "arglist" parameters */
#define WG_COND_EQUAL       0x0001      /** = */
#define WG_COND_NOT_EQUAL   0x0002      /** != */
#define WG_COND_LESSTHAN    0x0004      /** < */
#define WG_COND_GREATER     0x0008      /** > */
#define WG_COND_LTEQUAL     0x0010      /** <= */
#define WG_COND_GTEQUAL     0x0020      /** >= */

/* Query types. Python extension module uses the API and needs these. */
#define WG_QTYPE_TTREE      0x01
#define WG_QTYPE_HASH       0x02
#define WG_QTYPE_SCAN       0x04
#define WG_QTYPE_PREFETCH   0x80

/* Direct access to field */
#define RECORD_HEADER_GINTS 3
#define wg_field_addr(db,record,fieldnr) (((wg_int*)(record))+RECORD_HEADER_GINTS+(fieldnr))

/* WhiteDB data types */

typedef ptrdiff_t wg_int;
typedef size_t wg_uint;

/** Query argument list object */
typedef struct {
  wg_int column;      /** column (field) number this argument applies to */
  wg_int cond;        /** condition (equal, less than, etc) */
  wg_int value;       /** encoded value */
} wg_query_arg;

/** Query object */
typedef struct {
  wg_int qtype;         /** Query type (T-tree, hash, full scan, prefetch) */
  /* Argument list based query is the only one supported at the moment. */
  wg_query_arg *arglist;    /** check each row in result set against these */
  wg_int argc;              /** number of elements in arglist */
  wg_int column;            /** index on this column used */
  /* Fields for T-tree query (XXX: some may be re-usable for
   * other types as well) */
  wg_int curr_offset;
  wg_int end_offset;
  wg_int curr_slot;
  wg_int end_slot;
  wg_int direction;
  /* Fields for full scan */
  wg_int curr_record;       /** offset of the current record */
  /* Fields for prefetch; with/without mpool */
  void *mpool;              /** storage for row offsets */
  void *curr_page;          /** current page of results */
  wg_int curr_pidx;         /** current index on page */
  wg_uint res_count;        /** number of rows in results */
} wg_query;

/* prototypes of wg database api functions

*/

#ifdef __cplusplus
}
#endif

#endif /* DEFINED_DBAPI_H */
