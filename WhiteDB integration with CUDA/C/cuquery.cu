//#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dballoc.h"
#include "dbquery.h"
#include "dbcompare.h"
#include "dbdata.h"
#include "dbindex.h"
//#include "dbapi.h"

/* ======= Private protos ================ */

__device__
static gint find_ttree_bounds(void *db, gint index_id, gint col,
		gint start_bound, gint end_bound, gint start_inclusive, gint end_inclusive,
		gint *curr_offset, gint *curr_slot, gint *end_offset, gint *end_slot);

__device__
static gint check_arglist(void *db, void *rec, wg_query_arg *arglist,
		gint argc);

__device__
static gint find_ttree_bounds(void *db, gint index_id, gint col,
		gint start_bound, gint end_bound, gint start_inclusive, gint end_inclusive,
		gint *curr_offset, gint *curr_slot, gint *end_offset, gint *end_slot);

__device__
static gint show_query_error(void* db, char* errmsg);

/* ====== Functions ============== */
__device__
static gint check_arglist(void *db, void *rec, wg_query_arg *arglist,  // 285
		gint argc) {

	int i, reclen;

	reclen = wg_get_record_len(db, rec);
	for(i=0; i<argc; i++) {
		gint encoded;
		if(arglist[i].column < reclen)
		encoded = wg_get_field(db, rec, arglist[i].column);
		else
		return 0; /* XXX: should shorter records always fail?
		 * other possiblities here: compare to WG_ILLEGAL
		 * or WG_NULLTYPE. Current idea is based on SQL
		 * concept of comparisons to NULL always failing.
		 */

		switch(arglist[i].cond) {
			case WG_COND_EQUAL:
			if(WG_COMPARE(db, encoded, arglist[i].value) != WG_EQUAL)
			return 0;
			break;
			case WG_COND_LESSTHAN:
			if(WG_COMPARE(db, encoded, arglist[i].value) != WG_LESSTHAN)
			return 0;
			break;
			case WG_COND_GREATER:
			if(WG_COMPARE(db, encoded, arglist[i].value) != WG_GREATER)
			return 0;
			break;
			case WG_COND_LTEQUAL:
			if(WG_COMPARE(db, encoded, arglist[i].value) == WG_GREATER)
			return 0;
			break;
			case WG_COND_GTEQUAL:
			if(WG_COMPARE(db, encoded, arglist[i].value) == WG_LESSTHAN)
			return 0;
			break;
			case WG_COND_NOT_EQUAL:
			if(WG_COMPARE(db, encoded, arglist[i].value) == WG_EQUAL)
			return 0;
			break;
			default:
			break;
		}
	}

	return 1;
}

__device__
static gint find_ttree_bounds(void *db, gint index_id, gint col,  //  445
  gint start_bound, gint end_bound, gint start_inclusive, gint end_inclusive,
  gint *curr_offset, gint *curr_slot, gint *end_offset, gint *end_slot)
{
  /* hold the offsets temporarily */
  gint co = *curr_offset;
  gint cs = *curr_slot;
  gint eo = *end_offset;
  gint es = *end_slot;
  wg_index_header *hdr = (wg_index_header *) offsettoptr(db, index_id);
  struct wg_tnode *node;

  if(start_bound==WG_ILLEGAL) {
    /* Find leftmost node in index */
#ifdef TTREE_CHAINED_NODES
    co = TTREE_MIN_NODE(hdr);
#else
    /* LUB node search function has the useful property
     * of returning the leftmost node when called directly
     * on index root node */
    co = wg_ttree_find_lub_node(db, TTREE_ROOT_NODE(hdr));
#endif
    cs = 0; /* leftmost slot */
  } else {
    gint boundtype;

    if(start_inclusive) {
      /* In case of inclusive range, we get the leftmost
       * node for the given value and the first slot that
       * is equal or greater than the given value.
       */
      co = wg_search_ttree_leftmost(db,
        TTREE_ROOT_NODE(hdr), start_bound, &boundtype, NULL);
      if(boundtype == REALLY_BOUNDING_NODE) {
        cs = wg_search_tnode_first(db, co, start_bound, col);
        if(cs == -1) {
          show_query_error(db, "Starting index node was bad");
          return -1;
        }
      } else if(boundtype == DEAD_END_RIGHT_NOT_BOUNDING) {
        /* No exact match, but the next node should be in
         * range. */
        node = (struct wg_tnode *) offsettoptr(db, co);
        co = TNODE_SUCCESSOR(db, node);
        cs = 0;
      } else if(boundtype == DEAD_END_LEFT_NOT_BOUNDING) {
        /* Simplest case, values that are in range start
         * with this node. */
        cs = 0;
      }
    } else {
      /* For non-inclusive, we need the rightmost node and
       * the last slot+1. The latter may overflow into next node.
       */
      co = wg_search_ttree_rightmost(db,
        TTREE_ROOT_NODE(hdr), start_bound, &boundtype, NULL);
      if(boundtype == REALLY_BOUNDING_NODE) {
        cs = wg_search_tnode_last(db, co, start_bound, col);
        if(cs == -1) {
          show_query_error(db, "Starting index node was bad");
          return -1;
        }
        cs++;
        node = (struct wg_tnode *) offsettoptr(db, co);
        if(node->number_of_elements <= cs) {
          /* Crossed node boundary */
          co = TNODE_SUCCESSOR(db, node);
          cs = 0;
        }
      } else if(boundtype == DEAD_END_RIGHT_NOT_BOUNDING) {
        /* Since exact value was not found, this case is exactly
         * the same as with the inclusive range. */
        node = (struct wg_tnode *) offsettoptr(db, co);
        co = TNODE_SUCCESSOR(db, node);
        cs = 0;
      } else if(boundtype == DEAD_END_LEFT_NOT_BOUNDING) {
        /* No exact value in tree, same as inclusive range */
        cs = 0;
      }
    }
  }

  /* Finding of the end of the range is more or less opposite
   * of finding the beginning. */
  if(end_bound==WG_ILLEGAL) {
    /* Rightmost node in index */
#ifdef TTREE_CHAINED_NODES
    eo = TTREE_MAX_NODE(hdr);
#else
    /* GLB search on root node returns the rightmost node in tree */
    eo = wg_ttree_find_glb_node(db, TTREE_ROOT_NODE(hdr));
#endif
    if(eo) {
      node = (struct wg_tnode *) offsettoptr(db, eo);
      es = node->number_of_elements - 1; /* rightmost slot */
    }
  } else {
    gint boundtype;

    if(end_inclusive) {
      /* Find the rightmost node with a given value and the
       * righmost slot that is equal or smaller than that value
       */
      eo = wg_search_ttree_rightmost(db,
        TTREE_ROOT_NODE(hdr), end_bound, &boundtype, NULL);
      if(boundtype == REALLY_BOUNDING_NODE) {
        es = wg_search_tnode_last(db, eo, end_bound, col);
        if(es == -1) {
          show_query_error(db, "Ending index node was bad");
          return -1;
        }
      } else if(boundtype == DEAD_END_RIGHT_NOT_BOUNDING) {
        /* Last node containing values in range. */
        node = (struct wg_tnode *) offsettoptr(db, eo);
        es = node->number_of_elements - 1;
      } else if(boundtype == DEAD_END_LEFT_NOT_BOUNDING) {
        /* Previous node should be in range. */
        node = (struct wg_tnode *) offsettoptr(db, eo);
        eo = TNODE_PREDECESSOR(db, node);
        if(eo) {
          node = (struct wg_tnode *) offsettoptr(db, eo);
          es = node->number_of_elements - 1; /* rightmost */
        }
      }
    } else {
      /* For non-inclusive, we need the leftmost node and
       * the first slot-1.
       */
      eo = wg_search_ttree_leftmost(db,
        TTREE_ROOT_NODE(hdr), end_bound, &boundtype, NULL);
      if(boundtype == REALLY_BOUNDING_NODE) {
        es = wg_search_tnode_first(db, eo,
          end_bound, col);
        if(es == -1) {
          show_query_error(db, "Ending index node was bad");
          return -1;
        }
        es--;
        if(es < 0) {
          /* Crossed node boundary */
          node = (struct wg_tnode *) offsettoptr(db, eo);
          eo = TNODE_PREDECESSOR(db, node);
          if(eo) {
            node = (struct wg_tnode *) offsettoptr(db, eo);
            es = node->number_of_elements - 1;
          }
        }
      } else if(boundtype == DEAD_END_RIGHT_NOT_BOUNDING) {
        /* No exact value in tree, same as inclusive range */
        node = (struct wg_tnode *) offsettoptr(db, eo);
        es = node->number_of_elements - 1;
      } else if(boundtype == DEAD_END_LEFT_NOT_BOUNDING) {
        /* No exact value in tree, same as inclusive range */
        node = (struct wg_tnode *) offsettoptr(db, eo);
        eo = TNODE_PREDECESSOR(db, node);
        if(eo) {
          node = (struct wg_tnode *) offsettoptr(db, eo);
          es = node->number_of_elements - 1; /* rightmost slot */
        }
      }
    }
  }

  /* Now detect the cases where the above bound search
   * has produced a result with an empty range.
   */
  if(co) {
    /* Value could be bounded inside a node, but actually
     * not present. Note that we require the end_slot to be
     * >= curr_slot, this implies that query->direction == 1.
     */
    if(eo == co && es < cs) {
      co = 0; /* query will return no rows */
      eo = 0;
    } else if(!eo) {
      /* If one offset is 0 the other should be forced to 0, so that
       * if we want to switch direction we won't run into any surprises.
       */
      co = 0;
    } else {
      /* Another case we have to watch out for is when we have a
       * range that fits in the space between two nodes. In that case
       * the end offset will end up directly left of the start offset.
       */
      node = (struct wg_tnode *) offsettoptr(db, co);
      if(eo == TNODE_PREDECESSOR(db, node)) {
        co = 0; /* no rows */
        eo = 0;
      }
    }
  } else {
    eo = 0; /* again, if one offset is 0,
             * the other should be, too */
  }

  *curr_offset = co;
  *curr_slot = cs;
  *end_offset = eo;
  *end_slot = es;
  return 0;
}

__device__
gint wg_free_query_param(void* db, gint data) {  //1294
#ifdef CHECK
  if (!dbcheck(db)) {
    show_query_error(db,"wrong database pointer given to wg_free_query_param");
    return 0;
  }
#endif
  if (isptr(data)) {
    gint offset;

    switch(data&NORMALPTRMASK) {
      case DATARECBITS:
        break;
      case SHORTSTRBITS:
        offset = decode_shortstr_offset(data);
        free(offsettoptr(db, offset));
        break;
      case LONGSTRBITS:
        offset = decode_longstr_offset(data);
        free(offsettoptr(db, offset));
        break;
      case FULLDOUBLEBITS:
        offset = decode_fulldouble_offset(data);
        free(offsettoptr(db, offset));
        break;
      case FULLINTBITSV0:
      case FULLINTBITSV1:
        offset = decode_fullint_offset(data);
        free(offsettoptr(db, offset));
        break;
      default:
        show_query_error(db,"Bad encoded value given to wg_free_query_param");
        break;
    }
  }
  return 0;
}

__device__ gint wg_encode_query_param_int(void *db, gint data) {
  void *dptr;

  if(fits_smallint(data)) {
    return encode_smallint(data);
  } else {
    dptr=malloc(sizeof(gint));
    if(!dptr) {
      show_query_error(db, "Failed to encode query parameter");
      return WG_ILLEGAL;
    }
    *((gint *) dptr) = data;
    return encode_fullint_offset(ptrtooffset(db, dptr));
  }
}

__device__
void *wg_find_record(void *db, gint fieldnr, gint cond, gint data,  // 2035
    void* lastrecord) {
  gint index_id = -1;

  /* find index on colum */
  if(cond != WG_COND_NOT_EQUAL) {
    index_id = wg_multi_column_to_index_id(db, &fieldnr, 1,
      WG_INDEX_TYPE_TTREE, NULL, 0);
  }

  if(index_id > 0) {
    int start_inclusive = 1, end_inclusive = 1;
    /* WG_ILLEGAL is interpreted as "no bound" */
    gint start_bound = WG_ILLEGAL;
    gint end_bound = WG_ILLEGAL;
    gint curr_offset = 0, curr_slot = -1, end_offset = 0, end_slot = -1;
    void *prev = NULL;

    switch(cond) {
      case WG_COND_EQUAL:
        start_bound = end_bound = data;
        break;
      case WG_COND_LESSTHAN:
        end_bound = data;
        end_inclusive = 0;
        break;
      case WG_COND_GREATER:
        start_bound = data;
        start_inclusive = 0;
        break;
      case WG_COND_LTEQUAL:
        end_bound = data;
        break;
      case WG_COND_GTEQUAL:
        start_bound = data;
        break;
      default:
        show_query_error(db, "Invalid condition (ignoring)");
        return NULL;
    }

    if(find_ttree_bounds(db, index_id, fieldnr,
        start_bound, end_bound, start_inclusive, end_inclusive,
        &curr_offset, &curr_slot, &end_offset, &end_slot)) {
      return NULL;
    }

    /* We have the bounds, scan to lastrecord */
    while(curr_offset) {
      struct wg_tnode *node = (struct wg_tnode *) offsettoptr(db, curr_offset);
      void *rec = offsettoptr(db, node->array_of_values[curr_slot]);

      if(prev == lastrecord) {
        /* if lastrecord is NULL, first match returned */
        return rec;
      }

      prev = rec;
      if(curr_offset==end_offset && curr_slot==end_slot) {
        /* Last slot reached */
        break;
      } else {
        /* Some rows still left */
        curr_slot += 1; /* direction implied as 1 */
        if(curr_slot >= node->number_of_elements) {
#ifdef CHECK
          if(end_offset==curr_offset) {
            /* This should not happen */
            show_query_error(db, "Warning: end slot mismatch, possible bug");
            break;
          } else {
#endif
            curr_offset = TNODE_SUCCESSOR(db, node);
            curr_slot = 0;
#ifdef CHECK
          }
#endif
        }
      }
    }
  }
  else {
    /* no index (or cond == WG_COND_NOT_EQUAL), do a scan */
    wg_query_arg arg;
    void *rec;

    if(lastrecord) {
      rec = wg_get_next_record(db, lastrecord);
    } else {
      rec = wg_get_first_record(db);
    }

    arg.column = fieldnr;
    arg.cond = cond;
    arg.value = data;

    while(rec) {
      if(check_arglist(db, rec, &arg, 1)) {
        return rec;
      }
      rec = wg_get_next_record(db, rec);
    }
  }

  /* No records found (this can also happen if matching records were
   * found but lastrecord does not match any of them or matches the
   * very last one).
   */
  return NULL;
}

__device__
void *wg_cuda_find_record_int(void *db, gint fieldnr, gint cond, int data, // 2219
    void* lastrecord) {
  gint enc = wg_encode_query_param_int(db, data);
  void *rec = wg_find_record(db, fieldnr, cond, enc, lastrecord);
  wg_free_query_param(db, enc);
  return rec;
}

__device__
void *wg_find_record_int(void *db, gint fieldnr, gint cond, int data, // 2219
    void* lastrecord) {
  gint enc = wg_encode_query_param_int(db, data);
  void *rec = wg_find_record(db, fieldnr, cond, enc, lastrecord);
  wg_free_query_param(db, enc);
  return rec;
}

__device__
static gint show_query_error(void* db, char* errmsg) {  // 2279
#ifdef WG_NO_ERRPRINT
#else
  //fprintf(stderr,"query error: %s\n",errmsg);
	//printf("\nquery error: %s\n",errmsg);
#endif
  return -1;
}
