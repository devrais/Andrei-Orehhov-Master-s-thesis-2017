/** Find leftmost node containing given value
 *  returns NULL if node was not found
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../config.h"

#include "dbdata.h"
#include "dbindex.h"
#include "dbcompare.h"
#include "dbhash.h"

/* ====== Private defs =========== */

#define LL_CASE 0
#define LR_CASE 1
#define RL_CASE 2
#define RR_CASE 3

#ifndef max
#define max(a,b) (a>b ? a : b)
#endif

#define HASHIDX_OP_STORE 1
#define HASHIDX_OP_REMOVE 2
#define HASHIDX_OP_FIND 3

#ifndef TTREE_SINGLE_COMPARE

#else
/* "rightmost" node search is the improved tree search described in
 * the original T-tree paper.
 */
#define db_find_bounding_tnode wg_search_ttree_rightmost
#endif

#define INDEX_ADD_ROW(d, h, i, r) \
  switch(h->type) { \
    case WG_INDEX_TYPE_TTREE: \
      if(ttree_add_row(d, i, r)) \
        return -2; \
      break; \
    case WG_INDEX_TYPE_TTREE_JSON: \
      if(is_plain_record(r)) { \
        if(ttree_add_row(d, i, r)) \
          return -2; \
      } \
      break; \
    case WG_INDEX_TYPE_HASH: \
      if(hash_add_row(d, i, r)) \
        return -2; \
      break; \
    case WG_INDEX_TYPE_HASH_JSON: \
      if(is_plain_record(r)) { \
        if(hash_add_row(d, i, r)) \
          return -2; \
      } \
      break; \
    default: \
      show_index_error(db, "unknown index type, ignoring"); \
      break; \
  }

#define INDEX_REMOVE_ROW(d, h, i, r) \
  switch(h->type) { \
    case WG_INDEX_TYPE_TTREE: \
      if(ttree_remove_row(d, i, r) < -2) \
        return -2; \
      break; \
    case WG_INDEX_TYPE_TTREE_JSON: \
      if(is_plain_record(r)) { \
        if(ttree_remove_row(d, i, r) < -2) \
          return -2; \
      } \
      break; \
    case WG_INDEX_TYPE_HASH: \
      if(hash_remove_row(d, i, r) < -2) \
        return -2; \
      break; \
    case WG_INDEX_TYPE_HASH_JSON: \
      if(is_plain_record(r)) { \
        if(hash_remove_row(d, i, r) < -2) \
          return -2; \
      } \
      break; \
    default: \
      show_index_error(db, "unknown index type, ignoring"); \
      break; \
  }

/* ======= Private protos ================ */
__device__  static gint ttree_add_row(void *db, gint index_id, void *rec);
__device__  static gint hash_add_row(void *db, gint index_id, void *rec);
__device__  static gint show_index_error(void* db, char* errmsg);

__device__  static gint hash_recurse(void *db, wg_index_header *hdr,
		char *prefix, gint prefixlen, gint *values, gint count, void *rec,
		gint op, gint expand);
__device__ static gint hash_extend_prefix(void *db, wg_index_header *hdr, char *prefix,
  gint prefixlen, gint nextval, gint *values, gint count, void *rec, gint op,
  gint expand);

__device__ static int db_rotate_ttree(void *db, gint index_id, struct wg_tnode *root,
  int overw);

__device__ static int db_which_branch_causes_overweight(void *db, struct wg_tnode *root);
__device__ static gint ttree_remove_row(void *db, gint index_id, void * rec);

__device__ static gint hash_remove_row(void *db, gint index_id, void *rec);

/* ====== Functions ============== */

__device__ static gint hash_extend_prefix(void *db, wg_index_header *hdr, char *prefix,
  gint prefixlen, gint nextval, gint *values, gint count, void *rec, gint op,
  gint expand) {

  char *fldbytes, *newprefix;
  gint newlen, fldlen, retv;

  fldlen = wg_decode_for_hashing(db, nextval, &fldbytes);
  if(fldlen < 1) {
    show_index_error(db,"Failed to decode a field value for hash");
    return -1;
  }

  if(prefix && prefixlen) {
    newlen = prefixlen + fldlen + 1;
  } else {
    newlen = fldlen;
  }

  newprefix = (char *)malloc(newlen);
  if(!newprefix) {
    free(fldbytes);
    show_index_error(db, "Failed to allocate memory");
    return -1;
  }
  if(prefix) {
    memcpy(newprefix, prefix, prefixlen);
    newprefix[prefixlen] = '\0'; /* XXX: why? double-check this */
  }

  memcpy(newprefix + (newlen - fldlen), fldbytes, fldlen);
  retv = hash_recurse(db, hdr, newprefix,
    newlen, values, count, rec, op, expand);
  free(fldbytes);
  free(newprefix);
  return retv;
}

__device__ static gint hash_recurse(void *db, wg_index_header *hdr, char *prefix,
  gint prefixlen, gint *values, gint count, void *rec, gint op, gint expand) {

  if(count) {
    gint nextvalue = values[0];
    if(expand) {
      /* In case of a JSON/array index, check the value */
      if(wg_get_encoded_type(db, nextvalue) == WG_RECORDTYPE) {
        void *valrec = wg_decode_record(db, nextvalue);

        if(is_schema_array(valrec)) {
          /* expand the array */
          gint i, reclen, retv = 0;
          reclen = wg_get_record_len(db, valrec);
          for(i=0; i<reclen; i++) {
            retv = hash_extend_prefix(db, hdr, prefix, prefixlen,
              wg_get_field(db, valrec, i),
              &values[1], count - 1, rec, op, expand);
            if(retv)
              break;
          }
          return retv; /* This skips adding the array record itself. It's
                        * not useful as we can only hash the offset. */
        }
      }
    }
    /* Regular index. JSON/array index also falls back to this. */
    return hash_extend_prefix(db, hdr, prefix, prefixlen,
      nextvalue, &values[1], count - 1, rec, op, expand);
  }
  else {
    /* No more values, the hash string is complete. Add it to the index */
    if(op == HASHIDX_OP_STORE) {
      return wg_idxhash_store(db, HASHIDX_ARRAYP(hdr),
        prefix, prefixlen, ptrtooffset(db, rec));
    } else if(op == HASHIDX_OP_REMOVE) {
      return wg_idxhash_remove(db, HASHIDX_ARRAYP(hdr),
        prefix, prefixlen, ptrtooffset(db, rec));
    } else {
      /* assume HASHIDX_OP_FIND */
      return wg_idxhash_find(db, HASHIDX_ARRAYP(hdr), prefix, prefixlen);
    }
  }
  return 0; /* pacify the compiler */
}

__device__  static gint hash_add_row(void *db, gint index_id, void *rec) {
	wg_index_header *hdr = (wg_index_header *) offsettoptr(db, index_id);
	gint i;
	gint values[MAX_INDEX_FIELDS];

	for (i = 0; i < hdr->fields; i++) {
		values[i] = wg_get_field(db, rec, hdr->rec_field_index[i]);
	}
	return hash_recurse(db, hdr, NULL, 0, values, hdr->fields, rec,
	HASHIDX_OP_STORE, (hdr->type == WG_INDEX_TYPE_HASH_JSON));
}

__device__  static gint ttree_add_row(void *db, gint index_id, void *rec) {
	gint rootoffset, column;
	gint newvalue, boundtype, bnodeoffset, newoffset;
	struct wg_tnode *node;
	wg_index_header *hdr = (wg_index_header *) offsettoptr(db, index_id);
	db_memsegment_header* dbh = dbmemsegh(db);

	rootoffset = TTREE_ROOT_NODE(hdr);
#ifdef CHECK
	if (rootoffset == 0) {
#ifdef WG_NO_ERRPRINT
#else
		//printf("index at offset %d does not exist\n", (int) index_id);
#endif
		return -1;
	}
#endif
	column = hdr->rec_field_index[0]; /* always one column for T-tree */

	//extract real value from the row (rec)
	newvalue = wg_get_field(db, rec, column);

	//find bounding node for the value
	bnodeoffset = db_find_bounding_tnode(db, rootoffset, newvalue, &boundtype,
			NULL);
	node = (struct wg_tnode *) offsettoptr(db, bnodeoffset);
	newoffset = 0; //save here the offset of newly created tnode - 0 if no node added into the tree
	//if bounding node exists - follow one algorithm, else the other
	if (boundtype == REALLY_BOUNDING_NODE) {

		//check if the node has room for a new entry
		if (node->number_of_elements < WG_TNODE_ARRAY_SIZE) {
			int i, j;
			gint cr;

			/* add array entry and update control data. We keep the
			 * array sorted, smallest values left. */
			for (i = 0; i < node->number_of_elements; i++) {
				/* The node is small enough for naive scans to be
				 * "good enough" inside the node. Note that we
				 * branch into re-sort loop as early as possible
				 * with >= operator (> would be algorithmically correct too)
				 * since here the compare is more expensive than the slot
				 * copying.
				 */
				cr =
						WG_COMPARE(db,
								wg_get_field(db, (void *)offsettoptr(db,node->array_of_values[i]), column),
								newvalue);

				if (cr != WG_LESSTHAN) { /* value >= newvalue */
					/* Push remaining values to the right */
					for (j = node->number_of_elements; j > i; j--)
						node->array_of_values[j] = node->array_of_values[j - 1];
					break;
				}
			}
			/* i is either number_of_elements or a vacated slot
			 * in the array now. */
			node->array_of_values[i] = ptrtooffset(db, rec);
			node->number_of_elements++;

			/* Update min. Due to the >= comparison max is preserved
			 * in this case. Note that we are overwriting values that
			 * WG_COMPARE() may deem equal. This is intentional, because other
			 * parts of T-tree algorithm rely on encoded values of min/max fields
			 * to be in sync with the leftmost/rightmost slots.
			 */
			if (i == 0) {
				node->current_min = newvalue;
			}
		} else {
			//still, insert the value here, but move minimum out of this node
			//get the minimum element from this node
			int i, j;
			gint cr, minvalue, minvaluerowoffset;

			minvalue = node->current_min;
			minvaluerowoffset = node->array_of_values[0];

			/* Now scan for the matching slot. However, since
			 * we already know the 0 slot will be re-filled, we
			 * do this scan (and sort) in reverse order, compared to the case
			 * where array had some space left. */
			for (i = WG_TNODE_ARRAY_SIZE - 1; i > 0; i--) {
				cr =
						WG_COMPARE(db,
								wg_get_field(db, (void *)offsettoptr(db,node->array_of_values[i]), column),
								newvalue);
				if (cr != WG_GREATER) { /* value <= newvalue */
					/* Push remaining values to the left */
					for (j = 0; j < i; j++)
						node->array_of_values[j] = node->array_of_values[j + 1];
					break;
				}
			}
			/* i is either 0 or a freshly vacated slot */
			node->array_of_values[i] = ptrtooffset(db, rec);

			/* Update minimum. Thanks to the sorted array, we know for a fact
			 * that the minimum sits in slot 0. */
			if (i == 0) {
				node->current_min = newvalue;
			} else {
				node->current_min = wg_get_field(db,
						(void *) offsettoptr(db, node->array_of_values[0]),
						column);
				/* The scan for the free slot starts from the right and
				 * tries to exit as fast as possible. So it's possible that
				 * the rightmost slot was changed.
				 */
				if (i == WG_TNODE_ARRAY_SIZE - 1) {
					node->current_max = newvalue;
				}
			}

			//proceed to the node that holds greatest lower bound - must be leaf (can be the initial bounding node)
			if (node->left_child_offset != 0) {
#ifndef TTREE_CHAINED_NODES
				gint greatestlb = wg_ttree_find_glb_node(db,node->left_child_offset);
#else
				gint greatestlb = node->pred_offset;
#endif
				node = (struct wg_tnode *) offsettoptr(db, greatestlb);
			}
			//if the greatest lower bound node has room, insert value
			//otherwise make the new node as right child and put the value there
			if (node->number_of_elements < WG_TNODE_ARRAY_SIZE) {
				//add array entry and update control data
				node->array_of_values[node->number_of_elements] =
						minvaluerowoffset;    //save offset, use first free slot
				node->number_of_elements++;
				node->current_max = minvalue;

			} else {
				//create, initialize and save first value
				struct wg_tnode *leaf;
				gint newnode = wg_alloc_fixlen_object(db,
						&dbh->tnode_area_header);
				if (newnode == 0)
					return -1;
				leaf = (struct wg_tnode *) offsettoptr(db, newnode);
				leaf->parent_offset = ptrtooffset(db, node);
				leaf->left_subtree_height = 0;
				leaf->right_subtree_height = 0;
				leaf->current_max = minvalue;
				leaf->current_min = minvalue;
				leaf->number_of_elements = 1;
				leaf->left_child_offset = 0;
				leaf->right_child_offset = 0;
				leaf->array_of_values[0] = minvaluerowoffset;
				/* If the original, full node did not have a left child, then
				 * there also wasn't a separate GLB node, so we are adding one now
				 * as the left child. Otherwise, the new node is added as the right
				 * child to the current GLB node.
				 */
				if (bnodeoffset == ptrtooffset(db, node)) {
					node->left_child_offset = newnode;
#ifdef TTREE_CHAINED_NODES
					/* Create successor / predecessor relationship */
					leaf->succ_offset = ptrtooffset(db, node);
					leaf->pred_offset = node->pred_offset;

					if (node->pred_offset) {
						struct wg_tnode *pred = (struct wg_tnode *) offsettoptr(
								db, node->pred_offset);
						pred->succ_offset = newnode;
					} else {
						TTREE_MIN_NODE(hdr) = newnode;
					}
					node->pred_offset = newnode;
#endif
				} else {
#ifdef TTREE_CHAINED_NODES
					struct wg_tnode *succ;
#endif
					node->right_child_offset = newnode;
#ifdef TTREE_CHAINED_NODES
					/* Insert the new node in the sequential chain between
					 * the original node and the GLB node found */
					leaf->succ_offset = node->succ_offset;
					leaf->pred_offset = ptrtooffset(db, node);

#ifdef CHECK
					if (!node->succ_offset) {
						show_index_error(db, "GLB with no successor, panic");
						return -1;
					} else {
#endif
						succ = (struct wg_tnode *) offsettoptr(db,
								leaf->succ_offset);
						succ->pred_offset = newnode;
#ifdef CHECK
					}
#endif
					node->succ_offset = newnode;
#endif /* TTREE_CHAINED_NODES */
				}
				newoffset = newnode;
			}
		}

	}      //the bounding node existed - first algorithm
	else {      // bounding node does not exist
				//try to insert the new value to that node - becoming new min or max
				//if the node has room for a new entry
		if (node->number_of_elements < WG_TNODE_ARRAY_SIZE) {
			int i;

			/* add entry, keeping the array sorted (see also notes for the
			 * bounding node case. The difference this time is that we already
			 * know if this value is becoming the new min or max).
			 */
			if (boundtype == DEAD_END_LEFT_NOT_BOUNDING) {
				/* our new value is the new min, push everything right */
				for (i = node->number_of_elements; i > 0; i--)
					node->array_of_values[i] = node->array_of_values[i - 1];
				node->array_of_values[0] = ptrtooffset(db, rec);
				node->current_min = newvalue;
			} else { /* DEAD_END_RIGHT_NOT_BOUNDING */
				/* even simpler case, new value is added to the right */
				node->array_of_values[node->number_of_elements] = ptrtooffset(
						db, rec);
				node->current_max = newvalue;
			}

			node->number_of_elements++;

			/* XXX: not clear if the empty node can occur here. Until this
			 * is checked, we'll be paranoid and overwrite both min and max. */
			if (node->number_of_elements == 1) {
				node->current_max = newvalue;
				node->current_min = newvalue;
			}
		} else {
			//make a new node and put data there
			struct wg_tnode *leaf;
			gint newnode = wg_alloc_fixlen_object(db, &dbh->tnode_area_header);
			if (newnode == 0)
				return -1;
			leaf = (struct wg_tnode *) offsettoptr(db, newnode);
			leaf->parent_offset = ptrtooffset(db, node);
			leaf->left_subtree_height = 0;
			leaf->right_subtree_height = 0;
			leaf->current_max = newvalue;
			leaf->current_min = newvalue;
			leaf->number_of_elements = 1;
			leaf->left_child_offset = 0;
			leaf->right_child_offset = 0;
			leaf->array_of_values[0] = ptrtooffset(db, rec);
			newoffset = newnode;
			//set new node as left or right leaf
			if (boundtype == DEAD_END_LEFT_NOT_BOUNDING) {
				node->left_child_offset = newnode;
#ifdef TTREE_CHAINED_NODES
				/* Set the new node as predecessor of the parent */
				leaf->succ_offset = ptrtooffset(db, node);
				leaf->pred_offset = node->pred_offset;

				if (node->pred_offset) {
					/* Notify old predecessor that the node following
					 * it changed */
					struct wg_tnode *pred = (struct wg_tnode *) offsettoptr(db,
							node->pred_offset);
					pred->succ_offset = newnode;
				} else {
					TTREE_MIN_NODE(hdr) = newnode;
				}
				node->pred_offset = newnode;
#endif
			} else if (boundtype == DEAD_END_RIGHT_NOT_BOUNDING) {
				node->right_child_offset = newnode;
#ifdef TTREE_CHAINED_NODES
				/* Set the new node as successor of the parent */
				leaf->succ_offset = node->succ_offset;
				leaf->pred_offset = ptrtooffset(db, node);

				if (node->succ_offset) {
					/* Notify old successor that the node preceding
					 * it changed */
					struct wg_tnode *succ = (struct wg_tnode *) offsettoptr(db,
							node->succ_offset);
					succ->pred_offset = newnode;
				} else {
					TTREE_MAX_NODE(hdr) = newnode;
				}
				node->succ_offset = newnode;
#endif
			}
		}
	}    //no bounding node found - algorithm 2

	//if new node was added to tree - must update child height data in nodes from leaf to root
	//or until find a node with imbalance
	//then determine the bad balance case: LL, LR, RR or RL and execute proper rotation
	if (newoffset) {
		struct wg_tnode *child = (struct wg_tnode *) offsettoptr(db, newoffset);
		struct wg_tnode *parent;
		int left = 0;
		while (child->parent_offset != 0) {  //this is not a root
			int balance;
			parent = (struct wg_tnode *) offsettoptr(db, child->parent_offset);
			//determine which child the child is, left or right one
			if (parent->left_child_offset == ptrtooffset(db, child))
				left = 1;
			else
				left = 0;
			//increment parent left or right subtree height
			if (left)
				parent->left_subtree_height++;
			else
				parent->right_subtree_height++;

			//check balance
			balance = parent->left_subtree_height
					- parent->right_subtree_height;
			if (balance == 0) {
				/* As a result of adding a new node somewhere below, left
				 * and right subtrees of the node we're checking became
				 * of EQUAL height. This means that changes in subtree heights
				 * do not propagate any further (the max depth in this node
				 * dit NOT change).
				 */
				break;
			}
			if (balance > 1 || balance < -1) {  //must rebalance
			//the current parent is root for balancing operation
			//determine the branch that causes overweight
				int overw = db_which_branch_causes_overweight(db, parent);
				//fix balance
				db_rotate_ttree(db, index_id, parent, overw);
				break; //while loop because balance does not change in the next levels
			} else {        //just proceed to the parent node
				child = parent;
			}
		}
	}
	return 0;
}

__device__ gint wg_index_add_rec(void *db, void *rec) {
	gint i;
	db_memsegment_header* dbh = dbmemsegh(db);
	gint reclen = wg_get_record_len(db, rec);

#ifdef CHECK
	if (is_special_record(rec))
		return -1;
#endif

	if (reclen > MAX_INDEXED_FIELDNR)
		reclen = MAX_INDEXED_FIELDNR + 1;

	for (i = 0; i < reclen; i++) {
		gint *ilist;
		gcell *ilistelem;

		/* Find all indexes on the column */
		ilist = &dbh->index_control_area_header.index_table[i];
		while (*ilist) {
			ilistelem = (gcell *) offsettoptr(db, *ilist);
			if (ilistelem->car) {
				wg_index_header *hdr = (wg_index_header *) offsettoptr(db,
						ilistelem->car);
				if (hdr->rec_field_index[hdr->fields - 1] == i) {
					/* Only add the record if we're at the last column
					 * of the index. This way we ensure that a.) a record
					 * is entered once into a multi-column index and b.) the
					 * record is long enough so that it qualifies for the
					 * multi-column index.
					 * For a single-column index, the indexed column is
					 * also the last column, therefore the above is valid,
					 * altough the check is unnecessary.
					 */
					if (MATCH_TEMPLATE(db, hdr, rec)) {
						INDEX_ADD_ROW(db, hdr, ilistelem->car, rec)
					}
				}
			}
			ilist = &ilistelem->cdr;
		}

#ifdef USE_INDEX_TEMPLATE
		ilist = &dbh->index_control_area_header.index_template_table[i];
		while (*ilist) {
			ilistelem = (gcell *) offsettoptr(db, *ilist);
			if (ilistelem->car) {
				wg_index_header *hdr = (wg_index_header *) offsettoptr(db,
						ilistelem->car);
				wg_index_template *tmpl = (wg_index_template *) offsettoptr(db,
						hdr->template_offset);
				void *matchrec;
				gint mreclen;
				int j, firstmatch = -1;

				/* Here the check for a match is slightly more complicated.
				 * If there is a match *but* the current column is not the
				 * first fixed one in the template, the match has
				 * already occurred earlier.
				 */
				matchrec = offsettoptr(db, tmpl->offset_matchrec);
				mreclen = wg_get_record_len(db, matchrec);
				if (mreclen > reclen) {
					goto nexttmpl1;
				}
				for (j = 0; j < mreclen; j++) {
					gint enc = wg_get_field(db, matchrec, j);
					if (wg_get_encoded_type(db, enc) != WG_VARTYPE) {
						if (WG_COMPARE(db, enc,
								wg_get_field(db, rec, j)) != WG_EQUAL)
							goto nexttmpl1;
						if (firstmatch < 0)
							firstmatch = j;
					}
				}
				if (firstmatch == i
						&& reclen > hdr->rec_field_index[hdr->fields - 1]) {
					/* The record matches AND this is the first time we
					 * see this index. Update it.
					 */
					INDEX_ADD_ROW(db, hdr, ilistelem->car, rec)
				}
			}
			nexttmpl1: ilist = &ilistelem->cdr;
		}
#endif

	}
	return 0;
}

#ifdef USE_INDEX_TEMPLATE
__device__
   static gint find_index_template(void *db, gint *matchrec,
		gint reclen);
#endif

__device__
   static gint sort_columns(gint *sorted_cols, gint *columns,
		gint col_count);
__device__
 static gint show_index_error(void* db, char* errmsg);
__device__
 static gint show_index_error_nr(void* db, char* errmsg, gint nr);

__device__ gint wg_search_ttree_rightmost(void *db, gint rootoffset,  // 1180
		gint key, gint *result, struct wg_tnode *rb_node) {

	struct wg_tnode * node;

#ifdef TTREE_SINGLE_COMPARE
	node = (struct wg_tnode *) offsettoptr(db, rootoffset);

	/* Improved(?) tree search algorithm with a single compare per node.
	 * only lower bound is examined, if the value is larger the right subtree
	 * is selected immediately. If the search ends in a dead end, the node where
	 * the right branch was taken is examined again.
	 */
	if (WG_COMPARE(db, key, node->current_min) == WG_LESSTHAN) {
		/* key < node->current_min */
		if (node->left_child_offset != 0) {
			return wg_search_ttree_rightmost(db, node->left_child_offset, key,
					result, rb_node);
		} else if (rb_node) {
			/* Dead end, but we still have an unexamined node left */
			if (WG_COMPARE(db, key, rb_node->current_max) != WG_GREATER) {
				/* key<=rb_node->current_max */
				*result = REALLY_BOUNDING_NODE;
				return ptrtooffset(db, rb_node);
			}
		}
		/* No left child, no rb_node or it's right bound was not interesting */
		*result = DEAD_END_LEFT_NOT_BOUNDING;
		return rootoffset;
	} else {
		if (node->right_child_offset != 0) {
			/* Here we jump the gun and branch to right, ignoring the
			 * current_max of the node (therefore avoiding one expensive
			 * compare operation).
			 */
			return wg_search_ttree_rightmost(db, node->right_child_offset, key,
					result, node);
		} else if (WG_COMPARE(db, key, node->current_max) != WG_GREATER) {
			/* key<=node->current_max */
			*result = REALLY_BOUNDING_NODE;
			return rootoffset;
		}
		/* key is neither left of or inside this node and
		 * there is no right child */
		*result = DEAD_END_RIGHT_NOT_BOUNDING;
		return rootoffset;
	}
#else
	gint bnodeoffset;

	bnodeoffset = db_find_bounding_tnode(db, rootoffset, key, result, NULL);
	if(*result != REALLY_BOUNDING_NODE)
	return bnodeoffset;

	/* There is at least one node with the key we're interested in,
	 * now make sure we have the rightmost */
	node = offsettoptr(db, bnodeoffset);
	while(WG_COMPARE(db, node->current_max, key) == WG_EQUAL) {
		gint nextoffset = TNODE_SUCCESSOR(db, node);
		if(nextoffset) {
			struct wg_tnode *next = offsettoptr(db, nextoffset);
			if(WG_COMPARE(db, next->current_min, key) == WG_GREATER)
			/* next->current_min > key */
			break; /* overshot */
			node = next;
		}
		else
		break; /* last node in chain */
	}
	return ptrtooffset(db, node);
#endif
}

__device__ gint wg_search_ttree_leftmost(void *db, gint rootoffset,  //1257
		gint key, gint *result, struct wg_tnode *lb_node) {

	struct wg_tnode * node;

#ifdef TTREE_SINGLE_COMPARE
	node = (struct wg_tnode *) offsettoptr(db, rootoffset);

	/* Rightmost bound search mirrored */
	if (WG_COMPARE(db, key, node->current_max) == WG_GREATER) {
		/* key > node->current_max */
		if (node->right_child_offset != 0) {
			return wg_search_ttree_leftmost(db, node->right_child_offset, key,
					result, lb_node);
		} else if (lb_node) {
			/* Dead end, but we still have an unexamined node left */
			if (WG_COMPARE(db, key, lb_node->current_min) != WG_LESSTHAN) {
				/* key>=lb_node->current_min */
				*result = REALLY_BOUNDING_NODE;
				return ptrtooffset(db, lb_node);
			}
		}
		*result = DEAD_END_RIGHT_NOT_BOUNDING;
		return rootoffset;
	} else {
		if (node->left_child_offset != 0) {
			return wg_search_ttree_leftmost(db, node->left_child_offset, key,
					result, node);
		} else if (WG_COMPARE(db, key, node->current_min) != WG_LESSTHAN) {
			/* key>=node->current_min */
			*result = REALLY_BOUNDING_NODE;
			return rootoffset;
		}
		*result = DEAD_END_LEFT_NOT_BOUNDING;
		return rootoffset;
	}
#else
	gint bnodeoffset;

	bnodeoffset = db_find_bounding_tnode(db, rootoffset, key, result, NULL);
	if(*result != REALLY_BOUNDING_NODE)
	return bnodeoffset;

	/* One (we don't know which) bounding node found, traverse the
	 * tree to the leftmost. */
	node = offsettoptr(db, bnodeoffset);
	while(WG_COMPARE(db, node->current_min, key) == WG_EQUAL) {
		gint prevoffset = TNODE_PREDECESSOR(db, node);
		if(prevoffset) {
			struct wg_tnode *prev = offsettoptr(db, prevoffset);
			if(WG_COMPARE(db, prev->current_max, key) == WG_LESSTHAN)
			/* prev->current_max < key */
			break; /* overshot */
			node = prev;
		}
		else
		break; /* first node in chain */
	}
	return ptrtooffset(db, node);
#endif
}

/** Find first occurrence of a value in a T-tree node
 *  returns the number of the slot. If the value itself
 *  is missing, the location of the first value that
 *  exceeds it is returned.
 */
__device__ gint wg_search_tnode_first(void *db, gint nodeoffset, gint key, // 1325
		gint column) {

	gint i, encoded;
	struct wg_tnode *node = (struct wg_tnode *) offsettoptr(db, nodeoffset);

	for (i = 0; i < node->number_of_elements; i++) {
		/* Naive scan is ok for small values of WG_TNODE_ARRAY_SIZE. */
		encoded = wg_get_field(db,
				(void *) offsettoptr(db, node->array_of_values[i]), column);
		if (WG_COMPARE(db, encoded, key) != WG_LESSTHAN)
			/* encoded >= key */
			return i;
	}

	return -1;
}

__device__ gint wg_search_tnode_last(void *db, gint nodeoffset, gint key, // 1348
		gint column) {

	gint i, encoded;
	struct wg_tnode *node = (struct wg_tnode *) offsettoptr(db, nodeoffset);

	for (i = node->number_of_elements - 1; i >= 0; i--) {
		encoded = wg_get_field(db,
				(void *) offsettoptr(db, node->array_of_values[i]), column);
		if (WG_COMPARE(db, encoded, key) != WG_GREATER)
			/* encoded <= key */
			return i;
	}

	return -1;
}

__device__ gint wg_index_del_field(void *db, void *rec, gint column) {
	gint *ilist;
	gcell *ilistelem;
	db_memsegment_header* dbh = dbmemsegh(db);
	gint reclen = wg_get_record_len(db, rec);

#ifdef CHECK
	/* XXX: if used from wg_set_field() only, this is redundant */
	if (column > MAX_INDEXED_FIELDNR || column >= reclen)
		return -1;
	if (is_special_record(rec))
		return -1;
#endif

#if 0
	/* XXX: if used from wg_set_field() only, this is redundant */
	if(!dbh->index_control_area_header.index_table[column])
	return -1;
#endif

	/* Find all indexes on the column */
	ilist = &dbh->index_control_area_header.index_table[column];
	while (*ilist) {
		ilistelem = (gcell *) offsettoptr(db, *ilist);
		if (ilistelem->car) {
			wg_index_header *hdr = (wg_index_header *) offsettoptr(db,
					ilistelem->car);

			if (reclen > hdr->rec_field_index[hdr->fields - 1]) {
				if (MATCH_TEMPLATE(db, hdr, rec)) {
					INDEX_REMOVE_ROW(db, hdr, ilistelem->car, rec)
				}
			}
		}
		ilist = &ilistelem->cdr;
	}

#ifdef USE_INDEX_TEMPLATE
	/* Find all indexes on the column */
	ilist = &dbh->index_control_area_header.index_template_table[column];
	while (*ilist) {
		ilistelem = (gcell *) offsettoptr(db, *ilist);
		if (ilistelem->car) {
			wg_index_header *hdr = (wg_index_header *) offsettoptr(db,
					ilistelem->car);

			if (reclen > hdr->rec_field_index[hdr->fields - 1]) {
				if (MATCH_TEMPLATE(db, hdr, rec)) {
					INDEX_REMOVE_ROW(db, hdr, ilistelem->car, rec)
				}
			}
		}
		ilist = &ilistelem->cdr;
	}
#endif

	return 0;
}

__device__
   static gint find_index_template(void *db, gint *matchrec,
		gint reclen) {  // 1858
	gint *ilist;
	void *rec;
	db_memsegment_header* dbh = dbmemsegh(db);
	wg_index_template *tmpl;
	gint fixed_columns = 0, last_fixed = 0;
	int i;

	/* Get some statistics about the match record and validate it */
	for (i = 0; i < reclen; i++) {
		gint type = wg_get_encoded_type(db, matchrec[i]);
		if (type == WG_RECORDTYPE) {
			show_index_error(db, "record links not allowed in index templates");
			return 0;
		}
		if (type != WG_VARTYPE) {
			fixed_columns++;
			last_fixed = i;
		}
	}
	if (!fixed_columns) {
		show_index_error(db, "not a legal match record");
		return 0;
	}
	reclen = last_fixed + 1;

	/* Find a matching template. */
	ilist = &dbh->index_control_area_header.index_template_list;
	while (*ilist) {
		gcell *ilistelem = (gcell *) offsettoptr(db, *ilist);
		if (!ilistelem->car) {
			show_index_error(db, "Invalid header in index tempate list");
			return 0;
		}
		tmpl = (wg_index_template *) offsettoptr(db, ilistelem->car);
		if (tmpl->fixed_columns == fixed_columns) {
			rec = offsettoptr(db, tmpl->offset_matchrec);
			if (reclen != wg_get_record_len(db, rec))
				goto nextelem;
			/* match not possible */
			for (i = 0; i < reclen; i++) {
				if (wg_get_encoded_type(db, matchrec[i]) != WG_VARTYPE) {
					if (WG_COMPARE(db,
							matchrec[i], wg_get_field(db, rec, i)) != WG_EQUAL)
						goto nextelem;
				}
			}
			/* We have a match. */
			return ilistelem->car;
		} else if (tmpl->fixed_columns < fixed_columns) {
			/* No matching record found. New template should be inserted
			 * ahead of current element. */
			break;
		}
		nextelem: ilist = &ilistelem->cdr;
	}

	return 0;
}

__device__
   static gint sort_columns(gint *sorted_cols, gint *columns,  // 2001
		gint col_count) {
	gint i = 0;
	gint prev = -1;
	while (i < col_count) {
		gint lowest = MAX_INDEXED_FIELDNR + 1;
		gint j;
		for (j = 0; j < col_count; j++) {
			if (columns[j] < lowest && columns[j] > prev)
				lowest = columns[j];
		}
		if (lowest == MAX_INDEXED_FIELDNR + 1)
			break;
		sorted_cols[i++] = lowest;
		prev = lowest;
	};
	return i;
}

__device__ gint wg_multi_column_to_index_id(void *db, gint *columns,
		gint col_count, // 2396
		gint type, gint *matchrec, gint reclen) {
	int i;
	gint template_offset = 0;
	db_memsegment_header* dbh = dbmemsegh(db);
	gint *ilist;
	gcell *ilistelem;
	gint sorted_cols[MAX_INDEX_FIELDS];

#ifdef USE_INDEX_TEMPLATE
	/* Validate the match record and find the template */
	if (matchrec) {
		if (!reclen) {
			show_index_error(db, "Zero-length match record not allowed");
			return -1;
		}

		if (reclen > MAX_INDEXED_FIELDNR + 1) {
			show_index_error_nr(db, "Match record too long, max",
			MAX_INDEXED_FIELDNR + 1);
			return -1;
		}

		template_offset = find_index_template(db, matchrec, reclen);
		if (!template_offset) {
			/* No matching template */
			return -1;
		}
	}
#endif

	/* Column count validation */
	if (col_count < 1) {
		show_index_error(db, "need at least one indexed column");
		return -1;
	} else if (col_count > MAX_INDEX_FIELDS) {
		show_index_error_nr(db, "Max allowed indexed fields",
		MAX_INDEX_FIELDS);
		return -1;
	}

	if (col_count > 1) {
		if (sort_columns(sorted_cols, columns, col_count) < col_count) {
			show_index_error(db, "Duplicate columns not allowed");
			return -1;
		}
	} else {
		sorted_cols[0] = columns[0];
	}

	for (i = 0; i < col_count; i++) {
		if (sorted_cols[i] > MAX_INDEXED_FIELDNR) {
			show_index_error_nr(db, "Max allowed column number",
			MAX_INDEXED_FIELDNR);
			return -1;
		}
	}

	/* Find all indexes on the first column */
	ilist = &dbh->index_control_area_header.index_table[sorted_cols[0]];
	while (*ilist) {
		ilistelem = (gcell *) offsettoptr(db, *ilist);
		if (ilistelem->car) {
			wg_index_header *hdr = (wg_index_header *) offsettoptr(db,
					ilistelem->car);
#ifndef USE_INDEX_TEMPLATE
			if(!type || type==hdr->type) {
#else
			if ((!type || type == hdr->type)
					&& hdr->template_offset == template_offset) {
#endif
				if (hdr->fields == col_count) {
					for (i = 0; i < col_count; i++) {
						if (hdr->rec_field_index[i] != sorted_cols[i])
							goto nextindex;
					}
					return ilistelem->car; /* index id */
				}
			}
		}
		nextindex: ilist = &ilistelem->cdr;
	}

	return -1;
}

__device__ gint wg_index_add_field(void *db, void *rec, gint column) {
	gint *ilist;
	gcell *ilistelem;
	db_memsegment_header* dbh = dbmemsegh(db);
	gint reclen = wg_get_record_len(db, rec);

#ifdef CHECK
	/* XXX: if used from wg_set_field() only, this is redundant */
	if (column > MAX_INDEXED_FIELDNR || column >= reclen)
		return -1;
	if (is_special_record(rec))
		return -1;
#endif

#if 0
	/* XXX: if used from wg_set_field() only, this is redundant */
	if(!dbh->index_control_area_header.index_table[column])
	return -1;
#endif

	ilist = &dbh->index_control_area_header.index_table[column];
	while (*ilist) {
		ilistelem = (gcell *) offsettoptr(db, *ilist);
		if (ilistelem->car) {
			wg_index_header *hdr = (wg_index_header *) offsettoptr(db,
					ilistelem->car);
			if (reclen > hdr->rec_field_index[hdr->fields - 1]) {
				if (MATCH_TEMPLATE(db, hdr, rec)) {
					INDEX_ADD_ROW(db, hdr, ilistelem->car, rec)
				}
			}
		}
		ilist = &ilistelem->cdr;
	}

#ifdef USE_INDEX_TEMPLATE
	/* Other candidates are indexes that have match
	 * records. The current record may have become compatible
	 * with their template.
	 */
	ilist = &dbh->index_control_area_header.index_template_table[column];
	while (*ilist) {
		ilistelem = (gcell *) offsettoptr(db, *ilist);
		if (ilistelem->car) {
			wg_index_header *hdr = (wg_index_header *) offsettoptr(db,
					ilistelem->car);
			if (reclen > hdr->rec_field_index[hdr->fields - 1]) {
				if (MATCH_TEMPLATE(db, hdr, rec)) {
					INDEX_ADD_ROW(db, hdr, ilistelem->car, rec)
				}
			}
		}
		ilist = &ilistelem->cdr;
	}
#endif

	return 0;
}

__device__
gint wg_match_template(void *db, wg_index_template *tmpl, void *rec) {
  void *matchrec;
  gint reclen, mreclen;
  int i;

#ifdef CHECK
  /* Paranoia */
  if(!tmpl->offset_matchrec) {
    show_index_error(db, "Invalid match record template");
    return 0;
  }
#endif

  matchrec = offsettoptr(db, tmpl->offset_matchrec);
  mreclen = wg_get_record_len(db, matchrec);
  reclen = wg_get_record_len(db, rec);
  if(mreclen > reclen) {
    /* Match records always end in a fixed column, so
     * this is guaranteed to be a mismatch
     */
    return 0;
  }
  else if(mreclen < reclen) {
    /* Fields outside the template always match */
    reclen = mreclen;
  }
  for(i=0; i<reclen; i++) {
    gint enc = wg_get_field(db, matchrec, i);
    if(wg_get_encoded_type(db, enc) != WG_VARTYPE) {
      if(WG_COMPARE(db, enc, wg_get_field(db, rec, i)) != WG_EQUAL)
        return 0;
    }
  }
  return 1;
}

__device__ static int db_which_branch_causes_overweight(void *db, struct wg_tnode *root){
  struct wg_tnode *child;
  if(root->left_subtree_height > root->right_subtree_height){
    child = (struct wg_tnode *)offsettoptr(db,root->left_child_offset);
    if(child->left_subtree_height >= child->right_subtree_height)return LL_CASE;
    else return LR_CASE;
  }else{
    child = (struct wg_tnode *)offsettoptr(db,root->right_child_offset);
    if(child->left_subtree_height > child->right_subtree_height)return RL_CASE;
    else return RR_CASE;
  }
}

__device__ static int db_rotate_ttree(void *db, gint index_id, struct wg_tnode *root, int overw){
  gint grandparent = root->parent_offset;
  gint initialrootoffset = ptrtooffset(db,root);
  struct wg_tnode *r = NULL;
  struct wg_tnode *g = (struct wg_tnode *)offsettoptr(db,grandparent);
  wg_index_header *hdr = (wg_index_header *)offsettoptr(db,index_id);
  gint column = hdr->rec_field_index[0]; /* always one column for T-tree */

  if(overw == LL_CASE){

/*                       A                          B
*                     B     C                    D     A
*                   D  E             ->        N     E  C
*                  N
*/
    //printf("LL_CASE\n");
    //save some stuff into variables for later use
    gint offset_left_child = root->left_child_offset;
    gint offset_right_grandchild = ((struct wg_tnode *)offsettoptr(db,offset_left_child))->right_child_offset;
    gint right_grandchild_height = ((struct wg_tnode *)offsettoptr(db,offset_left_child))->right_subtree_height;


    //first switch: E goes to A's left child
    root->left_child_offset = offset_right_grandchild;
    root->left_subtree_height = right_grandchild_height;
    if(offset_right_grandchild != 0){
      ((struct wg_tnode *)offsettoptr(db,offset_right_grandchild))->parent_offset = ptrtooffset(db,root);
    }
    //second switch: A goes to B's right child
    ((struct wg_tnode *)offsettoptr(db,offset_left_child)) -> right_child_offset = ptrtooffset(db,root);
    ((struct wg_tnode *)offsettoptr(db,offset_left_child)) -> right_subtree_height = max(root->left_subtree_height,root->right_subtree_height)+1;
    root->parent_offset = offset_left_child;
    //for later grandparent fix
    r = (struct wg_tnode *)offsettoptr(db,offset_left_child);

  }else if(overw == RR_CASE){

/*                       A                          C
*                     B     C                    A     E
*                         D   E         ->     B  D      N
*                              N
*/
    //printf("RR_CASE\n");
    //save some stuff into variables for later use
    gint offset_right_child = root->right_child_offset;
    gint offset_left_grandchild = ((struct wg_tnode *)offsettoptr(db,offset_right_child))->left_child_offset;
    gint left_grandchild_height = ((struct wg_tnode *)offsettoptr(db,offset_right_child))->left_subtree_height;
    //first switch: D goes to A's right child
    root->right_child_offset = offset_left_grandchild;
    root->right_subtree_height = left_grandchild_height;
    if(offset_left_grandchild != 0){
      ((struct wg_tnode *)offsettoptr(db,offset_left_grandchild))->parent_offset = ptrtooffset(db,root);
    }
    //second switch: A goes to C's left child
    ((struct wg_tnode *)offsettoptr(db,offset_right_child)) -> left_child_offset = ptrtooffset(db,root);
    ((struct wg_tnode *)offsettoptr(db,offset_right_child)) -> left_subtree_height = max(root->right_subtree_height,root->left_subtree_height)+1;
    root->parent_offset = offset_right_child;
    //for later grandparent fix
    r = (struct wg_tnode *)offsettoptr(db,offset_right_child);

  }else if(overw == LR_CASE){
/*               A                    E
*             B     C             B       A
*          D    E        ->     D  F    G    C
*             F  G                 N
*             N
*/
    struct wg_tnode *bb, *ee;
    //save some stuff into variables for later use
    gint offset_left_child = root->left_child_offset;
    gint offset_right_grandchild = ((struct wg_tnode *)offsettoptr(db,offset_left_child))->right_child_offset;

    //first swtich: G goes to A's left child
    ee = (struct wg_tnode *)offsettoptr(db,offset_right_grandchild);
    root -> left_child_offset = ee -> right_child_offset;
    root -> left_subtree_height = ee -> right_subtree_height;
    if(ee -> right_child_offset != 0){
      ((struct wg_tnode *)offsettoptr(db,ee->right_child_offset))->parent_offset = ptrtooffset(db, root);
    }
    //second switch: F goes to B's right child
    bb = (struct wg_tnode *)offsettoptr(db,offset_left_child);
    bb -> right_child_offset = ee -> left_child_offset;
    bb -> right_subtree_height = ee -> left_subtree_height;
    if(ee -> left_child_offset != 0){
      ((struct wg_tnode *)offsettoptr(db,ee->left_child_offset))->parent_offset = offset_left_child;
    }
    //third switch: B goes to E's left child
    /* The Lehman/Carey "special" LR rotation - instead of creating
     * an internal node with one element, the values of what will become the
     * left child will be moved over to the parent, thus ensuring the internal
     * node is adequately filled. This is only allowed if E is a leaf.
     */
    if(ee->number_of_elements == 1 && !ee->right_child_offset &&\
      !ee->left_child_offset && bb->number_of_elements == WG_TNODE_ARRAY_SIZE){
      int i;

      /* Create space for elements from B */
      ee->array_of_values[bb->number_of_elements - 1] = ee->array_of_values[0];

      /* All the values moved are smaller than in E */
      for(i=1; i<bb->number_of_elements; i++)
        ee->array_of_values[i-1] = bb->array_of_values[i];
      ee->number_of_elements = bb->number_of_elements;

      /* Examine the new leftmost element to find current_min */
      ee->current_min = wg_get_field(db, (void *)offsettoptr(db,
        ee->array_of_values[0]), column);

      bb -> number_of_elements = 1;
      bb -> current_max = bb -> current_min;
    }

    //then switch the nodes
    ee -> left_child_offset = offset_left_child;
    ee -> left_subtree_height = max(bb->right_subtree_height,bb->left_subtree_height)+1;
    bb -> parent_offset = offset_right_grandchild;
    //fourth switch: A goes to E's right child
    ee -> right_child_offset = ptrtooffset(db, root);
    ee -> right_subtree_height = max(root->right_subtree_height,root->left_subtree_height)+1;
    root -> parent_offset = offset_right_grandchild;
    //for later grandparent fix
    r = ee;

  }else if(overw == RL_CASE){

/*               A                    E
*             C     B             A       B
*                 E   D  ->     C  G    F   D
*               G  F                    N
*                  N
*/
    struct wg_tnode *bb, *ee;
    //save some stuff into variables for later use
    gint offset_right_child = root->right_child_offset;
    gint offset_left_grandchild = ((struct wg_tnode *)offsettoptr(db,offset_right_child))->left_child_offset;

    //first swtich: G goes to A's left child
    ee = (struct wg_tnode *)offsettoptr(db,offset_left_grandchild);
    root -> right_child_offset = ee -> left_child_offset;
    root -> right_subtree_height = ee -> left_subtree_height;
    if(ee -> left_child_offset != 0){
      ((struct wg_tnode *)offsettoptr(db,ee->left_child_offset))->parent_offset = ptrtooffset(db, root);
    }

    //second switch: F goes to B's right child
    bb = (struct wg_tnode *)offsettoptr(db,offset_right_child);
    bb -> left_child_offset = ee -> right_child_offset;
    bb -> left_subtree_height = ee -> right_subtree_height;
    if(ee -> right_child_offset != 0){
      ((struct wg_tnode *)offsettoptr(db,ee->right_child_offset))->parent_offset = offset_right_child;
    }

    //third switch: B goes to E's right child
    /* "special" RL rotation - see comments for LR_CASE */
    if(ee->number_of_elements == 1 && !ee->right_child_offset &&\
      !ee->left_child_offset &&  bb->number_of_elements == WG_TNODE_ARRAY_SIZE){
      int i;

      /* All the values moved are larger than in E */
      for(i=1; i<bb->number_of_elements; i++)
        ee->array_of_values[i] = bb->array_of_values[i-1];
      ee->number_of_elements = bb->number_of_elements;

      /* Examine the new rightmost element to find current_max */
      ee->current_max = wg_get_field(db, (void *)offsettoptr(db,
        ee->array_of_values[ee->number_of_elements - 1]), column);

      /* Remaining B node array element should sit in slot 0 */
      bb->array_of_values[0] = \
        bb->array_of_values[bb->number_of_elements - 1];
      bb -> number_of_elements = 1;
      bb -> current_min = bb -> current_max;
    }

    ee -> right_child_offset = offset_right_child;
    ee -> right_subtree_height = max(bb->right_subtree_height,bb->left_subtree_height)+1;
    bb -> parent_offset = offset_left_grandchild;
    //fourth switch: A goes to E's right child

    ee -> left_child_offset = ptrtooffset(db, root);
    ee -> left_subtree_height = max(root->right_subtree_height,root->left_subtree_height)+1;
    root -> parent_offset = offset_left_grandchild;
    //for later grandparent fix
    r = ee;

  } else {
    /* catch an error case (can't really happen) */
    show_index_error(db, "tree rotate called with invalid argument, "\
      "index may have become corrupt");
    return -1;
  }

  //fix grandparent - regardless of current 'overweight' case

  if(grandparent == 0){//'grandparent' is index header data
    r->parent_offset = 0;
    //TODO more error check here
    TTREE_ROOT_NODE(hdr) = ptrtooffset(db,r);
  }else{//grandparent is usual node
    //printf("change grandparent node\n");
    r -> parent_offset = grandparent;
    if(g->left_child_offset == initialrootoffset){//new subtree must replace the left child of grandparent
      g->left_child_offset = ptrtooffset(db,r);
      g->left_subtree_height = max(r->left_subtree_height,r->right_subtree_height)+1;
    }else{
      g->right_child_offset = ptrtooffset(db,r);
      g->right_subtree_height = max(r->left_subtree_height,r->right_subtree_height)+1;
    }
  }

  return 0;
}

__device__ static gint ttree_remove_row(void *db, gint index_id, void * rec) {
  int i, found;
  gint key, rootoffset, column, boundtype, bnodeoffset;
  gint rowoffset;
  struct wg_tnode *node, *parent;
  wg_index_header *hdr = (wg_index_header *)offsettoptr(db,index_id);

  rootoffset = TTREE_ROOT_NODE(hdr);
#ifdef CHECK
  if(rootoffset == 0){
#ifdef WG_NO_ERRPRINT
#else
    //printf("index at offset %d does not exist\n", (int) index_id);
#endif
    return -1;
  }
#endif
  column = hdr->rec_field_index[0]; /* always one column for T-tree */
  key = wg_get_field(db, rec, column);
  rowoffset = ptrtooffset(db, rec);

  /* find bounding node for the value. Since non-unique values
   * are allowed, we will find the leftmost node and scan
   * right from there (we *need* the exact row offset).
   */

  bnodeoffset = wg_search_ttree_leftmost(db,
          rootoffset, key, &boundtype, NULL);
  node = (struct wg_tnode *)offsettoptr(db,bnodeoffset);

  //if bounding node does not exist - error
  if(boundtype != REALLY_BOUNDING_NODE) return -2;

  /* find the record inside the node. This is an expensive loop if there
   * are many repeated values, so unnecessary deleting should be avoided
   * on higher level.
   */
  found = -1;
  for(;;) {
    for(i=0;i<node->number_of_elements;i++){
      if(node->array_of_values[i] == rowoffset) {
        found = i;
        goto found_row;
      }
    }
    bnodeoffset = TNODE_SUCCESSOR(db, node);
    if(!bnodeoffset)
      break; /* no more successors */
    node = (struct wg_tnode *)offsettoptr(db,bnodeoffset);
    if(WG_COMPARE(db, node->current_min, key) == WG_GREATER)
      break; /* successor is not a bounding node */
  }

found_row:
  if(found == -1) return -3;

  //delete the key and rearrange other elements
  node->number_of_elements--;
  if(found < node->number_of_elements) { /* not the last element */
    /* slide the elements to the right of the found value
     * one step to the left */
    for(i=found; i<node->number_of_elements; i++)
      node->array_of_values[i] = node->array_of_values[i+1];
  }

  /* Update min/max */
  if(found==node->number_of_elements && node->number_of_elements != 0) {
    /* Rightmost element was removed, so new max should be updated to
     * the new rightmost value */
    node->current_max = wg_get_field(db, (void *)offsettoptr(db,
      node->array_of_values[node->number_of_elements - 1]), column);
  } else if(found==0 && node->number_of_elements != 0) {
    /* current_min removed, update to new leftmost value */
    node->current_min = wg_get_field(db, (void *)offsettoptr(db,
      node->array_of_values[0]), column);
  }

  //check underflow and take some actions if needed
  if(node->number_of_elements < 5){//TODO use macro
    //if the node is internal node - borrow its gratest lower bound from the node where it is
    if(node->left_child_offset != 0 && node->right_child_offset != 0){//internal node
#ifndef TTREE_CHAINED_NODES
      gint greatestlb = wg_ttree_find_glb_node(db,node->left_child_offset);
#else
      gint greatestlb = node->pred_offset;
#endif
      struct wg_tnode *glbnode = (struct wg_tnode *)offsettoptr(db, greatestlb);

      /* Make space for a new min value */
      for(i=node->number_of_elements; i>0; i--)
        node->array_of_values[i] = node->array_of_values[i-1];

      /* take the glb value (always the rightmost in the array) and
       * insert it in our node */
      node -> array_of_values[0] = \
        glbnode->array_of_values[glbnode->number_of_elements-1];
      node -> number_of_elements++;
      node -> current_min = glbnode -> current_max;
      if(node->number_of_elements == 1) /* we just got our first element */
        node->current_max = glbnode -> current_max;
      glbnode -> number_of_elements--;

      //reset new max for glbnode
      if(glbnode->number_of_elements != 0) {
        glbnode->current_max = wg_get_field(db, (void *)offsettoptr(db,
          glbnode->array_of_values[glbnode->number_of_elements - 1]), column);
      }

      node = glbnode;
    }
  }

  //now variable node points to the node which really lost an element
  //this is definitely leaf or half-leaf
  //if the node is empty - free it and rebalanc the tree
  parent = NULL;
  //delete the empty leaf
  if(node->left_child_offset == 0 && node->right_child_offset == 0 && node->number_of_elements == 0){
    if(node->parent_offset != 0){
      parent = (struct wg_tnode *)offsettoptr(db, node->parent_offset);
      //was it left or right child
      if(parent->left_child_offset == ptrtooffset(db,node)){
        parent->left_child_offset=0;
        parent->left_subtree_height=0;
      }else{
        parent->right_child_offset=0;
        parent->right_subtree_height=0;
      }
    }
#ifdef TTREE_CHAINED_NODES
    /* Remove the node from sequential chain */
    if(node->succ_offset) {
      struct wg_tnode *succ = \
        (struct wg_tnode *) offsettoptr(db, node->succ_offset);
      succ->pred_offset = node->pred_offset;
    } else {
      TTREE_MAX_NODE(hdr) = node->pred_offset;
    }
    if(node->pred_offset) {
      struct wg_tnode *pred = \
        (struct wg_tnode *) offsettoptr(db, node->pred_offset);
      pred->succ_offset = node->succ_offset;
    } else {
      TTREE_MIN_NODE(hdr) = node->succ_offset;
    }
#endif
    /* Free the node, unless it's the root node */
    if(node != offsettoptr(db, TTREE_ROOT_NODE(hdr))) {
      wg_free_tnode(db, ptrtooffset(db,node));
    } else {
      /* Set empty state of root node */
      node->current_max = WG_ILLEGAL;
      node->current_min = WG_ILLEGAL;
#ifdef TTREE_CHAINED_NODES
      TTREE_MAX_NODE(hdr) = TTREE_ROOT_NODE(hdr);
      TTREE_MIN_NODE(hdr) = TTREE_ROOT_NODE(hdr);
#endif
    }
    //rebalance if needed
  }

  //or if the node was a half-leaf, see if it can be merged with its leaf
  if((node->left_child_offset == 0 && node->right_child_offset != 0) || (node->left_child_offset != 0 && node->right_child_offset == 0)){
    int elements = node->number_of_elements;
    int left;
    struct wg_tnode *child;
    if(node->left_child_offset != 0){
      child = (struct wg_tnode *)offsettoptr(db, node->left_child_offset);
      left = 1;//true
    }else{
      child = (struct wg_tnode *)offsettoptr(db, node->right_child_offset);
      left = 0;//false
    }
    elements += child->number_of_elements;
    if(!(child->left_subtree_height == 0 && child->right_subtree_height == 0)){
      show_index_error(db,
        "index tree is not balanced, deleting algorithm doesn't work");
      return -4;
    }
    //if possible move all elements from child to node and free child
    if(elements <= WG_TNODE_ARRAY_SIZE){
      int i = node->number_of_elements;
      int j;
      node->number_of_elements = elements;
      if(left){
        /* Left child elements are all smaller than in current node */
        for(j=i-1; j>=0; j--){
          node->array_of_values[j + child->number_of_elements] = \
            node->array_of_values[j];
        }
        for(j=0;j<child->number_of_elements;j++){
          node->array_of_values[j]=child->array_of_values[j];
        }
        node->left_subtree_height=0;
        node->left_child_offset=0;
        node->current_min=child->current_min;
        if(!i) node->current_max=child->current_max; /* parent was empty */
      }else{
        /* Right child elements are all larger than in current node */
        for(j=0;j<child->number_of_elements;j++){
          node->array_of_values[i+j]=child->array_of_values[j];
        }
        node->right_subtree_height=0;
        node->right_child_offset=0;
        node->current_max=child->current_max;
        if(!i) node->current_min=child->current_min; /* parent was empty */
      }
#ifdef TTREE_CHAINED_NODES
      /* Remove the child from sequential chain */
      if(child->succ_offset) {
        struct wg_tnode *succ = \
          (struct wg_tnode *) offsettoptr(db, child->succ_offset);
        succ->pred_offset = child->pred_offset;
      } else {
        TTREE_MAX_NODE(hdr) = child->pred_offset;
      }
      if(child->pred_offset) {
        struct wg_tnode *pred = \
          (struct wg_tnode *) offsettoptr(db, child->pred_offset);
        pred->succ_offset = child->succ_offset;
      } else {
        TTREE_MIN_NODE(hdr) = child->succ_offset;
      }
#endif
      wg_free_tnode(db, ptrtooffset(db, child));
      if(node->parent_offset) {
        parent = (struct wg_tnode *)offsettoptr(db, node->parent_offset);
        if(parent->left_child_offset==ptrtooffset(db,node)){
          parent->left_subtree_height=1;
        }else{
          parent->right_subtree_height=1;
        }
      }
    }
  }

  //check balance and update subtree height data
  //stop when find a node where subtree heights dont change
  if(parent != NULL){
    int balance, height;
    for(;;) {
      balance = parent->left_subtree_height - parent->right_subtree_height;
      if(balance > 1 || balance < -1){//must rebalance
        //the current parent is root for balancing operation
        //rotarion fixes subtree heights in grandparent
        //determine the branch that causes overweight
        int overw = db_which_branch_causes_overweight(db,parent);
        //fix balance
        db_rotate_ttree(db,index_id,parent,overw);
      }
      else if(parent->parent_offset) {
        struct wg_tnode *gp;
        //manually set grandparent subtree heights
        height = max(parent->left_subtree_height,parent->right_subtree_height);
        gp = (struct wg_tnode *)offsettoptr(db, parent->parent_offset);
        if(gp->left_child_offset==ptrtooffset(db,parent)){
          gp->left_subtree_height=1+height;
        }else{
          gp->right_subtree_height=1+height;
        }
      }
      if(!parent->parent_offset)
        break; /* root node reached */
      parent = (struct wg_tnode *)offsettoptr(db, parent->parent_offset);
    }
  }
  return 0;
}

__device__
     static gint show_index_error(void* db, char* errmsg) {   // 2992
#ifdef WG_NO_ERRPRINT
#else
	// fprintf(stderr,"index error: %s\n",errmsg);
	printf("\nindex error: %s\n", errmsg);
#endif
	return -1;
}

__device__ static gint hash_remove_row(void *db, gint index_id, void *rec) {
  wg_index_header *hdr = (wg_index_header *)offsettoptr(db,index_id);
  gint i;
  gint values[MAX_INDEX_FIELDS];

  for(i=0; i<hdr->fields; i++) {
    values[i] = wg_get_field(db, rec, hdr->rec_field_index[i]);
  }
  return hash_recurse(db, hdr, NULL, 0, values, hdr->fields, rec,
    HASHIDX_OP_REMOVE, (hdr->type == WG_INDEX_TYPE_HASH_JSON));
}

__device__
     static gint show_index_error_nr(void* db, char* errmsg, gint nr) { // 3006
#ifdef WG_NO_ERRPRINT
#else
	// fprintf(stderr,"index error: %s %d\n", errmsg, (int) nr);
	printf("\nindex error: %s %d\n", errmsg, (int) nr);
#endif
	return -1;
}
