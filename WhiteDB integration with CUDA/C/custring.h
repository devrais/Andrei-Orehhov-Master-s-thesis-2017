/*
 * custring.h
 *
 *  Created on: Apr 16, 2017
 *      Author: devrais
 */

#ifndef CUSTRING_H_
#define CUSTRING_H_

__device__ size_t strlen(const char *str);
__device__ int strcmp(const char *s1, const char *s2);
__device__ int memcmp(const void *s1, const void *s2, size_t n);

#endif /* CUSTRING_H_ */
