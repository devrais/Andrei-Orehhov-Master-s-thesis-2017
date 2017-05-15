/*
 * ColumnBased.h
 *
 *  Created on: Apr 22, 2017
 *      Author: devrais
 */

#ifndef COLUMNBASED_H_
#define COLUMNBASED_H_

#define R 10000000 // number of rows in table
#define C 5 // number of columns in table

// Define number range in table
#define StartValue 1
#define EndValue 10

/*Define column for values addition
 *I choose 3rd column in every row */
#define TargetField 2
/*Define row for values addition
 *I choose 3rd row in table */
#define TargetColumn 2
#define SearchTarget 10 // We will search for number 10 in table

#define SubstituteValue 7 // We choose to substitute this value in our table


/* Every row will have its second field with data
 fillColumn will add numbers from 1 to 10
 */
void fillField(int (*tableRow)[C]);

// Function will find specific field in rows
__global__ void scaleCudaRows(int (*d_tableRow)[C], int *d_count);

// Function will substitute chosen value by SearchTarget
__global__ void substituteFieldCudaValue(int (*d_tableRow)[C]);

// Will set incrementor to zero. It will be used "scaleCudaRow" and "substituteFieldCudaValue"
__global__ void resetIncrementor();

// Function will substitute chosen value by SearchTarget
void substituteColumnValue(int (*tableRow)[C]);

// Search function for CPU type programm
void scaleRow(int (*tableRow)[C], int column, int target, int *count);

#endif /* COLUMNBASED_H_ */
