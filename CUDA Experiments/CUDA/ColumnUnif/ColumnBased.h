/*
 * ColumnBased.h
 *
 *  Created on: Apr 22, 2017
 *      Author: devrais
 */

#ifndef COLUMNBASED_H_
#define COLUMNBASED_H_

#define R 5 // number of rows in table
#define C 10000000 // number of columns in table

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
void fillColumn(int (*tableColumn)[C]);

/* Function will find specific column
 */
__global__ void scaleCudaColumn(int (*d_tableColumn)[C], int *d_count);

// Function will substitute chosen value by SearchTarget
__global__ void substituteCudaColumnValue(int (*d_column)[C]);

#endif /* COLUMNBASED_H_ */
