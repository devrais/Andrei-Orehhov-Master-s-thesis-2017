/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ColumnBased.h
 * Author: devrais
 *
 * Created on April 21, 2017, 11:20 PM
 */

#ifndef COLUMNBASED_H
#define COLUMNBASED_H

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
void fillColumn(int (*tableRow)[C]);

/* Function will find specific column
 */
int scaleColumn(int (*tableRow)[C]);

// Function will substitute chosen value by SearchTarget
void substituteColumnValue(int (*tableRow)[C]);

#endif /* COLUMNBASED_H */

