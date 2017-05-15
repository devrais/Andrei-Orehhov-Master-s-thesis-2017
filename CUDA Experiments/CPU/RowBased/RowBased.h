/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   RowBased.h
 * Author: devrais
 *
 * Created on April 21, 2017, 10:59 PM
 */

#ifndef ROWBASED_H
#define ROWBASED_H

#define R 10000000 // number of rows in table
#define C 5 // number of columns in table

// Define number range in table
#define StartValue 1
#define EndValue 10

/*Define column for values addition
 *I choose 3rd column */
#define TargetField 2
#define SearchTarget 10 // We will search for number 10 in table

#define SubstituteValue 7 // We choose to substitute this value in our table


/* Every row will have its second field with data
 fillColumn will add numbers from 1 to 10
 */
void fillField(int (*tableRow)[C]);

/* Function will find specific column
 */
int scaleRows(int (*tableRow)[C]);

// Function will substitute chosen value by SearchTarget
void substituteFieldValue(int (*tableRow)[C]);


#endif /* ROWBASED_H */

