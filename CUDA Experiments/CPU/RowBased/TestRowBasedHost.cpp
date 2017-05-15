/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: devrais
 *
 * Created on April 21, 2017, 5:45 PM
 */

#include <cstdlib>
#include <stdio.h>
#include <time.h>

#include "RowBased.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {

    clock_t start, finish, timeFillRow, timeFirstScale, timeSecondScale, timeRefactor;

    // We create row based table. It will have 10 million rows and 5 columns
    int (*tableRow)[C] = new int[R][C];

    // Start Measurements 

    // Fill selected rows with data
    start = clock();
    fillField(tableRow);
    finish = clock();
    timeFillRow = finish - start;

    //Scale Table 
    start = clock();
    printf("\nFound %d rows with target value %d ", scaleRows(tableRow), SearchTarget);
    finish = clock();
    timeFirstScale = finish - start;

    //Substitute value in row
    start = clock();
    substituteFieldValue(tableRow);
    finish = clock();
    timeRefactor = finish - start;

    //Scale Table second time
    start = clock();
    printf("\nFound %d rows with target value %d after substitute operation", scaleRows(tableRow), SearchTarget);
    finish = clock();
    timeSecondScale = finish - start;


    printf("\n");
    printf("\nIt took me %ld clicks (%f seconds) to fill selected rows.\n", timeFillRow, ((float) timeFillRow) / CLOCKS_PER_SEC);
    printf("\nIt took me %ld clicks (%f seconds) to scale selected rows.\n", timeFirstScale, ((float) timeFirstScale) / CLOCKS_PER_SEC);
    printf("\nIt took me %ld clicks (%f seconds) to re factor number %d with number %d.\n", timeRefactor, ((float) timeRefactor) / CLOCKS_PER_SEC, SubstituteValue, SearchTarget);
    printf("\nIt took me %ld clicks (%f seconds) to scale selected rows a second time.\n", timeSecondScale, ((float) timeSecondScale) / CLOCKS_PER_SEC);

    printf("\nTotal execution time = %f seconds",
            ((float) timeFillRow / CLOCKS_PER_SEC
            + (float) timeFirstScale / CLOCKS_PER_SEC
            + (float) timeRefactor / CLOCKS_PER_SEC
            + (float) timeSecondScale / CLOCKS_PER_SEC));
    return 0;
}

