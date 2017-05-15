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

#include "ColumnBased.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {

    clock_t start, finish, timeFillColumn, timeFirstScale, timeSecondScale, timeRefactor;

    // We create row based table. It will have 10 million rows and 5 columns
    int (*tableColumn)[C] = new int[R][C];
    
    tableColumn+=TargetColumn;
 //   int column = (int *) tableRow;

    // Start Measurements 

    // Fill selected column with data
    start = clock();
    fillColumn(tableColumn);
    finish = clock();
    timeFillColumn = finish - start;

    //Scale Table 
    start = clock();
    printf("\nFound %d values in column %d ", scaleColumn(tableColumn), SearchTarget);
    finish = clock();
    timeFirstScale = finish - start;

    //Substitute value in row
    start = clock();
    substituteColumnValue(tableColumn);
    finish = clock();
    timeRefactor = finish - start;

    //Scale Table second time
    start = clock();
    printf("\nFound %d values in column %d after substitute operation", scaleColumn(tableColumn), SearchTarget);
    finish = clock();
    timeSecondScale = finish - start;


    printf("\n");
    printf("\nIt took me %ld clicks (%f seconds) to fill selected column.\n", timeFillColumn, ((float) timeFillColumn) / CLOCKS_PER_SEC);
    printf("\nIt took me %ld clicks (%f seconds) to scale selected column.\n", timeFirstScale, ((float) timeFirstScale) / CLOCKS_PER_SEC);
    printf("\nIt took me %ld clicks (%f seconds) to refactor number %d with number %d.\n", timeRefactor, ((float) timeRefactor) / CLOCKS_PER_SEC, SubstituteValue, SearchTarget);
    printf("\nIt took me %ld clicks (%f seconds) to scale selected column a second time.\n", timeSecondScale, ((float) timeSecondScale) / CLOCKS_PER_SEC);

    printf("\nTotal execution time = %f seconds",
            ((float) timeFillColumn / CLOCKS_PER_SEC
            + (float) timeFirstScale / CLOCKS_PER_SEC
            + (float) timeRefactor / CLOCKS_PER_SEC
            + (float) timeSecondScale / CLOCKS_PER_SEC));
    return 0;
}


