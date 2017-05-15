#include "ColumnBased.h"
#include <stdio.h>

/* Every row will have its second field with data
 fillColumn will add numbers from 1 to 10
 */
void fillColumn(int (*tableColumn)[C]) {

    int i = 0;
    int current = StartValue;
    
    int *column = (int *) tableColumn;

    while (i < C) {
        if (current > EndValue) 
            current = StartValue;
        
        column[i] = current;
        i++;
        current++;
    }
}

// Function will find specific column

int scaleColumn(int (*tableColumn)[C]) {

    int i = 0;
    int count = 0;
    int *row;
    
    row = (int *) tableColumn;

    while ( i < C) {
        if (row[i] == SearchTarget)
            count += 1;
        i++;
    }

    return count;
}

// Function will substitute chosen value by SearchTarget

void substituteColumnValue(int (*tableColumn)[C]) {

    int i = 0;
    int *row;
    
    row = (int *) tableColumn;

    while (i < C) {
        if (row[i] == SubstituteValue)
            row[i] = SearchTarget;
        i++;
    }
}


