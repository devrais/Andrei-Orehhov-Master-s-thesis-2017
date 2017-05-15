#include "RowBased.h"
/* Every row will have its second field with data
 fillColumn will add numbers from 1 to 10
 */
void fillField(int (*tableRow)[C]) {

    int i = 0;
    int current = StartValue;
    int *row;

    while (i < R) {
        row = (int *) tableRow;
        tableRow++;
        if (current > EndValue)
            current = StartValue;
        row[TargetField] = current;
        i++;
        current++;
    }
}

// Function will find specific column
int scaleRows(int (*tableRow)[C]) {

    int count = 0;
    int *row;

    for (int i = 0; i < R; i++) {
        row = (int *) tableRow;  
        if (row[TargetField] == SearchTarget)
            count += 1;
         tableRow++;
    }

    return count;
}

// Function will substitute chosen value by SearchTarget
void substituteFieldValue(int (*tableRow)[C]) {

    int i = 0;
    int *row;

    while (i < R) {
        row = (int *) tableRow;
        if (row[TargetField] == SubstituteValue)
            row[TargetField] = SearchTarget;
        tableRow++;
        i++;
    }
}
