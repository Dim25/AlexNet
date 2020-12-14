#include <stdlib.h>
#include <stdio.h>
#include "alexnet.h"

Alexnet alex;
Feature feat;

int main(void)
{
    train(&alex, 10);
    return 0;
}