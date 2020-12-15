#include <stdlib.h>
#include <stdio.h>
#include "alexnet.h"

Alexnet alex;
int main(void)
{
    printf("sizeof(Alexnet) is %d KB\n", sizeof(Alexnet)>>10);
    global_params_initialize(&alex);
    //static Feature feat;
    train(&alex, 10);
    return 0;
}