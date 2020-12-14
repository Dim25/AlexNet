#include <stdlib.h>
#include <stdio.h>
#include "alexnet.h"

int main(void)
{
    static Alexnet alex;
    global_params_initialize(&alex);
    //static Feature feat;
    train(&alex, 10);
    return 0;
}