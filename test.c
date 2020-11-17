#include <stdlib.h>
#include <stdio.h>
#include "alexnet.h"

Alexnet alex;
Feature feat;

int main(void)
{
    zero_feats(&feat);
    for(int p=0; p<BATCH_SIZE; p++)
    {
        for(int in_c=0; in_c<IN_CHANNELS; in_c)
        {
            for(int i=0; i<FEATURE0_L; i++)
            {
                for(int j=0; j<FEATURE0_L; j++)
                {
                    feat.input[p][in_c][i][j] = 0.5;
                }            
            }
        }
        printf("batchID %d \n", p);
    }
    

    return 0;
}