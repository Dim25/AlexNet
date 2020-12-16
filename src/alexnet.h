#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "hyperparams.h"

#define IN_CHANNELS 3
#define C1_CHANNELS 96
#define C1_KERNEL_L 11
#define C2_CHANNELS 256
#define C2_KERNEL_L 5
#define C3_CHANNELS 384
#define C3_KERNEL_L 3
#define C4_CHANNELS 384
#define C4_KERNEL_L 3
#define C5_CHANNELS 256
#define C5_KERNEL_L 3
#define FC6KERNEL_L 6
#define FC6_LAYER   4096
#define FC7_LAYER   4096
#define OUT_LAYER   1000

#define C1_STRIDES 4
#define C2_STRIDES 1
#define C3_STRIDES 1
#define C4_STRIDES 1
#define C5_STRIDES 1

#define FEATURE0_L 227
#define FEATURE1_L 57
#define POOLING1_L 28
#define FEATURE2_L 28
#define POOLING2_L 13
#define FEATURE3_L 13
#define FEATURE4_L 13
#define FEATURE5_L 13
#define POOLING5_L 6


typedef struct {
    float *x_norm;
    float *avg;
    float *var;
}BN_params;


typedef struct{

    float C1_weights[C1_CHANNELS][IN_CHANNELS][C1_KERNEL_L][C1_KERNEL_L];
    float C2_weights[C2_CHANNELS][C1_CHANNELS][C2_KERNEL_L][C2_KERNEL_L];
    float C3_weights[C3_CHANNELS][C2_CHANNELS][C3_KERNEL_L][C3_KERNEL_L];
    float C4_weights[C4_CHANNELS][C3_CHANNELS][C4_KERNEL_L][C4_KERNEL_L];
    float C5_weights[C5_CHANNELS][C4_CHANNELS][C5_KERNEL_L][C5_KERNEL_L];
    float FC6weights[C5_CHANNELS][FC6_LAYER][FC6KERNEL_L][FC6KERNEL_L];
    float FC7weights[FC6_LAYER][FC7_LAYER];
    float FC8weights[FC7_LAYER][OUT_LAYER];
    
    float C1_bias[C1_CHANNELS];
    float C2_bias[C2_CHANNELS];
    float C3_bias[C3_CHANNELS];
    float C4_bias[C4_CHANNELS];
    float C5_bias[C5_CHANNELS];
    float FC6bias[FC6_LAYER];
    float FC7bias[FC7_LAYER];
    float FC8bias[OUT_LAYER];

    float BN1_gamma[C1_CHANNELS*FEATURE1_L*FEATURE1_L], BN1_b[C1_CHANNELS*FEATURE1_L*FEATURE1_L];
    float BN2_gamma[C2_CHANNELS*FEATURE2_L*FEATURE2_L], BN2_b[C2_CHANNELS*FEATURE2_L*FEATURE2_L];
    float BN3_gamma[C3_CHANNELS*FEATURE3_L*FEATURE3_L], BN3_b[C3_CHANNELS*FEATURE3_L*FEATURE3_L];
    float BN4_gamma[C4_CHANNELS*FEATURE4_L*FEATURE4_L], BN4_b[C4_CHANNELS*FEATURE4_L*FEATURE4_L];
    float BN5_gamma[C5_CHANNELS*FEATURE5_L*FEATURE5_L], BN5_b[C5_CHANNELS*FEATURE5_L*FEATURE5_L];

    BN_params bnp1, bnp2, bnp3, bnp4, bnp5;
            
}Alexnet;


typedef struct {

	float input[BATCH_SIZE][IN_CHANNELS][FEATURE0_L][FEATURE0_L];
	float C1[BATCH_SIZE][C1_CHANNELS][FEATURE1_L][FEATURE1_L];
	float BN1[BATCH_SIZE][C1_CHANNELS][FEATURE1_L][FEATURE1_L];
	float P1[BATCH_SIZE][C1_CHANNELS][POOLING1_L][POOLING1_L];

	float C2[BATCH_SIZE][C2_CHANNELS][FEATURE2_L][FEATURE2_L];
	float BN2[BATCH_SIZE][C2_CHANNELS][FEATURE2_L][FEATURE2_L];
	float P2[BATCH_SIZE][C2_CHANNELS][POOLING2_L][POOLING2_L];

	float C3[BATCH_SIZE][C3_CHANNELS][FEATURE3_L][FEATURE3_L];
	float BN3[BATCH_SIZE][C3_CHANNELS][FEATURE3_L][FEATURE3_L];

	float C4[BATCH_SIZE][C4_CHANNELS][FEATURE4_L][FEATURE4_L];
	float BN4[BATCH_SIZE][C4_CHANNELS][FEATURE4_L][FEATURE4_L];
    
	float C5[BATCH_SIZE][C5_CHANNELS][FEATURE5_L][FEATURE5_L];
	float BN5[BATCH_SIZE][C5_CHANNELS][FEATURE5_L][FEATURE5_L];
	float P5[BATCH_SIZE][C5_CHANNELS][POOLING5_L][POOLING5_L];

	float FC6[BATCH_SIZE][FC6_LAYER];

	float FC7[BATCH_SIZE][FC7_LAYER];

	float output[BATCH_SIZE][OUT_LAYER];

}Feature;


void zero_feats(Feature *feats);
void zero_grads(Alexnet *grads);

void global_params_initialize(Alexnet *net);

void net_forward(const Alexnet *alexnet, Feature *feats);
void net_backward(Feature *error, const Alexnet *alexnet, Alexnet *deltas, const Feature *feats, float lr);

void CatelogCrossEntropy(float *error, const float *preds, const float *labels, int units);
void CatelogCrossEntropy_backward(float *delta_preds, const float *preds, const float *labels, int units);




// Definiation of metric type
#define METRIC_ACCURACY  0
#define METRIC_PRECISION 1      // macro-precision
#define METRIC_RECALL    2      // macro-recall
#define METRIC_F1SCORE   3
#define METRIC_ROC       4


void metrics(float *ret, int *preds, int *labels, 
                int classes, int TotalNum, int type);

void predict(Alexnet *alexnet, float *inputs, float *outputs);

void train(Alexnet *alexnet, int epochs);
