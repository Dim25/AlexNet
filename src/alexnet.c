//
// File:        alexnet.c
// Description: Implemention of alexnet-related operations
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "alexnet.h"
#include "hyperparams.h"

//#define SHOW_OP_TIME


#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define EPSILON 0.0001


typedef struct{
    int batch_id;
    float *input; float *weights; float *bias; float *output;
    int in_channels; int out_channels; int kernel_size; int padding; int strides; int w; int h;
} conv_params;

void __ConvForward__(void *argv)
{
    /**
     *  input   data matrix
     *  w       weights of filter
     *  x, y    index of 'input'
     *  n       the width of filter
     **/

    conv_params cp;
    memcpy(&cp, (conv_params *)argv, sizeof(conv_params));
    float *input = cp.input;
    float *weights = cp.weights;
    float *bias = cp.bias;
    float *output = cp.output;
    int in_channels = cp.in_channels; int out_channels = cp.out_channels; 
    int kernel_size = cp.kernel_size; int padding=cp.padding; 
    int strides = cp.strides; int w = cp.w; int h = cp.h;
    int p = cp.batch_id;

    int out_w, out_h, cur_w, cur_h;
    out_w = (w+2*padding-kernel_size) / strides + 1;
    out_h = (h+2*padding-kernel_size) / strides + 1;
    
    for (int out_c = 0; out_c < out_channels; out_c++)
    {
        cur_w = 0; 
        for (int x = 0 - padding; x < w + padding; x += strides)
        {
            cur_h = 0;
            for (int y = 0 - padding; y < h + padding; y += strides)
            {
                //printf("cur_w is %d;  cur_h is %d\n", cur_w, cur_h);
            
                // output[out_c][cur_w][cur_h]
                
                for (int c = 0; c < in_channels; c++)
                {
                    /**
                     *  | -------------------------------------------------------------------|
                     *  | input[c][x][y]              input[c][x+kernel_size][y]             |
                     *  | input[c][x][y+kernel_size]  input[c][x+kernel_size][y+kernel_size] |
                     *  | -------------------------------------------------------------------|
                     * 
                     *          conv
                     * 
                     *   weights[out_c][c]
                     * 
                     *          ||
                     *          V
                     * 
                     *   output[c][cur_w][cur_h]
                     * */
                    float res = 0;
                    int x_shift = 0;
                    for(short x_shift=0; x_shift<kernel_size; x_shift++)
                    {
                        if(x+x_shift<0) // padding areas
                        {
                            continue;
                        }
                        if(x+x_shift>=w)
                        {
                            break;
                        }

                        for(short y_shift=0; y_shift<kernel_size; y_shift++)
                        {
                            if(y+y_shift<0) // padding areas
                            {
                                continue;
                            }
                            if(y+y_shift>=w)
                            {
                                break;
                            }
                    
                            res += input[(x+x_shift)*w+(y+y_shift)] * weights[x_shift*kernel_size+y_shift];
                        }
                    }
                    output[p*out_channels*out_w*out_h + out_c*out_w*out_h + cur_w*out_h + cur_h] += res;
                }
                output[p*out_channels*out_w*out_h + out_c*out_w*out_h + cur_w*out_h + cur_h] += bias[out_c];

                // printf("%.2f \n", x, y, output[p*out_channels*out_w*out_h + out_c*out_w*out_h + cur_w*out_h + cur_h]);
            
                cur_h++;
            }
            cur_w++;
        }
    }
}


void conv_forward(float *input, float *weights, float *bias, float *output, 
                int in_channels, int out_channels, int kernel_size, int padding, int strides, int w, int h)
{
    /**
     * Conv2D forward
     * 
     * Input:
     *      input
     *      weights
     *      bias
     * Output:
     *      output
     * */
    conv_params cp;
    cp.input = input;
    cp.weights = weights;
    cp.bias = bias;
    cp.output = output;
    cp.in_channels = in_channels; cp.out_channels = out_channels; cp.kernel_size = kernel_size;
    cp.padding = padding; cp.strides = strides; cp.w = w; cp.h = h;

    pthread_t tid[BATCH_SIZE+1];
    for(int p=0; p<BATCH_SIZE; p++)
    {
        cp.batch_id = p;
        pthread_create(&tid[p], NULL, __ConvForward__, (void *)(&cp));
    }
    for(int p=0; p<BATCH_SIZE; p++)
        pthread_join(tid[p], NULL);

}


void conv_backward(float *in_error, float *out_error, float *input, float *weights,
                   float *w_deltas, float *b_deltas, int in_channels, int out_channels,
                   int in_w, int in_h, int padding, int kernel_size, int strides)
{
    /**
     * Conv2D backward
     * 
     * Input:
     *      out_error
     *      input
     *      weights
     * Output:
     *      in_error
     *      w_deltas
     *      b_deltas
     * */
    int out_w = (in_w+2*padding-kernel_size) / strides + 1,
        out_h = (in_h+2*padding-kernel_size) / strides + 1;

    // compute b_deltas
    for (int c=0; c<out_channels; c++)
    {
        for (int i=0; i<out_w*out_h; i++)
        {
            b_deltas[c] += out_error[c*out_w*out_h+i];
        }
    }


    unsigned int w_shift, in_shift, out_shift;

    for (int in_c = 0; in_c < in_channels; in_c++)
    {
        for (int out_c = 0; out_c < out_channels; out_c++)
        {
            for (int out_x = 0; out_x < out_w; out_x++)
            {
                for (int out_y = 0; out_y < out_h; out_y++)
                {
                    for (int i = 0; i < kernel_size; i++)
                    {
                        if (strides*out_x+i-padding < 0 | strides*out_x+i-padding >= in_w)
                            continue;

                        for (int j = 0; j < kernel_size; j++)
                        {
                            if (strides*out_y+j-padding < 0 | strides*out_y+j-padding >= in_h)
                                continue;
                            
                            out_shift = out_c*out_w*out_h + out_x*out_h + out_y;
                            w_shift = out_c*in_channels*kernel_size*kernel_size + in_c*kernel_size*kernel_size + i*kernel_size + j;

                            // compute w_deltas[out_c][in_c][i][j]
                            for(int p=0; p<BATCH_SIZE; p++)
                            {                                
                                in_shift = p*in_channels*in_w*in_h + in_c*in_w*in_h + (strides*out_x+i-padding)*in_h + (strides*out_y+j-padding);
                                w_deltas[w_shift] += input[in_shift] * out_error[out_shift];
                            }
                            w_deltas[w_shift] /= BATCH_SIZE;
/*                             if(w_deltas[w_shift]>4)
                            {
                                printf("$$$$$$$$$$$$$$$$$ out_error[out_shift]: %.2f \n", out_error[out_shift]);
                                printf("$$$$$$$$$$$$$$$$$ w_deltas[w_shift]: %.2f \n", w_deltas[w_shift]);
                                printf("$$$$$$$$$$$$$$$$$ w_deltas[w_shift] too big!!! $$$$$$$$$$$$$$$$$ \n\n");
                                //exit(-1);
                            }
 */
                            // compute in_error[in_c][][]
                            in_shift = in_c*in_w*in_h + (strides*out_x+i-padding)*in_h + (strides*out_y+j-padding);
                            w_shift = out_c*in_channels*kernel_size*kernel_size + in_c*kernel_size*kernel_size + (kernel_size-i-1)*kernel_size + j;                                
                            in_error[in_shift] += out_error[out_shift] * weights[w_shift];
                        }
                    }
                }
            }
        }
    }

}


void nonlinear_forward(float *x, int units)
{
    /**
     * forward of ReLU activation
     * 
     * Input:
     *      x
     *      units
     * Output:
     *      x
     * */
    for (int i = 0; i < units; i++)
    {
        x[i] = (x[i]>0)?x[i]:0;
    }
}


void nonlinear_backward(float *x, int units)
{
    /**
     * backward of ReLU activation
     * 
     * Input:
     *      x
     *      units
     * Output:
     *      x
     * */

    for (int i = 0; i < units; i++)
    {
        x[i] = (x[i]>0);
    }
}


void max_pooling_forward(float *input, float *output, int channels, int in_length, int strides, int pool_size)
{
    /**
     * max pooling forward for multi-channel image
     * 
     * Input:
     *      input
     * Output:
     *      output
     * */

    int o_x, o_y, o_length = in_length / strides;
    float pixel;

    for (int p=0; p<BATCH_SIZE; p++)
    {
        for (int c = 0; c < channels; c++)
        {
            o_x=0;
            for (int i = 0; i < in_length-strides+1; i += strides)
            {
                o_y=0;
                for (int j = 0; j < in_length-strides+1; j += strides)
                {
                    /**
                     * inputs[i ~ i+pool_size][j ~ j+pool_size]
                     * outputs[o_x][o_j]
                     * */

                    pixel = input[p*channels*in_length*in_length+c*in_length*in_length+i*in_length+j];
                    for (int fx=i; fx<MIN(i+pool_size,in_length); fx++)
                    {
                        for (int fy=j; fy<MIN(j+pool_size,in_length); fy++)
                        {                        
                            pixel = MAX(pixel, input[p*channels*in_length*in_length+c*in_length*in_length+fx*in_length+fy]);
                        }
                    }
                    output[p*channels*o_length*o_length + c*o_length*o_length + o_x + o_y*o_length] = pixel;
                    o_y++;
                }
                o_x++;
            }
        }
    }

}


void max_pooling_backward(int channels, int pool_size, int in_length, float *in_error, float *out_error, float *input)
{
    /**
     * max pooling backward for multi-channel image
     * 
     * Input:
     *      out_error
     *      input
     * Output:
     *      in_error
     * */

    int out_length = ceil((float)in_length / pool_size);
    int max_idx, max_idy;
    float max_value, cur_value;
    int x, y;

    for (int c=0; c<channels; c++)
    {
        for (int i=0; i<out_length; i++)
        {
            for (int j=0; j<out_length; j++)
            {
                for (int p=0; p<BATCH_SIZE; p++)
                {
                    //
                    // output[c][i][j]
                    //
                    x = i*pool_size;    
                    y = j*pool_size;
                    cur_value = input[p*channels*in_length*in_length + c*in_length*in_length + y*in_length + x];
                    max_value = cur_value;
                    
                    while ( x<MIN((i + 1) * pool_size, in_length) )
                    {
                        while ( y<MIN((j + 1) * pool_size, in_length) )
                        {
                            cur_value = input[p*channels*in_length*in_length + c*in_length*in_length + y*in_length + x];
                            if(cur_value>=max_value)
                            {
                                max_value = cur_value;
                                max_idx = x;
                                max_idy = y;
                            }
                            y++;
                        }
                        x++;
                    }
                    in_error[c*in_length*in_length + max_idy*in_length + max_idx] += out_error[c*out_length*out_length + j*out_length + i]/BATCH_SIZE;
                }
            }
        }
    }

}


void fc_forward(float *input, float *output, float *weights, float *bias, int in_units, int out_units)
{
    /**
     * fully connected layer forward
     * 
     * Input:
     *      input    (BATCH_SIZE, in_units)
     *      weights  (in_units, out_units)
     *      bias     (out_units)
     * Output:
     *      out      (BATCH_SIZE, out_units)
     * */

    for (int p=0; p<BATCH_SIZE; p++)
    {
        for (int i = 0; i < in_units; i++)
        {
            for (int j = 0; j < out_units; j++)
            {
                output[p*out_units + j] += input[p*in_units + i] * weights[j * out_units + i];
            }
        }
        for (int i = 0; i < out_units; i++)
        {
            output[p*out_units + i] += bias[i];
        }
    }
}
 

void fc_backward(const float *input, const float *weights, float *in_error, const float *out_error,
                 float *w_deltas, float *b_deltas, int in_units, int out_units)
{
    /**
     * fully connected layer backward
     *
     * Input:
     *      input
     *      weights
     *      out_error
     *      out_units
     * Output:
     *      in_error
     *      w_deltas
     *      b_deltas
     * */
    for (int i=0; i<in_units; i++)
    {
        for (int j=0; j<out_units; j++)
        {
            in_error[i] += weights[i*out_units + j] * out_error[j];
            for (int r=0; r<BATCH_SIZE; r++)
            {
                w_deltas[i*out_units + j] += input[r*in_units + i] * out_error[j];
            }
            w_deltas[i*out_units + j] /= BATCH_SIZE;
        }
    }

    for (int p = 0; p < out_units; p++)
        b_deltas[p] = out_error[p];

}


void batch_normalization_forward(float *input, float *output, float *gamma, float *beta, int units, BN_params *bnp)
{
    /**
     * batch normalization forward
     * 
     * 
     * input    (BATCH_SIZE, units)
     * output   (BATCH_SIZE, units)
     * */
    bnp->avg = (float *)malloc(sizeof(float)*units);
    bnp->var = (float *)malloc(sizeof(float)*units);
    bnp->x_norm = (float *)malloc(sizeof(float)*BATCH_SIZE*units);

    // calculate mean for each unit along batch axis
    for (int i = 0; i < units; i++)
    {
        bnp->avg[i] = 0;
        for (int p = 0; p < BATCH_SIZE; p++)
        {
            bnp->avg[i] += input[p * units + i];
        }
        bnp->avg[i] /= BATCH_SIZE;
    }

    // calculate variance for each unit along batch axis
    for (int i = 0; i < units; i++)
    {
        bnp->var[i] = 0;
        for (int p = 0; p < BATCH_SIZE; p++)
        {
            bnp->var[i] += (input[p*units + i] - bnp->avg[i]) * (input[p*units + i] - bnp->avg[i]);
        }
        bnp->var[i] /= BATCH_SIZE;
    }

    for (int i = 0; i < units; i++)
    {
        for (int p = 0; p < BATCH_SIZE; p++)
        {
            bnp->x_norm[p*units+i] = (input[p*units + i] - bnp->avg[i]) / sqrt(bnp->var[i] + EPSILON); 
            //output[p*units+i] = gamma[i] * bnp->x_norm[p*units+i] + beta[i];
            output[p*units+i] = bnp->x_norm[p*units+i];
        }
    }

/*     for (int i = 0; i < units; i++)
    {
        for (int p = 0; p < BATCH_SIZE; p++)
            output[p*units+i] = input[p*units+i];
    } */
}


void batch_normalization_backward(float *in_error, const float *out_error, float *delta_gamma, float *delta_beta, 
                                    float *gamma, int units, BN_params *bnp)
{
    /**
     * batch normalization backward
     * 
     * Input:
     * Output:
     * */
    float   nn = 1.0/BATCH_SIZE;

/*   
    float *tmp = (float *)malloc(units * sizeof(float));
    for (int i = 0; i < units; i++)
    {
        for(int p=0; p<BATCH_SIZE; p++)
        {
            delta_gamma[i] += bnp->x_norm[p*units+i] * out_error[i];
        }
        delta_gamma[i] /= BATCH_SIZE;
        //delta_beta[i] += out_error[i];
        delta_beta[i]=0;

        tmp[i] = 0;
        for(int p=0; p<BATCH_SIZE; p++)
        {
            tmp[i] += bnp->x_norm[p*units+i];
        }
        tmp[i] /= BATCH_SIZE;
    }

    for (int i = 0; i < units; i++)
    {
        in_error[i] = gamma[i] * out_error[i] / sqrt(bnp->var[i]+EPSILON) * (1 - nn - nn*(1-nn)*tmp[i]*tmp[i]); 
        if(in_error[i]-out_error[i]>0.2)
        {
            printf("$$$$$$$$$$$$ batch_normalization_backward (1 - nn - nn*(1-nn)*tmp[i]*tmp[i]):%.2f \n", (1 - nn - nn*(1-nn)*tmp[i]*tmp[i]));
            printf("$$$$$$$$$$$$ batch_normalization_backward tmp[i]:%.2f \n", tmp[i]);
            printf("$$$$$$$$$$$$ batch_normalization_backward in_error:%.2f out_error:%.2f \n", in_error[i], out_error[i]);
        } 
    } 
    free(tmp);
*/

    
    for (int i = 0; i < units; i++)
    {
        delta_gamma[i] = 0;
        delta_beta[i] = 0;
        in_error[i] = out_error[i];
    }
    
    free(bnp->avg);
    free(bnp->var);
    free(bnp->x_norm);
}


void softmax_forward(float *x, int units)
{
    /**
     * softmax layer forward
     * 
     * Input:
     *      x    (BATCH_SIZE, units)
     * Output:
     *      x    (BATCH_SIZE, units)
     * */

    float sum;
    for (int p=0; p<BATCH_SIZE; p++)
    {
        sum = 0;
        for (int i = 0; i < units; i++)
        {
            sum += exp(x[p*units+i]);
        }

        for (int i = 0; i < units; i++)
        {
            x[p*units+i] = exp(x[p*units+i]) / sum;
        }
    }

}


void softmax_backward(float *error, int units)
{
    /**
     * softmax layer backward
     * 
     * Input:
     *      error   [units]
     * Output:
     *      error   [units]
     * */
    float *tmp = (float *)malloc(units*sizeof(float));
    memcpy(tmp, error, units*sizeof(float));
    for(int i=0; i<units; i++)
    {
        for(int j=0; j<units; j++)
        {
            if(i==j){
                error[j] += tmp[i] * (1-tmp[i]);
            }else{
                error[j] -= tmp[j] * tmp[i];
            }
        }
    }
    free(tmp);

    for(int i=0; i<units; i++)
    {
        error[i] /= units;
    }

}


void dropout(float *x, float prob, int units)
{
    /**
     * dropout regularization
     * 
     * Input:
     *      x   [BATCH_SIZE, units]
     *      prob    prob~(0,1)
     *      units   
     * Output:
     *      x   [BATCH_SIZE, units]
     * */
    for(int p=0; p<BATCH_SIZE; p++)
    {
        for(int i=0; i<units; i++)
        {
            if(rand()%100 < prob*100)
            {
                x[p*units+i] = 0;
            }
        }
    }
}


void zero_grads(Alexnet *grads)
{
    /**
     * set gradient struct to zero
     * */

    memset(grads->C1_weights, 0, C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L);
    memset(grads->C2_weights, 0, C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L);
    memset(grads->C3_weights, 0, C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L);
    memset(grads->C4_weights, 0, C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L);
    memset(grads->C5_weights, 0, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L);
    memset(grads->FC6weights, 0, C5_CHANNELS*FC6_LAYER*FC6KERNEL_L*FC6KERNEL_L);
    memset(grads->FC7weights, 0, FC6_LAYER*FC7_LAYER);
    memset(grads->FC8weights, 0, FC7_LAYER*OUT_LAYER);

    memset(grads->C1_bias, 0, C1_CHANNELS);
    memset(grads->C2_bias, 0, C2_CHANNELS);
    memset(grads->C3_bias, 0, C3_CHANNELS);
    memset(grads->C4_bias, 0, C4_CHANNELS);
    memset(grads->C5_bias, 0, C5_CHANNELS);
    memset(grads->FC6bias, 0, FC6_LAYER);
    memset(grads->FC7bias, 0, FC7_LAYER);
    memset(grads->FC8bias, 0, OUT_LAYER);

    memset(grads->BN1_gamma, 0, C1_CHANNELS*FEATURE1_L*FEATURE1_L* sizeof(float));
    memset(grads->BN1_b, 0, C1_CHANNELS*FEATURE1_L*FEATURE1_L* sizeof(float));
    memset(grads->BN2_gamma, 0, C2_CHANNELS*FEATURE2_L*FEATURE2_L* sizeof(float));
    memset(grads->BN2_b, 0, C2_CHANNELS*FEATURE2_L*FEATURE2_L* sizeof(float));
    memset(grads->BN3_gamma, 0, C3_CHANNELS*FEATURE3_L*FEATURE3_L* sizeof(float));
    memset(grads->BN3_b, 0, C3_CHANNELS*FEATURE3_L*FEATURE3_L* sizeof(float));
    memset(grads->BN4_gamma, 0, C4_CHANNELS*FEATURE4_L*FEATURE4_L* sizeof(float));
    memset(grads->BN4_b, 0, C4_CHANNELS*FEATURE4_L*FEATURE4_L* sizeof(float));
    memset(grads->BN5_gamma, 0, C5_CHANNELS*FEATURE5_L*FEATURE5_L* sizeof(float));
    memset(grads->BN5_b, 0, C5_CHANNELS*FEATURE5_L*FEATURE5_L* sizeof(float));

}


void zero_feats(Feature *feats)
{
    memset(feats->input, 0, BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L);
    memset(feats->C1, 0, BATCH_SIZE*C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    memset(feats->BN1, 0, BATCH_SIZE*C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    memset(feats->P1, 0, BATCH_SIZE*C1_CHANNELS*POOLING1_L*POOLING1_L);

    memset(feats->C2, 0, BATCH_SIZE*C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    memset(feats->BN2, 0, BATCH_SIZE*C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    memset(feats->P2, 0, BATCH_SIZE*C2_CHANNELS*POOLING2_L*POOLING2_L);

    memset(feats->C3, 0, BATCH_SIZE*C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    memset(feats->BN3, 0, BATCH_SIZE*C3_CHANNELS*FEATURE3_L*FEATURE3_L);

    memset(feats->C4, 0, BATCH_SIZE*C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    memset(feats->BN4, 0, BATCH_SIZE*C4_CHANNELS*FEATURE4_L*FEATURE4_L);

    memset(feats->C5, 0, BATCH_SIZE*C5_CHANNELS*FEATURE5_L*FEATURE5_L);
    memset(feats->BN5, 0, BATCH_SIZE*C5_CHANNELS*FEATURE5_L*FEATURE5_L);
    memset(feats->P5, 0, BATCH_SIZE*C5_CHANNELS*POOLING5_L*POOLING5_L);

    memset(feats->FC6, 0, BATCH_SIZE*FC6_LAYER);

    memset(feats->FC7, 0, BATCH_SIZE*FC7_LAYER);

    memset(feats->output, 0, BATCH_SIZE*OUT_LAYER);
}


void xavier_initialization(float *p, int n, int in_units, int out_units)
{
    float boundary = sqrt(12.0/(in_units+out_units));
    for(int shift=0; shift<n; shift++)
    {
        p[shift] = (rand()%100 *1.0 )/100.0 * (2*boundary) - boundary;
    }
}


void global_params_initialize(Alexnet *net)
{
    /**
     * initialize all the trainable parameters
     * */
    xavier_initialization(net->C1_weights, C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L, IN_CHANNELS*FEATURE0_L*FEATURE0_L, C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    xavier_initialization(net->C2_weights, C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L, C1_CHANNELS*POOLING1_L*POOLING1_L, C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    xavier_initialization(net->C3_weights, C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L, C2_CHANNELS*POOLING2_L*POOLING2_L, C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    xavier_initialization(net->C4_weights, C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L, C3_CHANNELS*FEATURE3_L*FEATURE3_L, C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    xavier_initialization(net->C5_weights, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L, C4_CHANNELS*FEATURE4_L*FEATURE4_L, C5_CHANNELS*FEATURE5_L*FEATURE5_L);
    xavier_initialization(net->FC6weights, C5_CHANNELS*FC6_LAYER*FC6KERNEL_L*FC6KERNEL_L, C5_CHANNELS*POOLING5_L*POOLING5_L, FC6_LAYER);
    xavier_initialization(net->FC7weights, FC6_LAYER*FC7_LAYER, FC6_LAYER, FC7_LAYER);
    xavier_initialization(net->FC8weights, FC7_LAYER*OUT_LAYER, FC7_LAYER, OUT_LAYER);

    int i;
    for(i=0; i<C1_CHANNELS; i++)
        net->C1_bias[i] = 1;
    for(i=0; i<C2_CHANNELS; i++)
        net->C2_bias[i] = 1;
    for(i=0; i<C3_CHANNELS; i++)
        net->C3_bias[i] = 1;
    for(i=0; i<C4_CHANNELS; i++)
        net->C4_bias[i] = 1;
    for(i=0; i<C5_CHANNELS; i++)
        net->C5_bias[i] = 1;
    for(i=0; i<FC6_LAYER; i++)
        net->FC6bias[i] = 1;
    for(i=0; i<FC7_LAYER; i++)
        net->FC7bias[i] = 1;
    for(i=0; i<OUT_LAYER; i++)
        net->FC8bias[i] = 1;

    for(i=0; i<C1_CHANNELS*FEATURE1_L*FEATURE1_L; i++)
    {
        net->BN1_gamma[i] = 1;
        net->BN1_b[i] = 0;
    }
    for(i=0; i<C2_CHANNELS*FEATURE2_L*FEATURE2_L; i++)
    {
        net->BN2_gamma[i] = 1;
        net->BN2_b[i] = 0;
    }
    for(i=0; i<C3_CHANNELS*FEATURE3_L*FEATURE3_L; i++)
    {
        net->BN3_gamma[i] = 1;
        net->BN3_b[i] = 0;
    }

    for(i=0; i<C4_CHANNELS*FEATURE4_L*FEATURE4_L; i++)
    {
        net->BN4_gamma[i] = 1;
        net->BN4_b[i] = 0;
    }

    for(i=0; i<C5_CHANNELS*FEATURE5_L*FEATURE5_L; i++)
    {
        net->BN4_gamma[i] = 1;
        net->BN4_b[i] = 0;
    }
};


void net_forward(const Alexnet *alexnet, Feature *feats)
{
    struct timespec start, finish;
    double duration;

    clock_gettime(CLOCK_MONOTONIC, &start);
    conv_forward(feats->input, alexnet->C1_weights, alexnet->C1_bias, feats->C1, IN_CHANNELS, C1_CHANNELS, C1_KERNEL_L, 4, C1_STRIDES, FEATURE0_L, FEATURE0_L);
    batch_normalization_forward(feats->C1, feats->BN1, alexnet->BN1_gamma, alexnet->BN1_b, C1_CHANNELS*FEATURE1_L*FEATURE1_L, &(alexnet->bnp1));
    nonlinear_forward(feats->BN1, C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_forward1 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    max_pooling_forward(feats->BN1, feats->P1, C1_CHANNELS, FEATURE1_L, 2, 3);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" max_pooling_forward1 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    conv_forward(feats->P1, alexnet->C2_weights, alexnet->C2_bias, feats->C2, C1_CHANNELS, C2_CHANNELS, C2_KERNEL_L, 1, C2_STRIDES, FEATURE1_L, FEATURE1_L);
    batch_normalization_forward(feats->C2, feats->BN2, alexnet->BN2_gamma, alexnet->BN2_b, C2_CHANNELS*FEATURE2_L*FEATURE2_L, &(alexnet->bnp2));
    nonlinear_forward(feats->BN2, C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_forward2 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    max_pooling_forward(feats->BN2, feats->P2, C2_CHANNELS, FEATURE2_L, 2, 3);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" max_pooling_forward2 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    conv_forward(feats->P2, alexnet->C3_weights, alexnet->C3_bias, feats->C3, C2_CHANNELS, C3_CHANNELS, C3_KERNEL_L, 1, C3_STRIDES, FEATURE2_L, FEATURE2_L);
    batch_normalization_forward(feats->C3, feats->BN3, alexnet->BN3_gamma, alexnet->BN3_b, C3_CHANNELS*FEATURE3_L*FEATURE3_L, &(alexnet->bnp3));
    nonlinear_forward(feats->BN3, C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_forward3 duration: %.2fs \n", duration);
#endif
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    conv_forward(feats->BN3, alexnet->C4_weights, alexnet->C4_bias, feats->C4, C3_CHANNELS, C4_CHANNELS, C4_KERNEL_L, 1, C4_STRIDES, FEATURE3_L, FEATURE4_L);
    batch_normalization_forward(feats->C4, feats->BN4, alexnet->BN4_gamma, alexnet->BN4_b, C4_CHANNELS*FEATURE4_L*FEATURE4_L, &(alexnet->bnp4));
    nonlinear_forward(feats->BN4, C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_forward4 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    conv_forward(feats->BN4, alexnet->C5_weights, alexnet->C5_bias, feats->C5, C4_CHANNELS, C5_CHANNELS, C5_KERNEL_L, 1, C5_STRIDES, FEATURE4_L, FEATURE4_L);
    batch_normalization_forward(feats->C5, feats->BN5, alexnet->BN5_gamma, alexnet->BN5_b, C5_CHANNELS*FEATURE5_L*FEATURE5_L, &(alexnet->bnp5));
    nonlinear_forward(feats->BN5, C5_CHANNELS*FEATURE5_L*FEATURE5_L);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_forward5 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    max_pooling_forward(feats->BN5, feats->P5, C5_CHANNELS, FEATURE5_L, 2, 3);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" max_pooling_forward3 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    fc_forward(feats->P5, feats->FC6, alexnet->FC6weights, alexnet->FC6bias, C5_CHANNELS*POOLING5_L*POOLING5_L, FC6_LAYER);
    dropout(feats->FC6, DROPOUT_PROB, FC6_LAYER);
    nonlinear_forward(feats->FC6, FC6_LAYER);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" fc_forward1 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    fc_forward(feats->FC6, feats->FC7, alexnet->FC7weights, alexnet->FC7bias, FC6_LAYER, FC7_LAYER);
    dropout(feats->FC7, DROPOUT_PROB, FC7_LAYER);
    nonlinear_forward(feats->FC7, FC7_LAYER);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" fc_forward2 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    fc_forward(feats->FC7, feats->output, alexnet->FC8weights, alexnet->FC8bias, FC7_LAYER, OUT_LAYER);
    softmax_forward(feats->output, OUT_LAYER);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" fc_forward3 duration: %.2fs \n", duration);
#endif
}


void gradient_descent(Alexnet *alexnet, Alexnet *deltas, float a)
{
    /**
     * Mini-batch gradient descent
     * 
     * Input:
     *      alexnet all the trainable-weights
     *      deltas  deltas of alexnet weights
     *      a       learning rate
     * Output: 
     *      alexnet
     * */
    int i;
    float *p_w, *p_d;

    p_w = &(alexnet->C1_weights);
    p_d = &(deltas->C1_weights); 
    for(i=0; i<C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L; i++)
    {
        p_w[i] -= a * MIN(0.8,p_d[i]);
    }

    p_w = &(alexnet->C2_weights);
    p_d = &(deltas->C2_weights); 
    for(i=0; i<C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L; i++)
    {
        p_w[i] -= a * MIN(0.8,p_d[i]);
    }

    p_w = &(alexnet->C3_weights);
    p_d = &(deltas->C3_weights); 
    for(i=0; i<C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L; i++)
    {
        p_w[i] -= a * MIN(0.8,p_d[i]);
    }

    p_w = &(alexnet->C4_weights);
    p_d = &(deltas->C4_weights); 
    for(i=0; i<C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L; i++)
    {
        p_w[i] -= a * MIN(0.8,p_d[i]);
    }

    p_w = &(alexnet->C5_weights);
    p_d = &(deltas->C5_weights); 
    for(i=0; i<C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L; i++)
    {
        p_w[i] -= a * MIN(0.8,p_d[i]);
    }

    p_w = &(alexnet->FC6weights);
    p_d = &(deltas->FC6weights); 
    for(i=0; i<FC6_LAYER*C5_CHANNELS*FC6KERNEL_L*FC6KERNEL_L; i++)
    {
        p_w[i] -= a * MIN(0.8,p_d[i]);
    }

    p_w = &(alexnet->FC7weights);
    p_d = &(deltas->FC7weights); 
    for(i=0; i<FC6_LAYER*FC7_LAYER; i++)
    {
        p_w[i] -= a * MIN(0.8,p_d[i]);
    }

    p_w = &(alexnet->FC8weights);
    p_d = &(deltas->FC8weights); 
    for(i=0; i<FC7_LAYER*OUT_LAYER; i++)
    {
        p_w[i] -= a * MIN(0.8,p_d[i]);
    }

    for(i=0; i<C1_CHANNELS; i++)
    {
        alexnet->C1_bias[i] -= a * deltas->C1_bias[i];
    }

    for(i=0; i<C2_CHANNELS; i++)
    {
        alexnet->C2_bias[i] -= a * deltas->C2_bias[i];
    }

    for(i=0; i<C3_CHANNELS; i++)
    {
        alexnet->C3_bias[i] -= a * deltas->C3_bias[i];
    }

    for(i=0; i<C4_CHANNELS; i++)
    {
        alexnet->C4_bias[i] -= a * deltas->C4_bias[i];
    }

    for(i=0; i<C5_CHANNELS; i++)
    {
        alexnet->C5_bias[i] -= a * deltas->C5_bias[i];
    }

    for(i=0; i<FC6_LAYER; i++)
    {
        alexnet->FC6bias[i] -= a * deltas->FC6bias[i];
    }

    for(i=0; i<FC7_LAYER; i++)
    {
        alexnet->FC7bias[i] -= a * deltas->FC7bias[i];
    }

    for(i=0; i<OUT_LAYER; i++)
    {
        alexnet->FC8bias[i] -= a * deltas->FC8bias[i];
    }

    for(i=0; i<C1_CHANNELS*FEATURE1_L*FEATURE1_L; i++)
    {
        alexnet->BN1_gamma[i] -= a * deltas->BN1_gamma[i];
        alexnet->BN1_b[i] -= a * deltas->BN1_b[i]; 
    }
    for(i=0; i<C2_CHANNELS*FEATURE2_L*FEATURE2_L; i++)
    {
        alexnet->BN2_gamma[i] -= a * deltas->BN2_gamma[i];
        alexnet->BN2_b[i] -= a * deltas->BN2_b[i]; 
    }
    for(i=0; i<C3_CHANNELS*FEATURE3_L*FEATURE3_L; i++)
    {
        alexnet->BN3_gamma[i] -= a * deltas->BN3_gamma[i];
        alexnet->BN3_b[i] -= a * deltas->BN3_b[i]; 
    }

    for(i=0; i<C4_CHANNELS*FEATURE4_L*FEATURE4_L; i++)
    {
        alexnet->BN4_gamma[i] -= a * deltas->BN4_gamma[i];
        alexnet->BN4_b[i] -= a * deltas->BN4_b[i]; 
    }

    for(i=0; i<C5_CHANNELS*FEATURE5_L*FEATURE5_L; i++)
    {
        alexnet->BN5_gamma[i] -= a * deltas->BN5_gamma[i];
        alexnet->BN5_b[i] -= a * deltas->BN5_b[i]; 
    }
}


void cal_v_detlas(Alexnet *v, Alexnet *d)
{
    /**
     * calculate new v_deltas with old v_deltas and deltas
     * 
     * Input:
     *      v
     *      d
     * Output:
     *      v
     * */ 
    int i;
    float *p_w, *p_d;
    p_w = &(v->C1_weights);
    p_d = &(d->C1_weights); 
    for(i=0; i<C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L; i++)
    {
        p_w[i] = BETA*p_w[i]  + (1-BETA)*p_d[i];
    }

    p_w = &(v->C2_weights);
    p_d = &(d->C2_weights); 
    for(i=0; i<C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L; i++)
    {
        p_w[i] = BETA*p_w[i]  + (1-BETA)*p_d[i];
    }

    p_w = &(v->C3_weights);
    p_d = &(d->C3_weights); 
    for(i=0; i<C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L; i++)
    {
        p_w[i] = BETA*p_w[i] + (1-BETA)*p_d[i];
    }

    p_w = &(v->C4_weights);
    p_d = &(d->C4_weights); 
    for(i=0; i<C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L; i++)
    {
        p_w[i] = BETA*p_w[i] + (1-BETA)*p_d[i];
    }

    p_w = &(v->C5_weights);
    p_d = &(d->C5_weights); 
    for(i=0; i<C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L; i++)
    {
        p_w[i] = BETA*p_w[i] + (1-BETA)*p_d[i];
    }

    p_w = &(v->FC6weights);
    p_d = &(d->FC6weights); 
    for(i=0; i<C5_CHANNELS*FC6_LAYER*FC6KERNEL_L*FC6KERNEL_L; i++)
    {
        p_w[i] = BETA*p_w[i] + (1-BETA)*p_d[i];
    }

    p_w = &(v->FC7weights);
    p_d = &(d->FC7weights); 
    for(i=0; i<FC6_LAYER*FC7_LAYER; i++)
    {
        p_w[i] = BETA*p_w[i] + (1-BETA)*p_d[i];
    }

    p_w = &(v->FC8weights);
    p_d = &(d->FC8weights); 
    for(i=0; i<FC7_LAYER*OUT_LAYER; i++)
    {
        p_w[i] = BETA*p_w[i] + (1-BETA)*p_d[i];
    }

    for(i=0; i<C1_CHANNELS; i++)
    {
        v->C1_bias[i] = BETA * v->C1_bias[i] + (1-BETA) * d->C1_bias[i];
    }

    for(i=0; i<C2_CHANNELS; i++)
    {
        v->C2_bias[i] = BETA * v->C2_bias[i] + (1-BETA) * d->C2_bias[i];
    }

    for(i=0; i<C3_CHANNELS; i++)
    {
        v->C3_bias[i] = BETA * v->C3_bias[i] + (1-BETA) * d->C3_bias[i];
    }

    for(i=0; i<C4_CHANNELS; i++)
    {
        v->C4_bias[i] = BETA * v->C4_bias[i] + (1-BETA) * d->C4_bias[i];
    }

    for(i=0; i<C5_CHANNELS; i++)
    {
        v->C5_bias[i] = BETA * v->C5_bias[i] + (1-BETA) * d->C5_bias[i];
    }

    for(i=0; i<FC6_LAYER; i++)
    {
        v->FC6bias[i] = BETA * v->FC6bias[i] + (1-BETA) * d->FC6bias[i];
    }

    for(i=0; i<FC7_LAYER; i++)
    {
        v->FC7bias[i] = BETA * v->FC7bias[i] + (1-BETA) * d->FC7bias[i];
    }

    for(i=0; i<OUT_LAYER; i++)
    {
        v->FC8bias[i] = BETA * v->FC8bias[i] + (1-BETA) * d->FC8bias[i];
    }



    for(i=0; i<C1_CHANNELS*FEATURE1_L*FEATURE1_L; i++)
    {
        v->BN1_gamma[i] = BETA * v->BN1_gamma[i] + (1-BETA) * d->BN1_gamma[i];
        v->BN1_b[i] = BETA * v->BN1_b[i] + (1-BETA) * d->BN1_b[i];
    }
    for(i=0; i<C2_CHANNELS*FEATURE2_L*FEATURE2_L; i++)
    {    
        v->BN2_gamma[i] = BETA * v->BN2_gamma[i] + (1-BETA) * d->BN2_gamma[i];
        v->BN2_b[i] = BETA * v->BN2_b[i] + (1-BETA) * d->BN2_b[i];
    }
    for(i=0; i<C3_CHANNELS*FEATURE3_L*FEATURE3_L; i++)
    {    
        v->BN3_gamma[i] = BETA * v->BN3_gamma[i] + (1-BETA) * d->BN3_gamma[i];
        v->BN3_b[i] = BETA * v->BN3_b[i] + (1-BETA) * d->BN3_b[i];
    }

    for(i=0; i<C4_CHANNELS*FEATURE4_L*FEATURE4_L; i++)
    {
        v->BN4_gamma[i] = BETA * v->BN4_gamma[i] + (1-BETA) * d->BN4_gamma[i];
        v->BN4_b[i] = BETA * v->BN4_b[i] + (1-BETA) * d->BN4_b[i];
    }

    for(i=0; i<C5_CHANNELS*FEATURE5_L*FEATURE5_L; i++)
    {
        v->BN5_gamma[i] = BETA * v->BN5_gamma[i] + (1-BETA) * d->BN5_gamma[i];
        v->BN5_b[i] = BETA * v->BN5_b[i] + (1-BETA) * d->BN5_b[i];
    }
}


void net_backward(Feature *error, const Alexnet *alexnet, Alexnet *deltas, const Feature *feats, float lr)
{
    struct timespec start, finish;
    double duration;

    softmax_backward(error->output, OUT_LAYER);
    fc_backward(feats->FC7, alexnet->FC8weights, error->FC7, error->output, deltas->FC8weights, deltas->FC8bias, FC7_LAYER, OUT_LAYER);

    nonlinear_backward(error->FC7, FC7_LAYER);
    fc_backward(feats->FC6, alexnet->FC7weights, error->FC6, error->FC7, deltas->FC7weights, deltas->FC7bias, FC6_LAYER, FC7_LAYER);

    nonlinear_backward(error->FC6, FC6_LAYER);
    fc_backward(feats->P5, alexnet->FC6weights, error->P5, error->FC6, deltas->FC6weights, deltas->FC6bias, C5_CHANNELS*POOLING5_L*POOLING5_L, FC6_LAYER);

    max_pooling_backward(C5_CHANNELS, 3, FEATURE5_L, error->BN5, error->P5, feats->C5);

    clock_gettime(CLOCK_MONOTONIC, &start);
    nonlinear_backward(error->BN5, C5_CHANNELS*FEATURE5_L*FEATURE5_L);
    batch_normalization_backward(error->C5, error->BN5,  deltas->BN5_gamma, deltas->BN5_b, alexnet->BN4_gamma, C5_CHANNELS*FEATURE5_L*FEATURE5_L, &(alexnet->bnp5));
    conv_backward(error->BN4, error->C5, feats->BN4, alexnet->C5_weights, deltas->C5_weights, deltas->C5_bias, 
                    C4_CHANNELS, C5_CHANNELS, FEATURE4_L, FEATURE4_L, 1, C5_KERNEL_L, C5_STRIDES);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_backward5 duration: %.2fs \n", duration);
#endif
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    nonlinear_backward(error->BN4, C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    batch_normalization_backward(error->C4, error->BN4,  deltas->BN4_gamma, deltas->BN4_b, alexnet->BN4_gamma, C4_CHANNELS*FEATURE4_L*FEATURE4_L, &(alexnet->bnp4));
    conv_backward(error->BN3, error->C4, feats->BN3, alexnet->C4_weights, deltas->C4_weights, deltas->C4_bias, 
                    C3_CHANNELS, C4_CHANNELS, FEATURE3_L, FEATURE3_L, 1, C4_KERNEL_L, C4_STRIDES);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_backward4 duration: %.2fs \n", duration);
#endif

    clock_gettime(CLOCK_MONOTONIC, &start);
    nonlinear_backward(error->BN3, C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    batch_normalization_backward(error->C3, error->BN3,  deltas->BN3_gamma, deltas->BN3_b, alexnet->BN3_gamma, C3_CHANNELS*FEATURE3_L*FEATURE3_L, &(alexnet->bnp3));
    conv_backward(error->P2, error->C3, feats->P2, alexnet->C3_weights, deltas->C3_weights, deltas->C3_bias, 
                    C2_CHANNELS, C3_CHANNELS, POOLING2_L, POOLING2_L, 1, C3_KERNEL_L, C3_STRIDES);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_backward3 duration: %.2fs \n", duration);
#endif

    max_pooling_backward(C2_CHANNELS, 3, FEATURE2_L, error->BN2, error->P2, feats->C2);

    clock_gettime(CLOCK_MONOTONIC, &start);
    nonlinear_backward(error->BN2, C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    batch_normalization_backward(error->C2, error->BN2,  deltas->BN2_gamma, deltas->BN2_b, alexnet->BN2_gamma, C2_CHANNELS*FEATURE2_L*FEATURE2_L, &(alexnet->bnp2));
    conv_backward(error->P1, error->C2, feats->P1, alexnet->C2_weights, deltas->C2_weights, deltas->C2_bias, 
                    C1_CHANNELS, C2_CHANNELS, POOLING1_L, POOLING1_L, 1, C2_KERNEL_L, C2_STRIDES);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_backward2 duration: %.2fs \n", duration);
#endif

    max_pooling_backward(C1_CHANNELS, 3, FEATURE1_L, error->BN1, error->P1, feats->C1);

    clock_gettime(CLOCK_MONOTONIC, &start);
    nonlinear_backward(error->BN1, C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    batch_normalization_backward(error->C1, error->BN1, deltas->BN1_gamma, deltas->BN1_b, alexnet->BN1_gamma, C1_CHANNELS*FEATURE1_L*FEATURE1_L, &(alexnet->bnp1));
    conv_backward(error->input, error->C1, feats->input, alexnet->C1_weights, deltas->C1_weights, deltas->C1_bias, 
                    IN_CHANNELS, C1_CHANNELS, FEATURE0_L, FEATURE0_L, 4, C1_KERNEL_L, C1_STRIDES);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" conv_backward1 duration: %.2fs \n", duration);
#endif
    static Alexnet v_deltas;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cal_v_detlas(&v_deltas, deltas);
    gradient_descent(alexnet, deltas, lr);
    clock_gettime(CLOCK_MONOTONIC, &finish);
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" MomentunSGD duration: %.2fs \n", duration);
#endif 

}


void CatelogCrossEntropy(float *error, const float *preds, const float *labels, int units)
{
    /**
     * Compute error between 'preds' and 'labels', then send to 'error'
     * 
     * Input:
     *      preds   [BATCH_SIZE, units]
     *      labels  [BATCH_SIZE, units]
     *      units
     * Output:
     *      error   [units]
     *  
     * */

    for(int i=0; i<units; i++)
    {
        error[i]=0;
        for(int p=0; p<BATCH_SIZE; p++)      
        {
            error[i] -= labels[p*units+i]*log(preds[p*units+i])+(1-labels[p*units+i])*log(1-preds[p*units+i]);
        }
        error[i] = MIN(error[i]/BATCH_SIZE, 0.8);
    }

}


void CatelogCrossEntropy_backward(float *delta_preds, const float *preds, const float *labels, int units)
{
    /**
     * CatelogCrossEntropy backward
     * 
     * Input:
     *      preds   [BATCH_SIZE, units]
     *      labels  [BATCH_SIZE, units]
     *      units   
     * Output:
     *      delta_preds [units]
     * */

    for(int i=0; i<units; i++)
    {
        for(int p=0; p<BATCH_SIZE; p++)
        {
            delta_preds[i] += labels[p*units+i]*preds[p*units+i]+(1-labels[p*units+i])/(1-preds[p*units+i]);
        }
        delta_preds[i] = delta_preds[i]/BATCH_SIZE;
    }

}
