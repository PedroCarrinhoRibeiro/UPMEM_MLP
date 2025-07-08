#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"



static void free_dpus(struct dpu_set_t set)
{}

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus)
{
	DPU_ASSERT(dpu_alloc(1, NULL, set));
	DPU_ASSERT(dpu_load(*set, DPU_BINARY, NULL));

	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}


int main()
{
	srand(1);

	float h_X[TRAINING_SIZE*TRAINING_DIM] = {
5.8, 4.0, 1.2, 0.2,
5.7, 4.4, 1.5, 0.4,
5.4, 3.9, 1.3, 0.4,
5.1, 3.5, 1.4, 0.3,
5.7, 3.8, 1.7, 0.3,
5.1, 3.8, 1.5, 0.3,
5.4, 3.4, 1.7, 0.2,
5.1, 3.7, 1.5, 0.4,
4.6, 3.6, 1.0, 0.2,
5.1, 3.3, 1.7, 0.5,
4.8, 3.4, 1.9, 0.2,
5.0, 3.0, 1.6, 0.2,
5.0, 3.4, 1.6, 0.4,
5.2, 3.5, 1.5, 0.2,
5.2, 3.4, 1.4, 0.2,
4.7, 3.2, 1.6, 0.2,
4.8, 3.1, 1.6, 0.2,
5.4, 3.4, 1.5, 0.4,
5.2, 4.1, 1.5, 0.1,
5.5, 4.2, 1.4, 0.2,
4.9, 3.1, 1.5, 0.1,
5.0, 3.2, 1.2, 0.2,
5.5, 3.5, 1.3, 0.2,
4.9, 3.1, 1.5, 0.1,
4.4, 3.0, 1.3, 0.2,
5.1, 3.4, 1.5, 0.2,
5.0, 3.5, 1.3, 0.3,
4.5, 2.3, 1.3, 0.3,
4.4, 3.2, 1.3, 0.2,
5.0, 3.5, 1.6, 0.6,
5.1, 3.8, 1.9, 0.4,
4.8, 3.0, 1.4, 0.3,
5.1, 3.8, 1.6, 0.2,
4.6, 3.2, 1.4, 0.2,
5.3, 3.7, 1.5, 0.2,
5.0, 3.3, 1.4, 0.2,
7.0, 3.2, 4.7, 1.4,
6.4, 3.2, 4.5, 1.5,
6.9, 3.1, 4.9, 1.5,
5.5, 2.3, 4.0, 1.3,
6.5, 2.8, 4.6, 1.5,
5.7, 2.8, 4.5, 1.3,
6.3, 3.3, 4.7, 1.6,
4.9, 2.4, 3.3, 1.0,
6.6, 2.9, 4.6, 1.3,
5.2, 2.7, 3.9, 1.4,
5.0, 2.0, 3.5, 1.0,
5.9, 3.0, 4.2, 1.5,
6.0, 2.2, 4.0, 1.0,
6.1, 2.9, 4.7, 1.4,
5.6, 2.9, 3.6, 1.3,
6.7, 3.1, 4.4, 1.4,
5.6, 3.0, 4.5, 1.5,
5.8, 2.7, 4.1, 1.0,
6.2, 2.2, 4.5, 1.5,
5.6, 2.5, 3.9, 1.1,
5.9, 3.2, 4.8, 1.8,
6.1, 2.8, 4.0, 1.3,
6.3, 2.5, 4.9, 1.5,
6.1, 2.8, 4.7, 1.2,
6.4, 2.9, 4.3, 1.3,
6.6, 3.0, 4.4, 1.4,
6.8, 2.8, 4.8, 1.4,
6.7, 3.0, 5.0, 1.7,
6.0, 2.9, 4.5, 1.5,
5.7, 2.6, 3.5, 1.0,
5.5, 2.4, 3.8, 1.1,
5.5, 2.4, 3.7, 1.0,
5.8, 2.7, 3.9, 1.2,
6.0, 2.7, 5.1, 1.6,
5.4, 3.0, 4.5, 1.5,
6.0, 3.4, 4.5, 1.6,
6.7, 3.1, 4.7, 1.5,
6.3, 2.3, 4.4, 1.3,
5.6, 3.0, 4.1, 1.3,
5.5, 2.5, 4.0, 1.3,
5.5, 2.6, 4.4, 1.2,
6.1, 3.0, 4.6, 1.4,
5.8, 2.6, 4.0, 1.2,
5.0, 2.3, 3.3, 1.0,
5.6, 2.7, 4.2, 1.3,
5.7, 3.0, 4.2, 1.2,
5.7, 2.9, 4.2, 1.3,
6.2, 2.9, 4.3, 1.3,
5.1, 2.5, 3.0, 1.1,
5.7, 2.8, 4.1, 1.3,
6.3, 3.3, 6.0, 2.5,
5.8, 2.7, 5.1, 1.9,
7.1, 3.0, 5.9, 2.1,
6.3, 2.9, 5.6, 1.8,
6.5, 3.0, 5.8, 2.2,
7.6, 3.0, 6.6, 2.1,
4.9, 2.5, 4.5, 1.7,
7.3, 2.9, 6.3, 1.8,
6.7, 2.5, 5.8, 1.8,
7.2, 3.6, 6.1, 2.5,
6.5, 3.2, 5.1, 2.0,
6.4, 2.7, 5.3, 1.9,
6.8, 3.0, 5.5, 2.1,
5.7, 2.5, 5.0, 2.0,
5.8, 2.8, 5.1, 2.4,
6.4, 3.2, 5.3, 2.3,
6.5, 3.0, 5.5, 1.8,
7.7, 3.8, 6.7, 2.2,
7.7, 2.6, 6.9, 2.3,
6.0, 2.2, 5.0, 1.5,
6.9, 3.2, 5.7, 2.3,
5.6, 2.8, 4.9, 2.0,
7.7, 2.8, 6.7, 2.0,
6.3, 2.7, 4.9, 1.8,
6.7, 3.3, 5.7, 2.1,
7.2, 3.2, 6.0, 1.8,
6.2, 2.8, 4.8, 1.8,
6.1, 3.0, 4.9, 1.8,
6.4, 2.8, 5.6, 2.1,
7.2, 3.0, 5.8, 1.6,
7.4, 2.8, 6.1, 1.9,
7.9, 3.8, 6.4, 2.0,
6.4, 2.8, 5.6, 2.2,
6.3, 2.8, 5.1, 1.5,
6.1, 2.6, 5.6, 1.4,
7.7, 3.0, 6.1, 2.3};


	const int X_size = sizeof(h_X);
	float *d_X;

	//WEIGHTS_0
	const int W0_size = L1_SIZE*TRAINING_DIM*sizeof(float);
	float *h_W0 = (float*)malloc(W0_size);
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	    h_W0[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	}

	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	const int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(float);

	float* h_layer_1 = (float*)malloc(L1_size);
	float* h_layer_1_delta = (float*)malloc(L1_size);
	float* h_buffer = (float*)malloc(L1_size);

	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++){
	    h_layer_1[i] = 0.0;
	    h_buffer[i] = 0.0;
	    h_layer_1_delta[i] = 0.0;
	}

	//WEIGHTS_1
	const int W1_size = L1_SIZE*sizeof(float);
	float *h_W1 = (float*)malloc(W1_size);
	for (int i = 0; i < L1_SIZE; i++){
	    h_W1[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	}

	//Y
	const int y_dim = 122;

	float h_Y[y_dim] = {
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1};
	const int y_size = sizeof(h_Y);

	//PRED AND PRED_DELTA
	float* h_pred = (float*)malloc(y_size);
	float* h_pred_delta = (float*)malloc(y_size);
	for (int i = 0; i < TRAINING_SIZE; i++){
	    h_pred[i] = 0.0;
	    h_pred_delta[i] = 0.0;
	}

	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus);

    printf("DPUs allocated: %u\n", nr_dpus);

    
    DPU_ASSERT(dpu_copy_to(set, "d_X", 0, h_X, sizeof(float) * TRAINING_SIZE*TRAINING_DIM));
    DPU_ASSERT(dpu_copy_to(set, "d_W0", 0, h_W0, W0_size));

    DPU_ASSERT(dpu_copy_to(set, "d_layer_1", 0, h_layer_1, L1_size));
    DPU_ASSERT(dpu_copy_to(set, "d_buffer", 0, h_buffer, L1_size));
    DPU_ASSERT(dpu_copy_to(set, "d_layer_1_delta", 0, h_layer_1_delta, L1_size));

    DPU_ASSERT(dpu_copy_to(set, "d_W1", 0, h_W1, W1_size));

    DPU_ASSERT(dpu_copy_to(set, "d_Y", 0, h_Y, y_size));

    DPU_ASSERT(dpu_copy_to(set, "d_pred", 0, h_pred, y_size));
	DPU_ASSERT(dpu_copy_to(set, "d_pred_delta", 0, h_pred_delta, y_size));


	//test vector for KTest function
	const int test_size = 4*28;
	float h_test[test_size] = {
6.0, 1.9, 4.5, 1.5,
4.3, 3.0, 1.1, 0.1,
5.0, 1.0, 3.5, 1.0,
6.3, 1.8, 5.1, 1.5,
5.1, 3.8, 1.5, 0.3,
5.6, 1.9, 3.6, 1.3,
4.5, 1.3, 1.3, 0.3,
6.7, 3.0, 5.2, 1.3,
4.4, 3.2, 1.3, 0.2,
6.7, 3.1, 5.6, 1.4,
5.0, 3.2, 1.2, 0.2,
5.6, 3.0, 4.5, 1.5,
6.1, 1.8, 4.7, 1.2,
4.8, 3.4, 1.6, 0.2,
5.8, 1.6, 4.0, 1.2,
5.7, 1.9, 4.2, 1.3,
6.3, 1.7, 4.9, 1.8,
6.7, 3.0, 5.0, 1.7,
6.5, 1.8, 4.6, 1.5,
6.9, 3.1, 5.4, 1.1,
6.1, 3.0, 4.9, 1.8,
4.8, 3.1, 1.6, 0.2,
6.4, 3.1, 5.5, 1.8,
7.3, 1.9, 6.3, 1.8,
6.7, 3.1, 4.4, 1.4,
7.7, 3.8, 6.7, 1.2,
4.8, 3.0, 1.4, 0.3,
6.3, 3.4, 5.6, 1.4
};
	const int test_pred_size = 28;
	float h_test_pred[test_pred_size] = {
1,
0,
1,
1,
0,
1,
0,
1,
0,
1,
0,
1,
1,
0,
1,
1,
1,
1,
1,
1,
1,
0,
1,
1,
1,
1,
0,
1};
	float h_test_l1[28*L1_SIZE];
	for (int i = 0; i < 28*L1_SIZE; i++){
	    h_test_l1[i] = 0;
	}

	DPU_ASSERT(dpu_copy_to(set, "d_test", 0, h_test, sizeof(float)*test_size));
	DPU_ASSERT(dpu_copy_to(set, "d_test_pred", 0, h_test_pred, sizeof(float)*test_pred_size));
	DPU_ASSERT(dpu_copy_to(set, "d_test_l1", 0, h_test_l1, sizeof(float)*28*L1_SIZE));

    //Run DPUS
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));


	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu)
	{
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}


	free(h_W0);
	free(h_layer_1);
	free(h_layer_1_delta);
	free(h_buffer);
	free(h_W1);
	free(h_pred);
	free(h_pred_delta);
	DPU_ASSERT(dpu_free(set));
	return 0;
}
