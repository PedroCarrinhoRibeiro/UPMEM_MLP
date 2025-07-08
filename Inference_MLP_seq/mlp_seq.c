#include <stdio.h>
#include <stdlib.h>
#include "common/common.h"
#include "common/timer.h"
#include "cpu_kernels.c"


int main()
{
	srand(1);

	//WEIGHTS_0
	T *h_W0 = (T*)malloc(L1_SIZE*TRAINING_DIM*sizeof(T));
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	    h_W0[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	}

	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	T* h_layer_1 = (T*)malloc(L1_SIZE*TRAINING_SIZE*sizeof(T));

	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++)
	{
	    h_layer_1[i] = 0.0;
	}

	//WEIGHTS_1
	T *h_W1 = (T*)malloc(L1_SIZE*L2_SIZE*sizeof(T));
	for (int i = 0; i < L1_SIZE*L2_SIZE; i++)
	{
	    h_W1[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	}

	//LAYER_2, LAYER_2_DELTA AND BUFFER OF LAYER 2 SIZE
	T* h_layer_2 = (T*)malloc(L2_SIZE*TRAINING_SIZE*sizeof(T));

	for (int i = 0; i < L2_SIZE*TRAINING_SIZE; i++)
	{
	    h_layer_2[i] = 0.0;
	}

	//WEIGHTS_2
	T *h_W2 = (T*)malloc(L2_SIZE*OUTPUT_SIZE*sizeof(T));
	for (int i = 0; i < L2_SIZE*OUTPUT_SIZE; i++)
	{
	    h_W2[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	}

	//test vector for KTest function
	T *h_test = (T*)malloc(TRAINING_DIM*TRAINING_SIZE*sizeof(T));
	for (int i = 0; i < TRAINING_DIM*TRAINING_SIZE; i++)
	{
	    h_test[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	}
	T *h_test_pred = (T*)malloc(TRAINING_SIZE*OUTPUT_SIZE*sizeof(T));

  	//timer
 	Timer timer;
	start(&timer, 0, 0);

	kTest(h_test,
	TRAINING_DIM, TRAINING_SIZE, OUTPUT_SIZE, h_layer_1, L1_SIZE, h_layer_2, L2_SIZE,
	h_test_pred, h_W0, h_W1, h_W2);

  	stop(&timer, 0);
	printf("MLP Sequential Version Time (ms): ");
	print(&timer, 0, 1);

	free(h_W0);
	free(h_W1);
	free(h_W2);
	free(h_layer_1);
	free(h_layer_2);
	free(h_test);
	free(h_test_pred);

	return 0;
}