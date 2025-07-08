#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "upmem_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


__mram float d_X[NR_ELEM_PER_DPU];
__mram float d_W0[NR_ELEM_PER_DPU];
__mram float d_layer_1[NR_ELEM_PER_DPU];
__mram float d_buffer[NR_ELEM_PER_DPU];
__mram float d_layer_1_delta[NR_ELEM_PER_DPU];
__mram float d_W1[NR_ELEM_PER_DPU];
__mram float d_Y[NR_ELEM_PER_DPU];
__mram float d_pred[NR_ELEM_PER_DPU];
__mram float d_pred_delta[NR_ELEM_PER_DPU];

__mram float prediction[NR_ELEM_PER_DPU];

__dma_aligned float cache1[CACHE_SIZE]; // cache in WRAM to perform transfer
__dma_aligned float cache2[CACHE_SIZE]; // cache in WRAM to perform transfer


__dma_aligned float cachea[CACHE_SIZE]; // cache in WRAM to perform transfer
__dma_aligned float cacheb[CACHE_SIZE]; // cache in WRAM to perform transfer

//For KTest function
__mram float d_test[4*28];
__mram float d_test_pred[28];
__mram float d_test_l1[28*8];



//__host float X_res[NR_ELEM_PER_DPU * 1];
__mram float X_res[NR_ELEM_PER_DPU];

__host float *aux;
//float X_res[NR_ELEM_PER_DPU * 1];

int main()
{
	barrier_t *barr_addr = &my_barrier;
	//me() gets the id of the thread that is running
	if(me() == 0)
	{
		printf("Nr of Tasklets is: %d\n", NR_TASKLETS);
		perfcounter_config(COUNT_CYCLES, true);
	}
	barrier_wait(barr_addr);
	

	kFit(NR_TASKLETS, cache1, cache2, barr_addr, d_X,
	TRAINING_DIM, TRAINING_SIZE, d_Y, 1, d_layer_1, 8, d_layer_1_delta,
	d_pred, d_pred_delta, d_W0, d_W1, d_buffer,0.1);

	if(me() == 0)
	{
		printf("\n\nW0 starts:\n\n");
		for(int i = 0; i < TRAINING_DIM; i++)
		{
			for(int j = 0; j < L1_SIZE; j++)
			{
				printf("%f ", d_W0[i*L1_SIZE+j]);
			}
			printf("\n");
		}
	}

	if(me() == 0)
	{
		printf("\n\nW1 starts:\n\n");
		for(int i = 0; i < L1_SIZE; i++)
		{
			for(int j = 0; j < 1; j++)
			{
				printf("%f ", d_W1[i*1+j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}


	kTest(NR_TASKLETS, cachea, cacheb, barr_addr, d_test,
	4, 28, 1, d_test_l1, 8,
	d_test_pred, d_W0, d_W1);

	return 0;
}
