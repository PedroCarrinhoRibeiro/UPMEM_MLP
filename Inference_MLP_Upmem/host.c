#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"
#include "common/timer.h"

#define KDOT_RELU_BINARY "KDOT_RELU"
#define KDOT_RELU2_BINARY "KDOT_RELU2"
#define KDOT_SIGMOID3_BINARY "KDOT_SIGMOID3"

int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus, int dpu_to_allocate)
{
	DPU_ASSERT(dpu_alloc(dpu_to_allocate, NULL, set));
	DPU_ASSERT(dpu_load(*set, KDOT_RELU_BINARY, NULL));
	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}

void Run_KDot_RELU(T *m1_, T *m2_, T *m3_, int m1_rows, int m1_cols, int m2_rows, int m2_cols, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[])
{

	//calculating block sizes to send to each dpu
	//usage of ceil and padding allow for blocks of same size
	//which is required by upmem assert functions for parallel
	//transfers
	//Note the blkocking must obey the following rules:
    //For m1:
    	//nr columns must be even
		//nr should be even but I don't think it is a must
	//For m2:
		//one of the dimensions must be the same as for m1 for multiplication to be possible.
		//the dimension that must be same depends if you want to transpose or not
	//The padding for matrix 2 since it will be the columns in the result matrix must occour in a way that it makes the number of
	//elements in the column even, that is why block2 is calculated diferently from block 1
    
    int block1_rows = Ceil((T)m1_rows/NR_DPUS1);
    int padding1 = block1_rows*NR_DPUS1-m1_rows;
	int block2_rows = Ceil((((T)m2_rows/NR_DPUS2)/2))*2;
    int padding2 = block2_rows*NR_DPUS2-m2_rows;
    //printf("padding2: %d\n", padding2);
    
	//allocate new matrices considering padding
    T *m1 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m1_cols));
    T *m2 = (T*)malloc(sizeof(T)*(m2_rows+padding2)*(m2_cols));
    T *m3 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m2_rows+padding2));

    //horizontal padding for m1
    for(int i = 0; i < m1_rows*m1_cols; i++)
    {
    	m1[i] = m1_[i];
    }

    for(int i = 0; i < padding1*m1_cols; i++)
    {
    	m1[(m1_rows*m1_cols)+i] = 0;
    }

    //horizontal padding for m2 because it's column major
    for(int i = 0; i < m2_rows*m2_cols; i++)
    {
    	m2[i] = m2_[i];
    }

    for(int i = 0; i < padding2*m2_cols; i++)
    {
    	m2[(m2_rows*m2_cols)+i] = 0;
    }


    //load dpu binary file
    //printf("You entered: %s\n.", binary_path);
    DPU_ASSERT(dpu_load(set, binary_path, NULL));
    //DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));

    //send matrix1 with parallel transfers
    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[each_dpu/NR_DPUS2* (block1_rows*m1_cols)]));
    }
    //DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * ((block1_rows*m1_cols)), DPU_XFER_DEFAULT));

    //send matrix2 with parallel transfers
	DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m2[(each_dpu%NR_DPUS2)*(block2_rows*m2_cols)]));
    }
    //DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    //printf("line jump is: %d\n",i);
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, 0, sizeof(T) * ((block2_rows*m2_cols)), DPU_XFER_DEFAULT));


	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

	//print dpu content if printf defined in dpu code
	/*DPU_FOREACH(set,dpu,each_dpu)
    {
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}*/

	//Copy outputs from DPUS to host
    for(int current_line = 0; current_line < block1_rows; current_line++)
    {
    	DPU_FOREACH(set,dpu,each_dpu)
    	{
    		
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[current_line*(m2_rows+padding2)+((each_dpu%NR_DPUS2)*block2_rows)+((each_dpu/NR_DPUS2)*block1_rows*(m2_rows+padding2))]));
    		//printf("val is: %d \n", current_line*(test2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(test2_cols+padding2)));
    	//DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
    	}
    	//printf("for ended with line: %d\n", current_line);	
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, current_line*(block2_rows)*sizeof(T), sizeof(T) * (block2_rows), DPU_XFER_DEFAULT));
    }

    /*for(int i = 0; i < m1_rows+padding1; i++)
    {
    	for(int j = 0; j < m2_rows+padding2; j++)
    	{
    		printf("%.1f ",m3[i*(m2_rows+padding2)+j]);
    	}
    	printf("\n");
    }*/

    //remove padding and copy to m3_
    //printf("\n\n\n\n");
    for(int i = 0; i < m1_rows; i++)
    {
    	for(int j = 0; j < m2_rows; j++)
    	{
    		m3_[i*m2_rows+j] = m3[i*m2_rows+(j+i*padding2)];
    		//printf("%f ", m3_[i*m2_cols+j]);
    	}
    	//printf("\n");
    }
}



void Run_KDot_Sigmoid_vector(T *m1_, T *m2_, T *m3_, int m1_rows, int m1_cols, int m2_rows, int m2_cols, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[])
{
    
    int block1_rows = Ceil(((T)m1_rows/(NR_DPUS1*NR_DPUS2))/2)*2;
    int padding1 = block1_rows*(NR_DPUS1*NR_DPUS2)-m1_rows;
    T *m1 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m1_cols));
    T *m3 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m2_cols));


    for(int i = 0; i < m1_rows*m1_cols; i++)
    {
    	m1[i] = m1_[i];
    }

    for(int i = 0; i < padding1*m1_cols; i++)
    {
    	m1[(m1_rows*m1_cols)+i] = 0.0;
    }


    DPU_ASSERT(dpu_load(set, binary_path, NULL));
 
    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[each_dpu* (block1_rows*m1_cols)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * ((block1_rows*m1_cols)), DPU_XFER_DEFAULT));
    
	DPU_FOREACH(set,dpu,each_dpu)
 	{
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m2_[0]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, 0, sizeof(T) * m2_rows, DPU_XFER_DEFAULT));


	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

	//print DPU content
	/*DPU_FOREACH(set,dpu,each_dpu)
    {
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}*/

	//Copy outputs from DPUS
    DPU_FOREACH(set,dpu,each_dpu)
    {
    		
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[each_dpu*block1_rows]));
    	//printf("val is: %d \n", current_line*(test2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(test2_cols+padding2)));
    }	
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, 0, sizeof(T) * block1_rows, DPU_XFER_DEFAULT));

    /*for(int i = 0; i < m1_rows; i++)
    {
    	for(int j = 0; j < m2_cols; j++)
    	{
    		printf("%f ",m3[i*m2_cols+j]);
    	}
    	printf("\n");
    }*/

    //remove padding and copy to m3_
    for(int i = 0; i < m1_rows; i++)
    {
    	m3_[i] = m3[i];
    }
}




int main()
{


	srand(1);
	
	T* h_X = (T*)malloc(TRAINING_SIZE*TRAINING_DIM*sizeof(T));
	for(int i = 0; i < TRAINING_SIZE*TRAINING_DIM; i++)
	{
		h_X[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		//h_X[i] = ((T)rand()%(T)(RAND_MAX))*5;
	}
	
	//WEIGHTS_0
	const int W0_size = TRAINING_DIM*L1_SIZE*sizeof(T);
	T *h_W0 = (T*)malloc(W0_size);
	for(int i = 0; i < TRAINING_DIM*L1_SIZE; i++)
	{
		h_W0[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		//h_W0[i] = ((T)rand()%(T)(RAND_MAX))*1;
	}

	//LAYER_1
	const int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(T);

	//output of previous layer
	T* h_layer_1 = (T*)malloc(L1_size);

	//WEIGHTS_1
	const int W1_size = L1_SIZE*L2_SIZE*sizeof(T);
	T *h_W1 = (T*)malloc(W1_size);
	for(int i = 0; i < L1_SIZE*L2_SIZE; i++)
	{
		h_W1[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		//h_W1[i] = ((T)rand()%(T)(RAND_MAX))*1;
	}

	//LAYER_2
	const int L2_size = L2_SIZE*TRAINING_SIZE*sizeof(T);

	//output of previous layer
	T* h_layer_2 = (T*)malloc(L2_size);

	//WEIGHTS_2
	const int W2_size = L2_SIZE*OUTPUT_SIZE*sizeof(T);
	T *h_W2 = (T*)malloc(W2_size);
	for(int i = 0; i < L2_SIZE*OUTPUT_SIZE; i++)
	{
		h_W2[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		//h_W2[i] = ((T)rand()/(T)(RAND_MAX))*1;
	}


	//PRED AND PRED_DELTA
	const int y_size = y_dim*sizeof(T);
	T* h_pred = (T*)malloc(y_size);

 	Timer timer;
	int warmup = 0;
	int reps = 1;

	for(int i = 0; i < reps+warmup; i++)
	{
		if(i >= warmup)
		{
			start(&timer, 0, i-warmup);
		}

		//Allocate set of DPUs
		struct dpu_set_t set, dpu;
	    uint32_t nr_dpus, each_dpu;
	    alloc_dpus(&set, &nr_dpus, NR_DPUS1*NR_DPUS2);
		Run_KDot_RELU(h_X, h_W0, h_layer_1, TRAINING_SIZE, TRAINING_DIM, L1_SIZE, TRAINING_DIM, set, dpu, nr_dpus, each_dpu, "d_X", "d_W0", "d_l1", KDOT_RELU_BINARY);
		Run_KDot_RELU(h_layer_1, h_W1, h_layer_2, TRAINING_SIZE, L1_SIZE, L2_SIZE, L1_SIZE, set, dpu, nr_dpus, each_dpu, "d_l1", "d_W1", "d_l2", KDOT_RELU2_BINARY);
		Run_KDot_Sigmoid_vector(h_layer_2, h_W2, h_pred, TRAINING_SIZE, L2_SIZE, L2_SIZE, OUTPUT_SIZE, set, dpu, nr_dpus, each_dpu, "d_l2", "d_W2", "d_pred", KDOT_SIGMOID3_BINARY);
	  	
	  	if(i >= warmup)
	  	{ 
	  		stop(&timer, 0);
	  	}
	  	DPU_ASSERT(dpu_free(set));
  	}
	printf("MLP Upmem Version Time (ms): ");
	print(&timer, 0, reps);

	printf("\n\nPrediction values are: \n");
	for(int i = 0; i < TRAINING_SIZE; i++)
	{
		printf("Prediction[%d]: %f\n",i, h_pred[i]);
	}
	printf("\n\n");

	free(h_X);
	free(h_W0);
	free(h_layer_1);
	free(h_W1);
	free(h_layer_2);
	free(h_W2);
	free(h_pred);

	return 0;
}


