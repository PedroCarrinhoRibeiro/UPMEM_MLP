#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

static union
{
    double d;
    struct {
      int j,i;
    } n;
 } _eco;

#define EXP_A 1512775
#define EXP_C 68243 /*see text for choice of values */
#define exponential(y) (_eco.n.i=EXP_A*(y)+(1072693248-EXP_C), _eco.d)

/*float exponential(float x)
{
    float sum = 1.0f; // initialize sum of series
 	int n = 100;
    for (int i = n - 1; i > 0; --i )
        sum = 1 + x * sum / i;
 
    return sum;
}*/

int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}

int min(int a, int b)
{
	return(a > b) ? b : a;
}

T max(T x,  T y) 
{
	return (x > y ? x : y);
}

__mram_ptr T* kMartixByMatrixElementwise(const int n_threads, const int width, const int height, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, struct barrier_t * barrier_address)
{

	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}

	if(me() < max_threads)
	{	
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			output[i] = m1[i] * m2[i];
		}
	}
	barrier_wait(barrier_address);
	return output;

}

__mram_ptr T* kMartixSubstractMatrix(const int n_threads, const int width, const int height, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, struct barrier_t * barrier_address)
{
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}
	barrier_wait(barrier_address);
 
	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			output[i] = m1[i] - m2[i];
		}
	}
	barrier_wait(barrier_address);
	return output;

}

__mram_ptr T* kSigmoid(const int n_threads, const int width, const int height, __mram_ptr T const *m1, __mram_ptr T *output, struct barrier_t * barrier_address)
{
    
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;

	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}

	
  	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			output[i] = 1.0 / (1.0 + exponential((int)-m1[i]));
		}
	}
	barrier_wait(barrier_address);


	return output;
}

__mram_ptr T* kSigmoid_inference(const int n_threads, const int width, const int height, __mram_ptr T const *m1, __mram_ptr T *output, struct barrier_t * barrier_address)
{
    
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;

	if(max_threads == 0)
		max_threads = 1;

	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}
	
  if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			output[i] = 1.0 / (1.0 + exponential((int)-m1[i]));
		}
	}
	barrier_wait(barrier_address);


	return output;
}

__mram_ptr T* kSigmoid_d(const int n_threads, const int width, const int height, __mram_ptr T const *m1, __mram_ptr T *output, struct barrier_t * barrier_address)
{
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}

	if(me() < max_threads)
	{	
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			output[i] = m1[i] * (1 - m1[i]);
		}
	}
	barrier_wait(barrier_address);
	return output;
}


__mram_ptr T* kReLU(const int n_threads, const int width, const int height, __mram_ptr T const *m1, __mram_ptr T *output, struct barrier_t * barrier_address)
{
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}

	if(me() < max_threads)
	{	
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			output[i] = max(0, m1[i]);
		}
	}
	barrier_wait(barrier_address);
	return output;
}



__mram_ptr T* kDot(const int n_threads, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	
	int max_threads = m1_rows * m2_columns / 2;
	T* t_output = (T*) mem_alloc(max_threads * sizeof(T));
	int n_elem = m1_rows * m2_columns / n_threads;
	
	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
 
	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			//printf("m2 cols: %d\n",m2_columns);
		    int r = (int)i / m2_columns;
		    int c = i % m2_columns;
		    t_output[me()] = 0.0;
		    for( int k = 0; k < m1_columns; ++k )
		    {
	        	t_output[me()] = t_output[me()] + (m1[r * m1_columns + k] * m2[k * m2_columns + c]);
		    }
		    output[i] = t_output[me()];
		    //printf("Stat is: %u. Value in column %d is %.10f\n",check_stack(),i, output[i]);
		}
	}
	barrier_wait(barrier_address);
	/*if(me() == 0)
	{
		for(int j = 0; j < m1_rows; j++)
		{
			for(int k = 0; k < m2_columns; k++)
			{
				printf("%.4f ", output[j*m2_columns+k]);
			}
			printf("\n");
		}
	}
	barrier_wait(barrier_address);*/
		
	return output;
}

__mram_ptr T* kDot_inference(const int n_threads, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	
	int max_threads = m1_rows * m2_columns / 2;
	T t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;
	
	if(max_threads == 0)
		max_threads = 1;
	
	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}


	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
		    int r = (int)(i / m2_columns);
		    int c = i % m2_columns;
		    t_output[me()] = 0.0;

		    for( int k = 0; k < m1_columns; ++k )
		    {
	        	t_output[me()] += m1[r * m1_columns + k] * m2[k * m2_columns + c];
		    }
		    output[i] = t_output[me()];
		    //printf("Stat is: %u. Value in column %d is %.10f\n",check_stack(),i, output[i]);
		}
	}
	barrier_wait(barrier_address);
	/*if(me() == 0)
	{
		for(int j = 0; j < m1_rows*m2_columns; j++)
		{
			printf("Value in column %d is %.10f\n",j, output[j]);
			//printf("%d\n",m1_columns);
		}
	}
	barrier_wait(barrier_address);*/
		
	return output;
}

__mram_ptr T* kDot_m1_m2T(const int n_threads, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, const int m1_rows, const int m1_columns, const int m2_rows, struct barrier_t * barrier_address)
{

	int max_threads = m1_rows*m2_rows / 2;
	T t_output[max_threads];
	int n_elem = m1_rows*m2_rows / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = m1_rows*m2_rows / max_threads;
	}
	//printf("n_elem value is: %d\n", n_elem);
	
  	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{

		    int r = (int)i / m2_rows;
		    int c = i % m2_rows;
		    t_output[me()] = 0.0;
		    int id_T;

		    for( int k = 0; k < m1_columns; ++k )
		    {	
		    	id_T = c * m1_columns + k;

		        t_output[me()] += m1[r * m1_columns + k] * m2[id_T];
		    }

		    output[i] = t_output[me()];
		}
	}
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr T* kDot_m1T_m2(const float lr, const int n_threads, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{

	int max_threads = m1_columns*m2_columns / 2;
	T t_output[max_threads];
	int n_elem = m1_columns*m2_columns / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = m1_columns*m2_columns / max_threads;
	}
 
	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
		    int r = (int)i / m2_columns;
		    int c = i % m2_columns;
		    t_output[me()] = 0.0;
		    int id_T;

		    for( int k = 0; k < m1_rows; ++k )
		    {
		    	id_T = k * m1_columns + r;
		        t_output[me()] += m1[id_T] * m2[k*m2_columns + c];
		    }
		    output[i] +=  lr * t_output[me()];     
		}
	}
	barrier_wait(barrier_address);
	/*if(me() == 0)
	{
		for(int i = 0; i < max_threads; i++)
		{
			printf("Value %d: %.10f\n", i, output[i]);
		}
	}
	barrier_wait(barrier_address);*/
	return output;
}