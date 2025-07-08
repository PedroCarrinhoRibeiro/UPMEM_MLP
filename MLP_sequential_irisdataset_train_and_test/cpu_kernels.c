#include <stdio.h>
#include "common/common.h"

//#include "math.h"

#define NR_ELEM_PER_TASKLETS (TRAINING_SIZE*TRAINING_DIM/NR_TASKLETS)

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

int min(int a, int b)
{
	return(a > b) ? b : a;
}

float* kMartixByMatrixElementwise(const int width, const int height, float const *m1, float const *m2, float *output)
{

	for(int i = 0; i < width * height; i+=1)
	{
		output[i] = m1[i] * m2[i];
		//printf("Value of element wise %d is: %.10f\n", i, output[i]);
	}
	
	return output;

}

float* kMartixSubstractMatrix(const int width, const int height, float const *m1, float const *m2, float *output)
{
	for(int i = 0; i < width * height; i+=1)
	{
		output[i] = m1[i] - m2[i];
		//printf("Value %d is: %.10f\n", i, output[i]);
	}
	return output;

}

float* kSigmoid(const int width, const int height, float const *m1, float *output)
{
	for(int i = 0; i < width * height; i+=1)
	{
		output[i] = 1.0 / (1.0 + exponential((int)-m1[i]));
	}
	return output;
}

float* kSigmoid_d(const int width, const int height, float const *m1, float *output)
{
	for(int i = 0; i < width * height; i+=1)
	{
		output[i] = m1[i] * (1 - m1[i]);
	}
	return output;
}

float* kDot(float const *m1, float const *m2, float *output, const int m1_rows, const int m1_columns, const int m2_columns)
{	
	float t_output;
	for(int i = 0; i < m1_rows * m2_columns; i+=1)
	{
		//printf("m2 cols: %d\n",m2_columns);
	    int r = (int)(i / m2_columns);
	    int c = i % m2_columns;
	    t_output = 0.0;

	    for( int k = 0; k < m1_columns; ++k )
	    {
        	t_output += m1[r * m1_columns + k] * m2[k * m2_columns + c];
	    }
	    output[i] = t_output;
	    //barrier_wait(barrier_address);
	    //printf("Stat is: %u. Value in column %d is %.10f\n", m1[i]);
	}
	/*for(int i = 0; i < m1_rows*m1_columns; i++)
	{
		printf("X value %d is %.10f\n",i, m1[i]);
	}*/
		
	return output;
}

float* kDot_m1_m2T(float const *m1, float const *m2, float *output, const int m1_rows, const int m1_columns, const int m2_rows)
{
	float t_output;
	for(int i = 0; i < m1_rows*m2_rows; i+=1)
	{
	    int r = (int)i / m2_rows;
	    int c = i % m2_rows;
	    t_output = 0.0;
	    int id_T;

	    for( int k = 0; k < m1_columns; ++k )
	    {	
	    	id_T = c * m1_columns + k;
	        t_output += m1[r * m1_columns + k] * m2[id_T];
	    }

	    output[i] = t_output;
	}

	return output;
}

float* kDot_m1T_m2(const float lr, float const *m1, float const *m2, float *output, const int m1_rows, const int m1_columns, const int m2_columns)
{
	float t_output;
	//printf("Im KDotT and I'm using thread nr: %d. The max is : %d\n", me(), max_threads);	
	for(int i = 0; i < m1_columns*m2_columns; i+=1)
	{
	    int r = (int)i / m2_columns;
	    int c = i % m2_columns;
	    t_output = 0.0;
	    int id_T;

	    for( int k = 0; k < m1_rows; ++k )
	    {
	    	id_T = k * m1_columns + r;
	        t_output += m1[id_T] * m2[k*m2_columns + c];

	    }
	    //barrier_wait(barrier_address);
	    output[i] += lr * t_output;
      //printf("Value %d: %.10f\n", i, output[i]);
         
	}
	return output;
}


void kFit(
		 float const *X, const int X_w,  const int X_h,
		 float const *y, const int y_w,
		 float *l1, const int l1_w,
		 float *l_1_d,
		 float *pred, float *pred_d,
		 float *W0,
		 float *W1,
		 float *buffer,
		 const float lr)
{
	for (unsigned i = 0; i < 500; ++i)
	{
		//activation of FCL with no bias x2
		//printf("First Sigmoid: \n");
        kSigmoid(l1_w, X_h, kDot(X, W0, l1, X_h, X_w, l1_w), l1);
        //printf("Second Sigmoid: \n");
        kSigmoid( y_w, X_h, kDot(l1, W1, pred, X_h, l1_w, y_w), pred);
        //backpropagation: gradient of y - x
		//printf("Backprop: \n");
		kMartixByMatrixElementwise(y_w, X_h, kMartixSubstractMatrix(y_w, X_h, y, pred, pred_d), kSigmoid_d(y_w, X_h, pred, buffer), pred_d);
      	//printf("\n");
      	kMartixByMatrixElementwise(l1_w, X_h, kDot_m1_m2T(pred_d, W1, l_1_d, X_h, y_w, l1_w), kSigmoid_d( l1_w, X_h, l1, buffer), l_1_d);
        //update weights
        //printf("Update weights W1\n");
        kDot_m1T_m2(lr, l1, pred_d, W1, X_h, l1_w, y_w);
        //printf("Update weights W0\n");
        kDot_m1T_m2(lr, X, l_1_d, W0, X_h, X_w, l1_w);
    	//printf("iter: %d\n", i);
	}
    
		kSigmoid(l1_w, X_h, kDot(X, W0, l1, X_h, X_w, l1_w), l1);
    kSigmoid( y_w, X_h, kDot(l1, W1, pred, X_h, l1_w, y_w), pred);

    /*for (int i = 0; i < X_h; i++)
    {
			printf("Prediction[%i] : %.10f True Value[%i] : %.10f Error[%i] : %f\n", i, pred[i], i, y[i], i, pred[i] - y[i]);
		}*/
}

void kTest(
		 float const *X, const int X_w,  const int X_h,
		 const int y_w,
		 float *l1, const int l1_w,
		 float *pred,
		 float *W0,
		 float *W1)
{

	  //printf("\nKtest began");
    kSigmoid(l1_w, X_h, kDot(X, W0, l1, X_h, X_w, l1_w), l1);
    kSigmoid( y_w, X_h, kDot(l1, W1, pred, X_h, l1_w, y_w), pred);

    /*for(int i = 0; i < X_h; i++)
    {
    	for(int j = 0; j < X_w; j++)
    	{
    		printf("matrix = %.10f\n",X[i*X_w+j]);
    	}
			printf("KTEST Prediction[%i] : %.10f\n", i, pred[i]);
		}*/ 
}

