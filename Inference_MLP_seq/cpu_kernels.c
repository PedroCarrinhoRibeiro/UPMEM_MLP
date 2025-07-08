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

T* kMartixByMatrixElementwise(const int width, const int height, T const *m1, T const *m2, T *output)
{

	for(int i = 0; i < width * height; i+=1)
	{
		output[i] = m1[i] * m2[i];
	}
	
	return output;

}

T* kMartixSubstractMatrix(const int width, const int height, T const *m1, T const *m2, T *output)
{
	for(int i = 0; i < width * height; i+=1)
	{
		output[i] = m1[i] - m2[i];
	}
	return output;

}

T* kSigmoid(const int width, const int height, T const *m1, T *output)
{
	for(int i = 0; i < width * height; i+=1)
	{
		output[i] = 1.0 / (1.0 + exponential((int)-m1[i])); 
	}
	return output;
}

T* kSigmoid_d(const int width, const int height, T const *m1, T *output)
{
	for(int i = 0; i < width * height; i+=1)
	{
		output[i] = m1[i] * (1 - m1[i]);
	}
	return output;
}

T* kDot(T const *m1, T const *m2, T *output, const int m1_rows, const int m1_columns, const int m2_columns)
{	
	T t_output;
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
	}
		
	return output;
}

T* kDot_m1_m2T(T const *m1, T const *m2, T *output, const int m1_rows, const int m1_columns, const int m2_rows)
{
	T t_output;
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

T* kDot_m1T_m2(const T lr, T const *m1, T const *m2, T *output, const int m1_rows, const int m1_columns, const int m2_columns)
{
	T t_output;
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
	    output[i] += lr * t_output;    
	}
	return output;
}


void kFit(
		 T const *X, const int X_w,  const int X_h,
		 T const *y, const int y_w,
		 T *l1, const int l1_w,
		 T *l_1_d,
		 T *pred, T *pred_d,
		 T *W0,
		 T *W1,
		 T *buffer,
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
		 T *X, const int X_w,  const int X_h,
		 const int y_w,
		 T *l1, const int l1_w,
		 T *l2, const int l2_w,
		 T *pred,
		 T *W0,
		 T *W1,
		 T *W2)
{

    kSigmoid(l1_w, X_h, kDot(X, W0, l1, X_h, X_w, l1_w), l1);
    kSigmoid(l2_w, X_h, kDot(l1, W1, l2, X_h, l1_w, l2_w), l2);
    kSigmoid(y_w, X_h, kDot(l2, W2, pred, X_h, l2_w, y_w), pred);

    for(int i = 0; i < X_h; i++)
    {
    	for(int j = 0; j < X_w; j++)
    	{
    		printf("KTEST Prediction[%i] : %.10f\n", i, pred[i]);
    	}
	}
}