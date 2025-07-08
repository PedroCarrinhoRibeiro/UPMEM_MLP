#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>

#define NR_DPUS1 8
#define NR_DPUS2 8

#define TRAINING_SIZE 128
#define TRAINING_DIM 128
#define OUTPUT_SIZE 1

#define y_dim (TRAINING_SIZE*OUTPUT_SIZE)

#define L1_SIZE 128
#define L2_SIZE 128

#define T float

#endif /* __COMMON_H__ */
