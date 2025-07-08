# UPMEM MLP



UPMEM and C parallel implementation of an Multi-Layer perceptron in UPMEM DIMMs.

The implementation follows a host-device fashion. The MRAM banks and for some cases the WRAM banks, of the UPMEM DIMMs are used.  
The training was implemented both on sequentially on a CPU and in single-DPU multithreading on the DIMMs.  
The inference was implemented both sequentially on a CPU and in multi-DPU multithreading on the DIMMs.  

The main kernel functions are the following:

exponential: macro for exponential. For further detail read our paper. note that math.h does not work  
Ceil: ceil function. note that math.h does not work  
min: min function. note that math.h does not work  
max: max function. note that math.h does not work  
kMartixByMatrixElementwise: multithreading element-wise matrix multiplication.  
kMartixSubstractMatrix: multithreading element-wise matrix subtraction.  
kSigmoid: applies sigmoid to a matrix. Multithreading is used  
kSigmoid_d: applies sigmoid derivative to a matrix. Multithreading is used  
kReLU: ReLu function. Multithreading is used  
kDot: Matrix multiplication. Multithreading is used  
kDot_m1_m2T: Matrix multiplication where matrix 2 is transposed. Multithreading is used.  
kDot_m1T_m2: Matrix multiplication where matrix 1 is transposed. Multithreading is used.  
kFit: trains an MLP  
kTest: inference on MLP.  

If you plan on using this code or parts of this code please cite: https://ieeexplore.ieee.org/abstract/document/10768222  
I highly advise that you read it before using the code.  
Author: Pedro Jose Carrinho Ribeiro  

```
@inproceedings{carrinho2024processing,
  title={Processing Multi-Layer Perceptrons In-Memory},
  author={Carrinho, Pedro and Ferraz, Oscar and Ferreira, Jo{\~a}o Dinis and Falevoz, Yann and Silva, Vitor and Falcao, Gabriel},
  booktitle={2024 IEEE Workshop on Signal Processing Systems (SiPS)},
  pages={7--12},
  year={2024},
  organization={IEEE}
}
```

## How to run

Just compile using make and then run. You can customize number of layers, number of neurons per layers, input dimensions, output dimensions, etc.

## Requirements

Either run on a UPMEM PIM-enabled machine or use UPMEM SDK: https://sdk.upmem.com/. The timer won't work on the SDK

## Project Organization

```
├── LICENSE                                       <- Open-source license if one is chosen
│
├── README.md                                     <- The top-level README for developers using this project
│
├── MLP_sequential_irisdataset_train_and_test     <- Train and test (iris dataset) in single-thread CPU     
│   │
│   ├── Makefile                                  <- Makefile to create the executables
│   │
│   ├── cpu_kernels.c                             <- Kernels that run sequentially on CPU
│   │
│   ├── mlp_sequential.c                          <- Train and test a 3-layer MLP sequentially
│   │
│   └──common
│      │
│      ├── common.h                               <- Header for layer dimsensions
│      │
│      └── timer.h                                <- timer
│
├── MLP_train_and_test_single_DPU_irisdataset     <- Train and test (iris dataset) on DPU (parallel)
│   │
│   ├── Makefile                                  <- Makefile to create the executables
│   │
│   ├── upmem_kernels.c                           <- Kernels that run in parallel on the DPUs
│   │
│   ├── mlp_host.c                                <- Host code to train and test a 3-layer MLP parallel
│   │
│   ├── mlp_dpu.c                                 <- DPU code to train and test a 3-layer MLP parallel
│   │
│   └──common
│      │
│      └── common.h                               <- Header for layer dimsensions
│
├── Inference_MLP_seq                             <- Large batch inference in single-thread CPU     
│   │
│   ├── Makefile                                  <- Makefile to create the executables
│   │
│   ├── cpu_kernels.c                             <- Kernels that run sequentially on CPU
│   │
│   ├── mlp_seq.c                                 <- Sequential inference in a 3-layer MLP
│   │
│   └──common
│      │
│      ├── common.h                               <- Header for layer dimsensions
│      │
│      └── timer.h                                <- timer
│
└── Inference_MLP_Upmem                           <- Train and test (iris dataset) on DPU (parallel)
    │
    ├── Makefile                                  <- Makefile to create the executables
    │
    ├── upmem_kernels.c                           <- Kernels that run in parallel on the DPUs
    │
    ├── host.c                                    <- Host code for parallel inference a 3-layer MLP
    │
    ├── KDOT_RELU.c                               <- DPU code for the 1st layer of the MLP using ReLU as activation
    │
    ├── KDOT_RELU2.c                              <- DPU code for the 2nd layer of the MLP using ReLU as activation
    │
    ├── KDOT_SIGMOID3.c                           <- DPU code for the output layer of the MLP using Sigmoid as activation
    │
    └──common
       │
       ├── common.h                               <- Header for layer dimsensions, and number of DPUs to allocate
       │     
       └── timer.h                                <- timer
```

--------

## Acknowledgements

Please cite my article: https://ieeexplore.ieee.org/abstract/document/10768222

```
@inproceedings{carrinho2024processing,
  title={Processing Multi-Layer Perceptrons In-Memory},
  author={Carrinho, Pedro and Ferraz, Oscar and Ferreira, Jo{\~a}o Dinis and Falevoz, Yann and Silva, Vitor and Falcao, Gabriel},
  booktitle={2024 IEEE Workshop on Signal Processing Systems (SiPS)},
  pages={7--12},
  year={2024},
  organization={IEEE}
}
```