HOST_BINARY := mlp_host
DPU_BINARY := mlp_dpu

HOST_SOURCES := mlp_host.c
DPU_SOURCES := mlp_dpu.c

.PHONY: all clean run

all: ${HOST_BINARY} ${DPU_BINARY}

clean:
	rm -f ${HOST_BINARY} ${DPU_BINARY}

run: all
	./${HOST_BINARY}

${HOST_BINARY}: ${DPU_BINARY} ${HOST_SOURCES}
	clang -O2 -DDPU_BINARY='"$(realpath ${DPU_BINARY})"' `dpu-pkg-config --cflags --libs dpu` ${HOST_SOURCES} -o $@

${DPU_BINARY}: ${DPU_SOURCES}
	clang --target=dpu-upmem-dpurte -DNR_TASKLETS=16 -DSTACK_SIZE_DEFAULT=1024 -O2 ${DPU_SOURCES} -o $@
