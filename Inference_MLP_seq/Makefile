CC = gcc
CFLAGS = -Wall -g

SRCS = mlp_seq.c
OBJS = $(SRCS:.c=.o)

TARGET = mlp_seq

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

mlp_sequential.o: mlp_seq.c
	$(CC) $(CFLAGS) -c mlp_seq.c

clean:
	$(RM) $(OBJS) $(TARGET)
