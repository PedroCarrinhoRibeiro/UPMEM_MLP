CC = gcc
CFLAGS = -Wall -g

SRCS = mlp_sequential.c
OBJS = $(SRCS:.c=.o)

TARGET = mlp_sequential

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

mlp_sequential.o: mlp_sequential.c
	$(CC) $(CFLAGS) -c mlp_sequential.c

clean:
	$(RM) $(OBJS) $(TARGET)
