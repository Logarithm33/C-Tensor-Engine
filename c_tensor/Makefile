CC = gcc
CFLAGS = -Wall -g -O3

TARGET = test_tensor

SRCS = main.c tensor.c
OBJS = $(SRCS:.c=.o)

all:$(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o:%.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean test memcheck

clean:
	rm -f $(OBJS) $(TARGET)

test:$(TARGET)
	./$(TARGET)

memcheck:$(TARGET)
	valgrind --leak-check=full ./$(TARGET)

