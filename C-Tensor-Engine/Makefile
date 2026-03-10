CC = gcc
CFLAGS = -Wall -g -O3 -I./include

SRC_DIR = src
INC_DIR = include
TEST_DIR = test

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:.c=.o)

TEST_SRC = $(TEST_DIR)/main.c
TEST_OBJS = $(TEST_SRC:.c=.o)

TARGET = test_tensor

all:$(TARGET)

$(TARGET):$(OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ -lm

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_DIR)/%.o: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean test memcheck

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(TARGET)

test:$(TARGET)
	./$(TARGET)

memcheck:$(TARGET)
	valgrind --leak-check=full ./$(TARGET)

