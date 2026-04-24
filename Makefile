# Makefile — Багатопотоковий спільний лічильник з OpenMP
#
# Збірка:   make
# Запуск:   make run
# Очищення: make clean

CC      = gcc
CFLAGS  = -O2 -fopenmp -Wall -Wextra -std=c11
LDFLAGS = -lpthread -lm

TARGET  = counter_omp
SRC     = counter_omp.c

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "  Збірка завершена: ./$(TARGET)"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) cnt_bench_*.log
