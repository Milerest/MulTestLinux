# Makefile — Багатопотокова обробка тексту з OpenMP
#
# Збірка:   make
# Запуск:   make run
# Очищення: make clean

CC      = gcc
CFLAGS  = -O2 -fopenmp -Wall -Wextra -std=c11
LDFLAGS = -lm -lpthread

TARGET  = text_proc_omp
SRC     = text_proc_omp.c
TEXT    = text_input.txt

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "  Збірка завершена: ./$(TARGET)"

run: $(TARGET)
	./$(TARGET) $(TEXT)
	
clean:
	rm -f $(TARGET) tp_bench_*.log