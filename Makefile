# Makefile — Черга «Виробник–Споживач» (OpenMP бенчмарк)
#
# Збірка:   make
# Запуск:   ./prod_cons_omp
#           ./prod_cons_omp 5000000   (5 млн елементів)
# Відлагод: make debug
# Очищення: make clean

CC      = gcc
TARGET  = prod_cons_omp
SRC     = prod_cons_omp.c

# -D_GNU_SOURCE: POSIX-розширення (sem_t, CLOCK_PROCESS_CPUTIME_ID, getrusage)
CFLAGS  = -O2 -fopenmp -Wall -Wextra -D_GNU_SOURCE
LDFLAGS = -lm

.PHONY: all debug clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Збірка завершена: ./$(TARGET)"

debug: $(SRC)
	$(CC) -O0 -g -fopenmp -Wall -Wextra -D_GNU_SOURCE \
	      -fsanitize=thread -o $(TARGET)_debug $< $(LDFLAGS)
	@echo "Debug збірка: ./$(TARGET)_debug"

clean:
	rm -f $(TARGET) $(TARGET)_debug *.log
	@echo "Очищено"
