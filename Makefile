# ── Makefile для sort_omp ──────────────────────────────────────────
# Збірка: make
# Запуск: make run
# Запуск з розміром масиву: make run N=20000000
# Очищення: make clean
# ───────────────────────────────────────────────────────────────────

CC      = gcc
CFLAGS  = -O2 -fopenmp -Wall -Wextra -std=c11
LDFLAGS = -lm
TARGET  = sort_omp
SRC     = sort_omp.c

# Розмір масиву за замовчуванням (10 млн)
N ?= 10000000

.PHONY: all clean run run-small perf-check

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)
	@echo "Збірка завершена: ./$(TARGET)"

run: $(TARGET)
	./$(TARGET) $(N)

# Швидкий тест на меншому масиві (1 млн)
run-small: $(TARGET)
	./$(TARGET) 1000000

# Перевірка доступності perf stat
perf-check:
	@which perf > /dev/null 2>&1 && echo "perf: доступний" || echo "perf: недоступний (тільки внутрішня статистика)"

clean:
	rm -f $(TARGET) sort_bench_*.log
