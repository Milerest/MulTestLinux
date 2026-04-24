# Makefile для monte_carlo_omp
# Требования: GCC с поддержкой OpenMP (пакет libgomp, входит в gcc по умолчанию)
#
# Использование:
#   make            — собрать с оптимизацией -O2
#   make debug      — собрать с отладочной информацией
#   make run        — собрать и запустить (100 млн выборок)
#   make run N=50000000  — запустить с 50 млн выборок
#   make clean      — удалить собранный файл

CC      = gcc
TARGET  = monte_carlo_omp
SRC     = monte_carlo_omp.c
N      ?= 100000000

CFLAGS_COMMON = -Wall -Wextra -fopenmp
CFLAGS_RELEASE = $(CFLAGS_COMMON) -O2 -march=native
CFLAGS_DEBUG   = $(CFLAGS_COMMON) -O0 -g3 -fsanitize=thread
LDFLAGS = -lm

# ── Цели ──────────────────────────────────────────────────────────────────

.PHONY: all debug run clean info

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS_RELEASE) -o $@ $< $(LDFLAGS)
	@echo "✓  Собрано: $(TARGET)"

debug: $(SRC)
	$(CC) $(CFLAGS_DEBUG) -o $(TARGET)_debug $< $(LDFLAGS)
	@echo "✓  Отладочная сборка: $(TARGET)_debug"

run: $(TARGET)
	@echo "Запуск с N=$(N) выборками..."
	./$(TARGET) $(N)

clean:
	rm -f $(TARGET) $(TARGET)_debug

info:
	@echo "Компилятор: $(CC) $$($(CC) --version | head -1)"
	@echo "Ядра CPU:   $$(nproc)"
