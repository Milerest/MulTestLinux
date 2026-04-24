/*
 * text_proc_omp.c
 *
 * Багатопотокова обробка тексту з файлу з використанням OpenMP.
 *
 * Завдання: обчислення середньої довжини слова (символів)
 *           та середньої довжини речення (слів у реченні).
 *
 * Три варіанти синхронізації:
 *   1) omp_lock_t (локал. накоп.) — кожен потік накопичує локально,
 *                                   один omp_lock_t для злиття результатів.
 *                                   Мінімальна конкуренція → висока швидкість.
 *   2) omp_lock_t (дрібна блок.) — глобальний omp_lock_t захоплюється
 *                                   на кожне слово. Висока конкуренція → overhead.
 *   3) POSIX sem_t (семафор)     — бінарний POSIX-семафор (sem_wait / sem_post)
 *                                   для злиття локальних результатів.
 *
 * Конфігурації потоків: 1  2  4  6  8
 * Прогонів:             20 на конфігурацію × метод (з очищенням кешу)
 *
 * Perf-подібна статистика (без root/perf):
 *   — середній та мін./макс. wall time, стандартне відхилення, CV%
 *   — час CPU (CLOCK_PROCESS_CPUTIME_ID), ефективність ядер
 *   — добровільні / примусові перемикання контексту (getrusage)
 *   — незначні / значні відмови сторінок (page faults)
 *
 * Логування (українська мова):
 *   Файл tp_bench_РРРРММДД_ГГХВСС.log з інформацією про CPU/кеш,
 *   інфо-рядком методу, детальним perf-рядком та ASCII-діаграмою.
 *
 * Збірка (Linux, GCC):
 *   gcc -O2 -fopenmp -Wall -Wextra -o text_proc_omp text_proc_omp.c -lm -lpthread
 *
 * Запуск:
 *   ./text_proc_omp [файл_тексту]   (за замовчуванням: text_input.txt)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <semaphore.h>
#include <omp.h>

/* ── Константи ────────────────────────────────────────────────────────────── */

#define N_RUNS              20          /* прогонів на кожну конфігурацію       */
#define LOG_PATH_LEN        256
#define BAR_WIDTH           36
#define SPEEDUP_WARN_THR    0.9         /* поріг попередження про слабке прискор.*/
#define CACHE_FLUSH_MB      48          /* МБ для витіснення L3-кешу            */
#define CACHE_FLUSH_SIZE    ((size_t)(CACHE_FLUSH_MB) * 1024 * 1024)

/* Мінімальний розмір тексту в пам'яті для осмисленого бенчмарку.
 * При 200 МБ/с пропускній здатності ≈ 20МБ / 200МБ/с ≈ 100 мс / прогін. */
#define TARGET_TEXT_MB      20
#define TARGET_TEXT_SIZE    ((size_t)(TARGET_TEXT_MB) * 1024 * 1024)

#define DEFAULT_TEXT_FILE   "text_input.txt"
#define MAX_THREADS         16          /* максимум потоків у масивах            */
#define MAX_RESULTS         (5 * 3)     /* 5 конфігурацій × 3 методи            */

/* Фіксовані конфігурації потоків (відповідно до вимог) */
static const int THREAD_CONFIGS[] = {1, 2, 4, 6, 8};
static const int NUM_CONFIGS      = 5;

/* ── Логування ────────────────────────────────────────────────────────────── */

typedef enum { LOG_INFO, LOG_WARN, LOG_ERROR } log_level_t;

static FILE *g_log           = NULL;
static char  g_log_path[LOG_PATH_LEN];

static const char *log_level_str(log_level_t lvl) {
    switch (lvl) {
        case LOG_INFO:  return "INFO ";
        case LOG_WARN:  return "WARN ";
        case LOG_ERROR: return "ERROR";
        default:        return "?????";
    }
}

static void current_timestamp(char *buf, size_t len) {
    time_t t      = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(buf, len, "%Y-%m-%d %H:%M:%S", tm);
}

static void log_write(log_level_t lvl, const char *fmt, ...) {
    if (!g_log) return;
    char ts[32];
    current_timestamp(ts, sizeof(ts));
    fprintf(g_log, "[%s] [%s] ", ts, log_level_str(lvl));
    va_list args;
    va_start(args, fmt);
    vfprintf(g_log, fmt, args);
    va_end(args);
    fprintf(g_log, "\n");
    fflush(g_log);
}

/* ── Інформація про CPU ───────────────────────────────────────────────────── */

static void log_cpu_info(void) {
    log_write(LOG_INFO, "──── Апаратне забезпечення ──────────────────────────────");
    log_write(LOG_INFO, "Логічних ядер (OpenMP): %d", omp_get_max_threads());

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        log_write(LOG_WARN, "Не вдалося відкрити /proc/cpuinfo");
        return;
    }

    char line[256];
    char model[160] = "невідомо";
    char cache[80]  = "невідомо";
    char freq[40]   = "невідомо";
    int  got_model = 0, got_cache = 0, got_freq = 0;

    while (fgets(line, sizeof(line), f)) {
        char *colon = strchr(line, ':');
        if (!colon) continue;
        line[strcspn(line, "\n")] = '\0';
        char *val = colon + 2;
        if (!got_model && strncmp(line, "model name", 10) == 0)
            { strncpy(model, val, sizeof(model) - 1); got_model = 1; }
        if (!got_cache && strncmp(line, "cache size", 10) == 0)
            { strncpy(cache, val, sizeof(cache) - 1); got_cache = 1; }
        if (!got_freq  && strncmp(line, "cpu MHz", 7) == 0) {
            snprintf(freq, sizeof(freq), "%.0f МГц", atof(val));
            got_freq = 1;
        }
        if (got_model && got_cache && got_freq) break;
    }
    fclose(f);

    log_write(LOG_INFO, "Модель CPU:     %s", model);
    log_write(LOG_INFO, "Частота CPU:    %s", freq);
    log_write(LOG_INFO, "Кеш (L2/L3):   %s", cache);
    log_write(LOG_INFO, "Буфер flush:   %d МБ (очищення кешу між прогонами)", CACHE_FLUSH_MB);
    log_write(LOG_INFO, "Прогонів:      %d × кожна конфігурація × кожен метод", N_RUNS);
    log_write(LOG_INFO, "Текст (RAM):   %d МБ", TARGET_TEXT_MB);
    log_write(LOG_INFO, "─────────────────────────────────────────────────────────");
}

/* ── Очищення кешу між прогонами ─────────────────────────────────────────── *
 *                                                                              *
 * Записуємо і зчитуємо буфер більший за L3, щоб витіснити робочі дані        *
 * бенчмарку. Без цього перший прогін "холодний", а решта 19 "теплі",          *
 * що спотворює порівняння. volatile + бар'єр не дають компілятору прибрати.   *
 * ─────────────────────────────────────────────────────────────────────────── */

static char *g_flush_buf = NULL;

static int flush_buf_init(void) {
    g_flush_buf = (char *)malloc(CACHE_FLUSH_SIZE);
    if (!g_flush_buf) return -1;
    memset(g_flush_buf, 0, CACHE_FLUSH_SIZE);
    return 0;
}

static void flush_buf_free(void) { free(g_flush_buf); g_flush_buf = NULL; }

static void flush_cache(void) {
    if (!g_flush_buf) return;
    volatile char *p = (volatile char *)g_flush_buf;
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64) p[i] = (char)(i & 0xFF);
    volatile long acc = 0;
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64) acc += p[i];
    (void)acc;
    __asm__ volatile("" ::: "memory");
}

/* ── Текстовий буфер ──────────────────────────────────────────────────────── */

static char  *g_text      = NULL;
static size_t g_text_size = 0;

/* Зчитує файл та реплікує текст до TARGET_TEXT_SIZE для осмисленого бенчмарку */
static int load_text(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "ERROR: не вдалося відкрити файл '%s'\n", filename);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    if (fsize <= 0) {
        fclose(f);
        fprintf(stderr, "ERROR: файл '%s' порожній або недоступний\n", filename);
        return -1;
    }

    char *raw = (char *)malloc((size_t)fsize + 1);
    if (!raw) { fclose(f); return -1; }

    size_t nread = fread(raw, 1, (size_t)fsize, f);
    fclose(f);
    raw[nread] = '\0';

    /* Якщо файл менший за мінімум — реплікуємо текст */
    size_t target = (nread < TARGET_TEXT_SIZE) ? TARGET_TEXT_SIZE : nread;
    g_text = (char *)malloc(target + 1);
    if (!g_text) { free(raw); return -1; }

    g_text_size = 0;
    while (g_text_size < target) {
        size_t to_copy = nread;
        if (g_text_size + to_copy > target) to_copy = target - g_text_size;
        memcpy(g_text + g_text_size, raw, to_copy);
        g_text_size += to_copy;
    }
    g_text[g_text_size] = '\0';
    free(raw);
    return 0;
}

static void unload_text(void) { free(g_text); g_text = NULL; g_text_size = 0; }

/* ── Межі чанків ──────────────────────────────────────────────────────────── *
 *                                                                              *
 * Ділимо буфер на nthreads приблизно рівних частин. Межу кожного чанку        *
 * зміщуємо до першого не-алфавітного символу, щоб жодне слово не             *
 * перетиналося через дві зони — це виключає подвійний підрахунок.             *
 * ─────────────────────────────────────────────────────────────────────────── */

static void compute_chunk_bounds(int nthreads, size_t starts[], size_t ends[]) {
    size_t base = g_text_size / (size_t)nthreads;
    starts[0] = 0;
    for (int i = 1; i < nthreads; i++) {
        size_t pos = base * (size_t)i;
        /* Переміщуємо позицію до кінця поточного слова (першого не-alpha) */
        while (pos < g_text_size && isalpha((unsigned char)g_text[pos])) pos++;
        starts[i] = pos;
        ends[i - 1] = pos;
    }
    ends[nthreads - 1] = g_text_size;
}

/* ── Підрахунок слів і речень у чанку ────────────────────────────────────── */

static void count_chunk(size_t start, size_t end,
                         long *out_words, long *out_chars, long *out_sents) {
    long words = 0, chars = 0, sents = 0;
    int  in_word = 0;
    long wlen    = 0;

    for (size_t i = start; i < end; i++) {
        unsigned char c = (unsigned char)g_text[i];
        if (isalpha(c)) {
            if (!in_word) { in_word = 1; wlen = 0; }
            wlen++;
        } else {
            if (in_word) { words++; chars += wlen; in_word = 0; }
            if (c == '.' || c == '!' || c == '?') sents++;
        }
    }
    if (in_word) { words++; chars += wlen; }

    *out_words = words;
    *out_chars = chars;
    *out_sents = sents;
}

/* ── Результат обробки тексту ─────────────────────────────────────────────── */

typedef struct {
    long   words;
    long   chars;
    long   sentences;
    double avg_word_len;   /* символів / слово    */
    double avg_sent_len;   /* слів / речення      */
} text_result_t;

static text_result_t make_result(long w, long c, long s) {
    text_result_t r;
    r.words        = w;
    r.chars        = c;
    r.sentences    = s;
    r.avg_word_len = (w > 0) ? (double)c / w : 0.0;
    r.avg_sent_len = (s > 0) ? (double)w / s : 0.0;
    return r;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  МЕТОДИ СИНХРОНІЗАЦІЇ
 * ══════════════════════════════════════════════════════════════════════════════ */

/* ── Метод 1: omp_lock_t — локальне накопичення + одне злиття ─────────────── *
 *                                                                              *
 * Кожен потік обробляє свій чанк цілком (count_chunk) і накопичує локально.   *
 * Після завершення чанку захоплює один глобальний omp_lock_t, додає локальну  *
 * суму до глобальних лічильників і відпускає lock. Один lock / потік —        *
 * мінімальна конкуренція → очікується практично лінійне прискорення.          *
 * ─────────────────────────────────────────────────────────────────────────── */

static text_result_t tp_local_mutex(int nthreads) {
    size_t starts[MAX_THREADS], ends[MAX_THREADS];
    compute_chunk_bounds(nthreads, starts, ends);

    long total_words = 0, total_chars = 0, total_sents = 0;

    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel num_threads(nthreads)
    {
        int  tid = omp_get_thread_num();
        long lw, lc, ls;
        count_chunk(starts[tid], ends[tid], &lw, &lc, &ls);

        /* Одне захоплення lock на потік — злиття локальних результатів */
        omp_set_lock(&lock);
        total_words += lw;
        total_chars += lc;
        total_sents += ls;
        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);
    return make_result(total_words, total_chars, total_sents);
}

/* ── Метод 2: omp_lock_t — mutex на кожне слово (висока конкуренція) ──────── *
 *                                                                              *
 * Глобальні лічильники захищені єдиним omp_lock_t, який захоплюється при      *
 * кожному знайденому слові та реченні. З ростом кількості потоків конкуренція  *
 * за lock зростає — демонструє overhead від дрібнозернистого locking.          *
 * ─────────────────────────────────────────────────────────────────────────── */

static text_result_t tp_fine_mutex(int nthreads) {
    size_t starts[MAX_THREADS], ends[MAX_THREADS];
    compute_chunk_bounds(nthreads, starts, ends);

    long total_words = 0, total_chars = 0, total_sents = 0;

    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel num_threads(nthreads)
    {
        int  tid    = omp_get_thread_num();
        int  in_word = 0;
        long wlen    = 0;

        for (size_t i = starts[tid]; i < ends[tid]; i++) {
            unsigned char c = (unsigned char)g_text[i];
            if (isalpha(c)) {
                if (!in_word) { in_word = 1; wlen = 0; }
                wlen++;
            } else {
                if (in_word) {
                    /* lock на кожне слово — основне джерело конкуренції */
                    omp_set_lock(&lock);
                    total_words++;
                    total_chars += wlen;
                    omp_unset_lock(&lock);
                    in_word = 0;
                }
                if (c == '.' || c == '!' || c == '?') {
                    omp_set_lock(&lock);
                    total_sents++;
                    omp_unset_lock(&lock);
                }
            }
        }
        /* Слово на межі чанку */
        if (in_word) {
            omp_set_lock(&lock);
            total_words++;
            total_chars += wlen;
            omp_unset_lock(&lock);
        }
    }

    omp_destroy_lock(&lock);
    return make_result(total_words, total_chars, total_sents);
}

/* ── Метод 3: POSIX sem_t — бінарний семафор ─────────────────────────────── *
 *                                                                              *
 * Кожен потік обробляє свій чанк локально (count_chunk), після чого для       *
 * злиття використовує POSIX sem_wait / sem_post. Семафор ініціалізовано зі    *
 * значенням 1 (бінарний, еквівалент mutex). Демонструє стандартний API POSIX. *
 * ─────────────────────────────────────────────────────────────────────────── */

static text_result_t tp_semaphore(int nthreads) {
    size_t starts[MAX_THREADS], ends[MAX_THREADS];
    compute_chunk_bounds(nthreads, starts, ends);

    long total_words = 0, total_chars = 0, total_sents = 0;

    sem_t sem;
    sem_init(&sem, 0, 1); /* pshared=0 (всередині процесу), value=1 (бінарний) */

    #pragma omp parallel num_threads(nthreads)
    {
        int  tid = omp_get_thread_num();
        long lw, lc, ls;
        count_chunk(starts[tid], ends[tid], &lw, &lc, &ls);

        /* Злиття через POSIX семафор — один sem_wait / потік */
        sem_wait(&sem);
        total_words += lw;
        total_chars += lc;
        total_sents += ls;
        sem_post(&sem);
    }

    sem_destroy(&sem);
    return make_result(total_words, total_chars, total_sents);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  СТРУКТУРИ СТАТИСТИКИ
 * ══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double       wall_time; /* реальний час (с)                        */
    double       cpu_time;  /* час CPU (CLOCK_PROCESS_CPUTIME_ID)      */
    long         nvcsw;     /* добровільні перемикання контексту        */
    long         nivcsw;    /* примусові перемикання контексту          */
    long         minflt;    /* незначні відмови сторінок                */
    long         majflt;    /* значні відмови сторінок                  */
    text_result_t result;
} run_stat_t;

typedef struct {
    double mean_wall;       /* середній wall time                      */
    double stddev_wall;     /* стандартне відхилення                   */
    double min_wall;        /* мінімум                                 */
    double max_wall;        /* максимум                                 */
    double cv_wall;         /* коефіцієнт варіації, %                  */
    double mean_cpu;        /* середній час CPU                        */
    double cpu_eff;         /* ефективність ядер, %                    */
    double mean_speedup;    /* середнє прискорення                     */
    double mean_word_len;   /* середня довжина слова                   */
    double mean_sent_len;   /* середня довжина речення                 */
    long   total_nvcsw;
    long   total_nivcsw;
    long   total_minflt;
    long   total_majflt;
} agg_stat_t;

/* ── Зведений результат для ASCII-діаграми ─────────────────────────────────── */

typedef struct {
    char   method[80];
    int    threads;
    double elapsed;
    double speedup;
} bench_result_t;

static bench_result_t g_results[MAX_RESULTS];
static int            g_result_count = 0;

/* ══════════════════════════════════════════════════════════════════════════════
 *  ASCII-ДІАГРАМА ПРИСКОРЕННЯ
 * ══════════════════════════════════════════════════════════════════════════════ */

static void build_bar(char *buf, int buf_size, int bar_len) {
    const char BLOCK[] = "\xe2\x96\x88"; /* UTF-8: ▉ (U+2588 FULL BLOCK) */
    const int  BLEN    = 3;
    int pos = 0;
    for (int i = 0; i < BAR_WIDTH && pos + BLEN < buf_size; i++) {
        if (i < bar_len) { memcpy(buf + pos, BLOCK, BLEN); pos += BLEN; }
        else              { buf[pos++] = ' '; }
    }
    buf[pos] = '\0';
}

static void render_chart(FILE *out) {
    if (g_result_count == 0) return;

    double max_sp = 1.0;
    for (int i = 0; i < g_result_count; i++)
        if (g_results[i].speedup > max_sp) max_sp = g_results[i].speedup;

    char sep[72]; memset(sep, '-', 70); sep[70] = '\0';
    fprintf(out, "\n%s\n", sep);
    fprintf(out, "  ДІАГРАМА ПРИСКОРЕННЯ  (1 \xe2\x96\x88 = %.2fx)\n",
            max_sp / BAR_WIDTH);
    fprintf(out, "%s\n", sep);

    char prev_method[80] = "";
    char bar_buf[BAR_WIDTH * 3 + 4];

    for (int i = 0; i < g_result_count; i++) {
        bench_result_t *r = &g_results[i];
        if (strcmp(r->method, prev_method) != 0) {
            snprintf(prev_method, sizeof(prev_method), "%s", r->method);
            fprintf(out, "\n  [%s]\n", r->method);
        }
        int bar_len = (int)((r->speedup / max_sp) * BAR_WIDTH);
        if (bar_len < 1) bar_len = 1;
        build_bar(bar_buf, sizeof(bar_buf), bar_len);
        fprintf(out, "  %2d п.  [%s]  %.2fx\n", r->threads, bar_buf, r->speedup);
    }
    fprintf(out, "\n%s\n", sep);
}

static void print_chart(void) { render_chart(stdout); }

static void log_chart(void) {
    if (!g_log) return;
    log_write(LOG_INFO, "──── ASCII-діаграма прискорення ────");
    render_chart(g_log);
}

/* ── Відкриття / закриття лог-файлу ──────────────────────────────────────── */

static int log_open(const char *filename, size_t text_size) {
    time_t t      = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(g_log_path, sizeof(g_log_path), "tp_bench_%Y%m%d_%H%M%S.log", tm);
    g_log = fopen(g_log_path, "w");
    if (!g_log) {
        fprintf(stderr, "WARN: не вдалося створити лог '%s'\n", g_log_path);
        return -1;
    }
    char ts[32];
    current_timestamp(ts, sizeof(ts));
    fprintf(g_log,
            "\xE2\x95\x94\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x97\n"
            "\xE2\x95\x91  \xD0\x9E\xD0\xB1\xD1\x80\xD0\xBE\xD0\xB1\xD0\xBA"
            "\xD0\xB0 \xD1\x82\xD0\xB5\xD0\xBA\xD1\x81\xD1\x82\xD1\x83 OpenMP"
            " \xD0\xB1\xD0\xB5\xD0\xBD\xD1\x87\xD0\xBC\xD0\xB0\xD1\x80\xD0"
            "\xBA \xE2\x80\x94 \xD1\x81\xD0\xB5\xD1\x81\xD1\x96\xD1\x8F %s  "
            "\xE2\x95\x91\n"
            "\xE2\x95\x9A\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x9D\n\n",
            ts);
    fflush(g_log);
    log_write(LOG_INFO, "Запуск програми");
    log_write(LOG_INFO, "Вхідний файл:   %s", filename);
    log_write(LOG_INFO, "Розмір тексту:  %.2f МБ (%.0f символів)",
              (double)text_size / (1024 * 1024), (double)text_size);
    log_write(LOG_INFO, "Прогонів:       %d", N_RUNS);
    return 0;
}

static void log_close(void) {
    if (!g_log) return;
    log_write(LOG_INFO, "Програму завершено успішно");
    fprintf(g_log, "\n\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90 EOF "
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90"
            "\xE2\x95\x90\n");
    fclose(g_log);
    g_log = NULL;
}

/* ── Допоміжні функції виводу ─────────────────────────────────────────────── */

static void print_separator(void) {
    printf("\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80"
           "\n");
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  ФУНКЦІЯ ЗАПУСКУ БЕНЧМАРКУ
 * ══════════════════════════════════════════════════════════════════════════════ */

typedef text_result_t (*tp_func_t)(int);

static void run_benchmark(const char *name, tp_func_t func,
                           double baseline_time,
                           const char *method_info) {

    /* Інфо-рядок методу у лог */
    log_write(LOG_INFO, "════════════════════════════════════════════════════════");
    log_write(LOG_INFO, "МЕТОД: %s", name);
    log_write(LOG_INFO, "ОПИС:  %s", method_info);
    log_write(LOG_INFO, "════════════════════════════════════════════════════════");

    printf("\n  Метод: %s\n", name);
    printf("  %-6s  %-8s  %-9s  %-9s  %-9s  %-9s  %-8s  %-6s  %-5s\n",
           "Потоки", "Сер.сл.", "Сер.реч.",
           "Сер.час с", "Відхил. с", "Min с",
           "Прискор.", "CV%", "CtxSw");
    printf("  %-6s  %-8s  %-9s  %-9s  %-9s  %-9s  %-8s  %-6s  %-5s\n",
           "──────", "────────", "─────────",
           "─────────", "─────────", "─────",
           "────────", "────", "─────");

    char log_line[640];
    int  log_pos  = 0;
    int  has_warn = 0;

    run_stat_t runs[N_RUNS];

    for (int ci = 0; ci < NUM_CONFIGS; ci++) {
        int t = THREAD_CONFIGS[ci];

        printf("  [%2dп] прогрес: ", t);
        fflush(stdout);

        for (int r = 0; r < N_RUNS; r++) {
            /* ── Очищення кешу ── */
            flush_cache();

            /* ── Стан до ── */
            struct rusage   ru0;
            struct timespec cpu0;
            getrusage(RUSAGE_SELF, &ru0);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu0);

            /* ── Власне обчислення ── */
            double t_start = omp_get_wtime();
            text_result_t res = func(t);
            double elapsed    = omp_get_wtime() - t_start;

            /* ── Стан після ── */
            struct rusage   ru1;
            struct timespec cpu1;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu1);
            getrusage(RUSAGE_SELF, &ru1);

            double cpu_sec = (cpu1.tv_sec  - cpu0.tv_sec) +
                             (cpu1.tv_nsec - cpu0.tv_nsec) * 1e-9;

            runs[r].wall_time = elapsed;
            runs[r].cpu_time  = cpu_sec;
            runs[r].result    = res;
            runs[r].nvcsw     = ru1.ru_nvcsw  - ru0.ru_nvcsw;
            runs[r].nivcsw    = ru1.ru_nivcsw - ru0.ru_nivcsw;
            runs[r].minflt    = ru1.ru_minflt - ru0.ru_minflt;
            runs[r].majflt    = ru1.ru_majflt - ru0.ru_majflt;

            printf("."); fflush(stdout);
        }
        printf(" OK\n");

        /* ── Агрегування ── */
        agg_stat_t agg;
        memset(&agg, 0, sizeof(agg));
        agg.min_wall = runs[0].wall_time;
        agg.max_wall = runs[0].wall_time;

        for (int r = 0; r < N_RUNS; r++) {
            agg.mean_wall     += runs[r].wall_time;
            agg.mean_cpu      += runs[r].cpu_time;
            agg.mean_word_len += runs[r].result.avg_word_len;
            agg.mean_sent_len += runs[r].result.avg_sent_len;
            agg.total_nvcsw   += runs[r].nvcsw;
            agg.total_nivcsw  += runs[r].nivcsw;
            agg.total_minflt  += runs[r].minflt;
            agg.total_majflt  += runs[r].majflt;
            if (runs[r].wall_time < agg.min_wall) agg.min_wall = runs[r].wall_time;
            if (runs[r].wall_time > agg.max_wall) agg.max_wall = runs[r].wall_time;
        }
        agg.mean_wall     /= N_RUNS;
        agg.mean_cpu      /= N_RUNS;
        agg.mean_word_len /= N_RUNS;
        agg.mean_sent_len /= N_RUNS;

        for (int r = 0; r < N_RUNS; r++) {
            double d = runs[r].wall_time - agg.mean_wall;
            agg.stddev_wall += d * d;
        }
        agg.stddev_wall  = sqrt(agg.stddev_wall / N_RUNS);
        agg.cv_wall      = (agg.mean_wall > 0.0)
                           ? (agg.stddev_wall / agg.mean_wall * 100.0) : 0.0;
        agg.mean_speedup = (baseline_time > 0.0)
                           ? (baseline_time / agg.mean_wall) : 1.0;
        agg.cpu_eff      = (agg.mean_wall > 0.0 && t > 0)
                           ? (agg.mean_cpu / (agg.mean_wall * t) * 100.0) : 0.0;

        /* ── Таблиця в консоль ── */
        printf("  %-6d  %-8.3f  %-9.2f  %-9.4f  %-9.4f  %-9.4f  %-8.3f  %-6.2f  %ld/%ld\n",
               t,
               agg.mean_word_len, agg.mean_sent_len,
               agg.mean_wall, agg.stddev_wall, agg.min_wall,
               agg.mean_speedup, agg.cv_wall,
               agg.total_nvcsw, agg.total_nivcsw);

        /* ── Детальний perf-рядок у лог ── */
        log_write(LOG_INFO,
                  "[%s | %2dп]  сл.=%.3f симв  реч.=%.2f сл  "
                  "час=%.4f\xC2\xB1%.4f\xD1\x81 [%.4f\xE2\x80\x93%.4f]  "
                  "прискор=%.3fx  CPU-ефект=%.1f%%  CV=%.2f%%  "
                  "ctx_sw=%ld/%ld  page_faults=%ld+%ld",
                  name, t,
                  agg.mean_word_len, agg.mean_sent_len,
                  agg.mean_wall, agg.stddev_wall,
                  agg.min_wall, agg.max_wall,
                  agg.mean_speedup, agg.cpu_eff, agg.cv_wall,
                  agg.total_nvcsw, agg.total_nivcsw,
                  agg.total_minflt, agg.total_majflt);

        /* ── Компактний підсумок для методу ── */
        log_pos += snprintf(log_line + log_pos,
                            sizeof(log_line) - (size_t)log_pos,
                            "%d\xD0\xBF=%.3fx(\xC2\xB1%.4f\xD1\x81,CV=%.1f%%)",
                            t, agg.mean_speedup, agg.stddev_wall, agg.cv_wall);
        if (ci < NUM_CONFIGS - 1)
            log_pos += snprintf(log_line + log_pos,
                                sizeof(log_line) - (size_t)log_pos, " | ");

        if (t >= 2 && agg.mean_speedup < SPEEDUP_WARN_THR) has_warn = 1;

        /* ── Для діаграми ── */
        if (g_result_count < MAX_RESULTS) {
            bench_result_t *br = &g_results[g_result_count++];
            strncpy(br->method, name, sizeof(br->method) - 1);
            br->method[sizeof(br->method) - 1] = '\0';
            br->threads = t;
            br->elapsed = agg.mean_wall;
            br->speedup = agg.mean_speedup;
        }
    }

    /* ── Один підсумковий рядок на метод ── */
    log_write(LOG_INFO,
              "\xD0\x9F\xD0\x86\xD0\x94\xD0\xA1\xD0\xA3\xD0\x9C\xD0\x9E\xD0"
              "\x9A %-30s \xE2\x86\x92 %s", name, log_line);
    if (has_warn)
        log_write(LOG_WARN,
                  "%-30s \xE2\x86\x92 \xD0\xBF\xD1\x80\xD0\xB8\xD1\x81\xD0"
                  "\xBA\xD0\xBE\xD1\x80\xD0\xB5\xD0\xBD\xD0\xBD\xD1\x8F < "
                  "%.1f \xD0\xBF\xD1\x80\xD0\xB8 \xE2\x89\xA5" "2 \xD0\xBF\xD0"
                  "\xBE\xD1\x82\xD0\xBE\xD0\xBA\xD0\xB0\xD1\x85: "
                  "\xD0\xB2\xD0\xB8\xD1\x81\xD0\xBE\xD0\xBA\xD0\xB0 "
                  "\xD0\xBA\xD0\xBE\xD0\xBD\xD0\xBA\xD1\x83\xD1\x80\xD0"
                  "\xB5\xD0\xBD\xD1\x86\xD1\x96\xD1\x8F \xD0\xB7\xD0\xB0 "
                  "lock",
                  name, SPEEDUP_WARN_THR);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  ГОЛОВНА ФУНКЦІЯ
 * ══════════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    const char *filename = (argc >= 2) ? argv[1] : DEFAULT_TEXT_FILE;

    /* Завантаження тексту */
    printf("\n  Завантаження тексту з '%s'...\n", filename);
    if (load_text(filename) != 0) return 1;
    printf("  Буфер у пам'яті: %.2f МБ (%zu символів)\n",
           (double)g_text_size / (1024.0 * 1024.0), g_text_size);

    /* Відкриття лога */
    if (log_open(filename, g_text_size) == 0)
        printf("  Лог-файл: %s\n", g_log_path);
    log_cpu_info();

    /* Ініціалізація буфера очищення кешу */
    if (flush_buf_init() != 0) {
        fprintf(stderr, "WARN: не вдалося виділити %d МБ для очищення кешу\n",
                CACHE_FLUSH_MB);
        log_write(LOG_WARN, "Буфер очищення кешу недоступний — "
                  "результати можуть бути завищені");
    } else {
        log_write(LOG_INFO, "Буфер очищення кешу виділено: %d МБ", CACHE_FLUSH_MB);
    }

    /* Заголовок */
    printf("\n");
    print_separator();
    printf("  Багатопотокова обробка тексту  \xE2\x80\x94  OpenMP бенчмарк\n");
    print_separator();
    printf("  Файл:           %s\n", filename);
    printf("  Текст (RAM):    %.2f МБ\n", (double)g_text_size / (1024.0 * 1024.0));
    printf("  Прогонів:       %d (з очищенням кешу %d МБ перед кожним)\n",
           N_RUNS, CACHE_FLUSH_MB);
    printf("  Конфігурації:   ");
    for (int i = 0; i < NUM_CONFIGS; i++)
        printf("%d%s", THREAD_CONFIGS[i], i < NUM_CONFIGS - 1 ? " " : "\n");
    print_separator();

    /* Базовий час — 1 потік, метод 1, N_RUNS прогонів з flush */
    printf("\n  Вимірювання базового часу (1 потік, omp_lock локал., %d прогонів)...\n",
           N_RUNS);
    printf("  прогрес: ");
    double baseline_times[N_RUNS];
    for (int r = 0; r < N_RUNS; r++) {
        flush_cache();
        double ts = omp_get_wtime();
        tp_local_mutex(1);
        baseline_times[r] = omp_get_wtime() - ts;
        printf("."); fflush(stdout);
    }
    printf(" OK\n");

    double baseline = 0.0;
    for (int r = 0; r < N_RUNS; r++) baseline += baseline_times[r];
    baseline /= N_RUNS;
    printf("  Базовий час (середній): %.4f с\n", baseline);
    log_write(LOG_INFO, "Базовий час (1 потік, N=%d): %.4f с", N_RUNS, baseline);

    /* Контрольний результат (перевірка коректності) */
    {
        text_result_t ref = tp_local_mutex(1);
        printf("\n  Контрольні результати (1 потік):\n");
        printf("    Слів:                    %ld\n",  ref.words);
        printf("    Символів у словах:       %ld\n",  ref.chars);
        printf("    Речень:                  %ld\n",  ref.sentences);
        printf("    Серед. довжина слова:    %.3f символів\n", ref.avg_word_len);
        printf("    Серед. довжина речення:  %.2f слів\n",     ref.avg_sent_len);
        log_write(LOG_INFO,
                  "Контрольний результат: слів=%ld, симв=%ld, речень=%ld, "
                  "сер.слово=%.3f, сер.речення=%.2f",
                  ref.words, ref.chars, ref.sentences,
                  ref.avg_word_len, ref.avg_sent_len);
    }

    /* ── Бенчмарк трьох методів ── */
    run_benchmark(
        "omp_lock (локал.накоп.+mutex-злиття)",
        tp_local_mutex, baseline,
        "Кожен потік накопичує локально, один omp_lock_t для злиття. "
        "Один lock/потік — мінімальна конкуренція. Очікується лінійне прискорення.");

    run_benchmark(
        "omp_lock (mutex на кожне слово)",
        tp_fine_mutex, baseline,
        "Глобальний omp_lock_t захоплюється при кожному знайденому слові. "
        "Висока конкуренція зростає з кількістю потоків — демонструє lock overhead.");

    run_benchmark(
        "POSIX sem_t (бінарний семафор)",
        tp_semaphore, baseline,
        "Кожен потік накопичує локально, злиття через sem_wait/sem_post. "
        "Один sem_wait/потік — поведінка аналогічна методу 1 через POSIX API.");

    /* Діаграма */
    print_chart();
    log_chart();

    /* Пояснення */
    printf("\n");
    print_separator();
    printf("  Пояснення методів синхронізації:\n");
    printf("  \xE2\x80\xA2 omp_lock (локал.) \xE2\x80\x94 кожен потік обробляє"
           " свій чанк цілком локально;\n");
    printf("                      один omp_lock_t захоплюється лише для злиття."
           "\n");
    printf("                      Мін. конкуренція \xE2\x86\x92 прискорення"
           " близьке до лінійного.\n");
    printf("  \xE2\x80\xA2 omp_lock (дрібний) \xE2\x80\x94 lock захоплюється"
           " на кожне слово у циклі;\n");
    printf("                      при N потоках N-1 чекають \xE2\x86\x92"
           " висока серіалізація.\n");
    printf("  \xE2\x80\xA2 POSIX sem_t \xE2\x80\x94 POSIX sem_init/sem_wait/sem_post;"
           "\n");
    printf("                      один sem_wait/потік, як метод 1, але через"
           " POSIX API.\n");
    printf("  \xE2\x80\xA2 CV%% \xE2\x80\x94 коефіцієнт варіації:"
           " стабільність між прогонами.\n");
    printf("  \xE2\x80\xA2 CtxSw \xE2\x80\x94 перемикання контексту"
           " вол./прим. за %d прогонів.\n", N_RUNS);
    print_separator();
    printf("\n");

    unload_text();
    flush_buf_free();
    log_close();
    return 0;
}
