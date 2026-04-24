/*
 * counter_omp.c
 *
 * Багатопотоковий спільний лічильник з використанням OpenMP.
 *
 * Задача: N_OPS атомарних інкрементів спільного лічильника розподіляються
 * між усіма потоками. Перевіряємо, що фінальне значення = N_OPS.
 *
 * Три варіанти синхронізації:
 *   1) omp_lock    — OpenMP-мʼютекс (omp_lock_t): кожен потік захоплює
 *                    лок, інкрементує глобальний лічильник, звільняє.
 *   2) semaphore   — POSIX binary semaphore (sem_t, init=1):
 *                    P-операція (sem_wait) → інкремент → V-операція (sem_post).
 *                    Семафор з лічильником 1 = мʼютекс, але з явним
 *                    інтерфейсом «ресурс доступний/зайнятий».
 *   3) local_array — кожен потік пише лише у свій слот вирівняного масиву
 *                    (64-байтовий padding → нема false sharing); фінальна
 *                    сума — один послідовний прохід без блокувань.
 *
 * Бенчмарк:
 *   Конфігурації потоків: 1 2 4 6 8
 *   N_RUNS=20 прогонів на кожну конфігурацію.
 *   Перед кожним прогоном кеш очищується великим буфером (CACHE_FLUSH_MB),
 *   щоб результати не залежали від «теплого» кешу.
 *
 * Perf-подібна статистика (без root/perf):
 *   — середній і мінімальний/максимальний wall-time
 *   — стандартне відхилення і CV%
 *   — CPU-time (CLOCK_PROCESS_CPUTIME_ID) і ефективність ядер
 *   — перемикання контексту (getrusage)
 *   — page faults (getrusage)
 *
 * Логування:
 *   Файл cnt_bench_РРРРММДД_ГГХВСС.log (UTF-8, українська мова):
 *     • блок конфігурації ЦП
 *     • інформаційний рядок на кожен метод
 *     • детальний рядок perf на кожну конфігурацію потоків
 *     • підсумковий рядок на метод
 *     • ASCII-діаграма прискорення
 *
 * Збірка (Linux, GCC):
 *   gcc -O2 -fopenmp -Wall -o counter_omp counter_omp.c -lpthread
 *
 * Запуск:
 *   ./counter_omp [кількість_операцій]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <semaphore.h>
#include <omp.h>

/* ── Константи ────────────────────────────────────────────────────────────── */

#define DEFAULT_OPS         10000000L   /* 10 млн операцій за замовчуванням   */
#define N_RUNS              20          /* прогонів на кожну конфігурацію      */
#define LOG_PATH_LEN        256
#define BAR_WIDTH           36
#define SPEEDUP_WARN_THR    0.9

/* Розмір буфера для очищення кешу: має перевищувати L3-кеш процесора.
 * 48 МБ покривають більшість сучасних десктопних CPU (L3 ≤ 32 МБ).    */
#define CACHE_FLUSH_MB      48
#define CACHE_FLUSH_SIZE    ((size_t)(CACHE_FLUSH_MB) * 1024 * 1024)

/* Розмір рядка кешу для вирівнювання локального масиву лічильника */
#define CACHE_LINE_SIZE     64

/* Фіксовані конфігурації потоків, як вимагає завдання */
static const int THREAD_CONFIGS[] = {1, 2, 4, 6, 8};
static const int NUM_CONFIGS      = 5;

/* Кількість методів × конфігурацій для масиву результатів */
#define MAX_METHODS     3
#define MAX_RESULTS     (MAX_METHODS * 5)

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

/* ── Інформація про процесор і кеш ───────────────────────────────────────── */

static void log_cpu_info(void) {
    log_write(LOG_INFO, "──── Апаратне забезпечення ────────────────────────────");
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
    int  got_model  = 0, got_cache = 0, got_freq = 0;

    while (fgets(line, sizeof(line), f)) {
        char *colon = strchr(line, ':');
        if (!colon) continue;
        line[strcspn(line, "\n")] = '\0';
        char *val = colon + 2;

        if (!got_model && strncmp(line, "model name", 10) == 0)
            { strncpy(model, val, sizeof(model)-1); got_model = 1; }
        if (!got_cache && strncmp(line, "cache size", 10) == 0)
            { strncpy(cache, val, sizeof(cache)-1); got_cache = 1; }
        if (!got_freq  && strncmp(line, "cpu MHz", 7) == 0) {
            snprintf(freq, sizeof(freq), "%.0f МГц", atof(val));
            got_freq = 1;
        }
        if (got_model && got_cache && got_freq) break;
    }
    fclose(f);

    log_write(LOG_INFO, "Модель CPU:     %s", model);
    log_write(LOG_INFO, "Частота CPU:    %s", freq);
    log_write(LOG_INFO, "Кеш (L2/L3):    %s", cache);
    log_write(LOG_INFO, "Буфер flush:    %d МБ (очищення кешу між прогонами)", CACHE_FLUSH_MB);
    log_write(LOG_INFO, "Прогонів:       %d × кожна конфігурація", N_RUNS);
    log_write(LOG_INFO, "Конфігурації:   1 2 4 6 8 потоків");
    log_write(LOG_INFO, "───────────────────────────────────────────────────────");
}

/* ── Очищення кешу між прогонами ─────────────────────────────────────────── */

static char *g_flush_buf = NULL;

static int flush_buf_init(void) {
    g_flush_buf = (char *)malloc(CACHE_FLUSH_SIZE);
    if (!g_flush_buf) return -1;
    memset(g_flush_buf, 0, CACHE_FLUSH_SIZE);
    return 0;
}

static void flush_buf_free(void) {
    free(g_flush_buf);
    g_flush_buf = NULL;
}

static void flush_cache(void) {
    if (!g_flush_buf) return;
    volatile char *p = (volatile char *)g_flush_buf;
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64)
        p[i] = (char)(i & 0xFF);
    volatile long acc = 0;
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64)
        acc += p[i];
    (void)acc;
    __asm__ volatile("" ::: "memory");
}

/* ── Структури статистики ─────────────────────────────────────────────────── */

typedef struct {
    double wall_time;
    double cpu_time;
    long   final_val;   /* фінальне значення лічильника (перевірка коректності) */
    long   nvcsw;
    long   nivcsw;
    long   minflt;
    long   majflt;
} run_stat_t;

typedef struct {
    double mean_wall;
    double stddev_wall;
    double min_wall;
    double max_wall;
    double cv_wall;
    double mean_cpu;
    double cpu_eff;
    double mean_speedup;
    long   total_nvcsw;
    long   total_nivcsw;
    long   total_minflt;
    long   total_majflt;
    int    errors;       /* кількість прогонів із некоректним результатом */
} agg_stat_t;

typedef struct {
    char   method[80];
    int    threads;
    double elapsed;
    double speedup;
} bench_result_t;

static bench_result_t g_results[MAX_RESULTS];
static int            g_result_count = 0;

/* ── ASCII-візуалізатор прискорення ──────────────────────────────────────── */

static void build_bar(char *buf, int buf_size, int bar_len) {
    const char BLOCK[] = "\xe2\x96\x88";
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
    fprintf(out, "  ДІАГРАМА ПРИСКОРЕННЯ  (1 \xe2\x96\x88 = %.3fx)\n", max_sp / BAR_WIDTH);
    fprintf(out, "%s\n", sep);

    char prev_method[80] = "";
    char bar_buf[BAR_WIDTH * 3 + 4];
    for (int i = 0; i < g_result_count; i++) {
        bench_result_t *r = &g_results[i];
        if (strcmp(r->method, prev_method) != 0) {
            strncpy(prev_method, r->method, sizeof(prev_method) - 1);
            fprintf(out, "\n  [%s]\n", r->method);
        }
        int bar_len = (int)((r->speedup / max_sp) * BAR_WIDTH);
        if (bar_len < 1) bar_len = 1;
        build_bar(bar_buf, sizeof(bar_buf), bar_len);
        fprintf(out, "  %2d п.  [%s]  %.3fx\n", r->threads, bar_buf, r->speedup);
    }
    fprintf(out, "\n%s\n", sep);
}

static void print_chart(void) { render_chart(stdout); }

static void log_chart(void) {
    if (!g_log) return;
    log_write(LOG_INFO, "──── ASCII-діаграма прискорення ────");
    render_chart(g_log);
}

/* ── Відкриття/закриття лог-файлу ────────────────────────────────────────── */

static int log_open(long n, int max_omp_threads) {
    time_t t      = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(g_log_path, sizeof(g_log_path), "cnt_bench_%Y%m%d_%H%M%S.log", tm);
    g_log = fopen(g_log_path, "w");
    if (!g_log) {
        fprintf(stderr, "WARN: не вдалося створити лог-файл '%s'\n", g_log_path);
        return -1;
    }
    char ts[32];
    current_timestamp(ts, sizeof(ts));
    fprintf(g_log,
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  Спільний лічильник OpenMP — сесія %s    ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n",
            ts);
    fflush(g_log);
    log_write(LOG_INFO, "Запуск програми");
    log_write(LOG_INFO, "Операцій:       %ld", n);
    log_write(LOG_INFO, "Макс. потоків:  %d (доступно системі)", max_omp_threads);
    log_write(LOG_INFO, "Прогонів:       %d", N_RUNS);
    return 0;
}

static void log_close(void) {
    if (!g_log) return;
    log_write(LOG_INFO, "Програму завершено успішно");
    fprintf(g_log, "\n══════════════════════════ EOF ══════════════════════════════\n");
    fclose(g_log);
    g_log = NULL;
}

/* ── Допоміжні функції виводу ─────────────────────────────────────────────── */

static void print_separator(void) {
    printf("─────────────────────────────────────────────────────────────────\n");
}

static void print_header(void) {
    printf("\n");
    print_separator();
    printf("  Спільний лічильник  ─  OpenMP бенчмарк синхронізації\n");
    print_separator();
}

/* ── Вирівняний слот для локального масиву (без false sharing) ─────────────
 *
 * Кожен потік пише лише у свій слот. Padding до CACHE_LINE_SIZE гарантує,
 * що різні слоти потрапляють на різні рядки кешу → жодної конкуренції
 * на рівні кешу між потоками.
 * ─────────────────────────────────────────────────────────────────────────── */

typedef struct {
    volatile long value;
    char          _pad[CACHE_LINE_SIZE - sizeof(long)];
} aligned_slot_t;

/* ── Метод 1: omp_lock_t ──────────────────────────────────────────────────── *
 *                                                                              *
 * OpenMP надає власний мʼютекс (omp_lock_t).  Кожен потік у своєму підциклі  *
 * захоплює лок, збільшує спільний лічильник і звільняє лок.                  *
 * Висока конкуренція при великій кількості потоків → сильна серіалізація.     *
 * ─────────────────────────────────────────────────────────────────────────── */

long count_omp_lock(long n, int nthreads) {
    long counter = 0;
    omp_lock_t lk;
    omp_init_lock(&lk);

    #pragma omp parallel num_threads(nthreads)
    {
        long per_thread = n / nthreads;
        /* останній потік забирає залишок */
        if (omp_get_thread_num() == nthreads - 1)
            per_thread += n % nthreads;

        for (long i = 0; i < per_thread; i++) {
            omp_set_lock(&lk);
            counter++;
            omp_unset_lock(&lk);
        }
    }

    omp_destroy_lock(&lk);
    return counter;
}

/* ── Метод 2: POSIX binary semaphore ─────────────────────────────────────── *
 *                                                                              *
 * sem_t, ініціалізований значенням 1 = binary semaphore = мʼютекс.           *
 * P-операція (sem_wait) зменшує лічильник до 0 (захоплення ресурсу).         *
 * V-операція (sem_post) повертає значення до 1 (звільнення ресурсу).         *
 *                                                                              *
 * Семантично відрізняється від мʼютексу: sem_post може викликати будь-який   *
 * потік, не тільки той, що захопив; підходить для сигналізації між потоками. *
 * ─────────────────────────────────────────────────────────────────────────── */

long count_semaphore(long n, int nthreads) {
    long  counter = 0;
    sem_t sem;
    /* pshared=0 → семафор між потоками одного процесу; init=1 → ресурс вільний */
    sem_init(&sem, 0, 1);

    #pragma omp parallel num_threads(nthreads)
    {
        long per_thread = n / nthreads;
        if (omp_get_thread_num() == nthreads - 1)
            per_thread += n % nthreads;

        for (long i = 0; i < per_thread; i++) {
            sem_wait(&sem);   /* P: заблокувати, якщо sem == 0 */
            counter++;
            sem_post(&sem);   /* V: збільшити та розбудити очікувача */
        }
    }

    sem_destroy(&sem);
    return counter;
}

/* ── Метод 3: локальний масив без блокувань ───────────────────────────────── *
 *                                                                              *
 * Кожен потік має власний вирівняний слот у масиві aligned_slot_t.            *
 * Під час паралельної фази жодних блокувань — кожен потік пише лише у свій   *
 * слот, тому false sharing відсутнє.                                           *
 * Після завершення паралельної секції головний потік підсумовує всі слоти     *
 * в одному послідовному проході — мінімальні накладні витрати.                *
 * ─────────────────────────────────────────────────────────────────────────── */

long count_local_array(long n, int nthreads) {
    /* Виділяємо масив вирівняних слотів */
    aligned_slot_t *slots = (aligned_slot_t *)aligned_alloc(
            CACHE_LINE_SIZE, (size_t)nthreads * sizeof(aligned_slot_t));
    if (!slots) return -1;
    memset(slots, 0, (size_t)nthreads * sizeof(aligned_slot_t));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        long per_thread = n / nthreads;
        if (tid == nthreads - 1)
            per_thread += n % nthreads;

        /* Без будь-якого захоплення лока — пишемо лише у свій слот */
        for (long i = 0; i < per_thread; i++)
            slots[tid].value++;
    }

    /* Послідовне підсумовування — один прохід, без синхронізації */
    long counter = 0;
    for (int t = 0; t < nthreads; t++)
        counter += slots[t].value;

    free(slots);
    return counter;
}

/* ── Функція запуску бенчмарку ───────────────────────────────────────────── *
 *                                                                              *
 * Для кожної конфігурації потоків виконує N_RUNS прогонів:                    *
 *   1) flush_cache()   — очищуємо кеш «холодними» даними                      *
 *   2) getrusage + CLOCK_PROCESS_CPUTIME_ID до і після                        *
 *   3) перевіряємо коректність: counter == n_ops                               *
 *   4) обчислюємо агреговану статистику                                        *
 * ─────────────────────────────────────────────────────────────────────────── */

typedef long (*cnt_func_t)(long, int);

static void run_benchmark(const char *name, const char *desc,
                          cnt_func_t func, long n_ops,
                          double baseline_time) {
    /* Заголовок таблиці */
    printf("\n  Метод: %s\n", name);
    printf("  %s\n", desc);
    printf("  %-6s  %-12s  %-9s  %-9s  %-9s  %-9s  %-8s  %-6s  %-5s\n",
           "Потоки", "Лічильник", "Сер.час с",
           "Відхил. с", "Min с", "Max с",
           "Прискор.", "CV%", "CtxSw");
    printf("  %-6s  %-12s  %-9s  %-9s  %-9s  %-9s  %-8s  %-6s  %-5s\n",
           "──────", "─────────", "─────────",
           "─────────", "─────", "─────",
           "────────", "────", "─────");

    /* Рядок-підсумок для лога */
    char log_line[640];
    int  log_pos  = 0;
    int  has_warn = 0;

    run_stat_t runs[N_RUNS];

    /* Інфо-рядок методу в лог */
    log_write(LOG_INFO, "════ Метод: %-42s ════", name);
    log_write(LOG_INFO, "Опис: %s", desc);

    for (int ci = 0; ci < NUM_CONFIGS; ci++) {
        int t = THREAD_CONFIGS[ci];

        printf("  [%2dп] прогрес: ", t);
        fflush(stdout);

        for (int r = 0; r < N_RUNS; r++) {
            flush_cache();

            struct rusage    ru0;
            struct timespec  cpu0;
            getrusage(RUSAGE_SELF, &ru0);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu0);

            double t_start = omp_get_wtime();
            long   result  = func(n_ops, t);
            double elapsed = omp_get_wtime() - t_start;

            struct rusage   ru1;
            struct timespec cpu1;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu1);
            getrusage(RUSAGE_SELF, &ru1);

            double cpu_sec = (cpu1.tv_sec  - cpu0.tv_sec) +
                             (cpu1.tv_nsec - cpu0.tv_nsec) * 1e-9;

            runs[r].wall_time = elapsed;
            runs[r].cpu_time  = cpu_sec;
            runs[r].final_val = result;
            runs[r].nvcsw     = ru1.ru_nvcsw  - ru0.ru_nvcsw;
            runs[r].nivcsw    = ru1.ru_nivcsw - ru0.ru_nivcsw;
            runs[r].minflt    = ru1.ru_minflt - ru0.ru_minflt;
            runs[r].majflt    = ru1.ru_majflt - ru0.ru_majflt;

            printf("."); fflush(stdout);
        }
        printf(" OK\n");

        /* Агрегування */
        agg_stat_t agg;
        memset(&agg, 0, sizeof(agg));
        agg.min_wall = runs[0].wall_time;
        agg.max_wall = runs[0].wall_time;

        for (int r = 0; r < N_RUNS; r++) {
            agg.mean_wall    += runs[r].wall_time;
            agg.mean_cpu     += runs[r].cpu_time;
            agg.total_nvcsw  += runs[r].nvcsw;
            agg.total_nivcsw += runs[r].nivcsw;
            agg.total_minflt += runs[r].minflt;
            agg.total_majflt += runs[r].majflt;
            if (runs[r].wall_time < agg.min_wall) agg.min_wall = runs[r].wall_time;
            if (runs[r].wall_time > agg.max_wall) agg.max_wall = runs[r].wall_time;
            if (runs[r].final_val != n_ops) agg.errors++;
        }
        agg.mean_wall /= N_RUNS;
        agg.mean_cpu  /= N_RUNS;

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

        /* Таблиця */
        char correctness[16];
        if (agg.errors == 0)
            snprintf(correctness, sizeof(correctness), "%ld", n_ops);
        else
            snprintf(correctness, sizeof(correctness), "ERR×%d", agg.errors);

        printf("  %-6d  %-12s  %-9.4f  %-9.4f  %-9.4f  %-9.4f  %-8.3f  %-6.2f  %ld/%ld\n",
               t, correctness,
               agg.mean_wall, agg.stddev_wall, agg.min_wall, agg.max_wall,
               agg.mean_speedup, agg.cv_wall,
               agg.total_nvcsw, agg.total_nivcsw);

        /* Детальний рядок perf у лог */
        log_write(LOG_INFO,
                  "[%s | %2dп]  рез=%s  час=%.4f±%.4fс [%.4f–%.4f]  "
                  "прискор=%.3fx  CPU-ефект=%.1f%%  CV=%.2f%%  "
                  "ctx_sw=%ld/%ld  page_faults=%ld+%ld",
                  name, t,
                  correctness,
                  agg.mean_wall, agg.stddev_wall,
                  agg.min_wall, agg.max_wall,
                  agg.mean_speedup, agg.cpu_eff, agg.cv_wall,
                  agg.total_nvcsw, agg.total_nivcsw,
                  agg.total_minflt, agg.total_majflt);

        /* Компактний підсумок методу */
        log_pos += snprintf(log_line + log_pos,
                            sizeof(log_line) - (size_t)log_pos,
                            "%dп=%.3fx(±%.3fs,CV=%.1f%%)",
                            t, agg.mean_speedup, agg.stddev_wall, agg.cv_wall);
        if (ci < NUM_CONFIGS - 1)
            log_pos += snprintf(log_line + log_pos,
                                sizeof(log_line) - (size_t)log_pos, " | ");

        if (t >= 2 && agg.mean_speedup < SPEEDUP_WARN_THR) has_warn = 1;

        /* Для ASCII-діаграми */
        if (g_result_count < MAX_RESULTS) {
            bench_result_t *res = &g_results[g_result_count++];
            strncpy(res->method, name, sizeof(res->method) - 1);
            res->threads = t;
            res->elapsed = agg.mean_wall;
            res->speedup = agg.mean_speedup;
        }
    }

    log_write(LOG_INFO, "ПІДСУМОК %-44s → %s", name, log_line);
    if (has_warn)
        log_write(LOG_WARN,
                  "%-44s → прискорення < %.1f при ≥2 потоках: "
                  "висока конкуренція або замало ядер CPU",
                  name, SPEEDUP_WARN_THR);
}

/* ── Точка входу ─────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    long n_ops = DEFAULT_OPS;
    if (argc >= 2) {
        long arg = atol(argv[1]);
        if (arg > 0) n_ops = arg;
        else fprintf(stderr,
                     "ERROR: кількість операцій має бути > 0, використовую %ld\n",
                     DEFAULT_OPS);
    }

    int max_omp = omp_get_max_threads();

    if (log_open(n_ops, max_omp) == 0)
        printf("  Лог-файл: %s\n", g_log_path);

    log_cpu_info();

    if (flush_buf_init() != 0) {
        fprintf(stderr, "WARN: не вдалося виділити %d МБ для очищення кешу\n",
                CACHE_FLUSH_MB);
        log_write(LOG_WARN, "Буфер очищення кешу недоступний — результати можуть бути завищені");
    } else {
        log_write(LOG_INFO, "Буфер очищення кешу виділено: %d МБ", CACHE_FLUSH_MB);
    }

    print_header();
    printf("  Операцій:       %ld\n", n_ops);
    printf("  Прогонів:       %d (з очищенням кешу %d МБ перед кожним)\n",
           N_RUNS, CACHE_FLUSH_MB);
    printf("  Макс. потоків:  %d (доступно системі)\n", max_omp);
    printf("  Конфігурації:   ");
    for (int i = 0; i < NUM_CONFIGS; i++)
        printf("%d%s", THREAD_CONFIGS[i], i < NUM_CONFIGS - 1 ? " " : "\n");
    print_separator();

    /* Базовий час — 1 потік, omp_lock, N_RUNS прогонів із flush */
    printf("\n  Вимірювання базового часу (1 потік, omp_lock, %d прогонів)...\n",
           N_RUNS);
    double baseline_times[N_RUNS];
    printf("  прогрес: ");
    for (int r = 0; r < N_RUNS; r++) {
        flush_cache();
        double ts = omp_get_wtime();
        count_omp_lock(n_ops, 1);
        baseline_times[r] = omp_get_wtime() - ts;
        printf("."); fflush(stdout);
    }
    printf(" OK\n");

    double baseline = 0.0;
    for (int r = 0; r < N_RUNS; r++) baseline += baseline_times[r];
    baseline /= N_RUNS;

    printf("  Базовий час (середній): %.4f с\n", baseline);
    log_write(LOG_INFO, "Базовий час (1 потік, omp_lock, N=%d): %.4f с", N_RUNS, baseline);

    /* ── Бенчмарк трьох методів ── */

    run_benchmark(
        "omp_lock",
        "OpenMP-мʼютекс: omp_set_lock / omp_unset_lock на кожну операцію",
        count_omp_lock,
        n_ops, baseline);

    run_benchmark(
        "semaphore",
        "POSIX binary semaphore (sem_t=1): sem_wait (P) / sem_post (V)",
        count_semaphore,
        n_ops, baseline);

    run_benchmark(
        "local_array",
        "Локальний масив (64-байт padding): лічимо без лока, підсумовуємо після",
        count_local_array,
        n_ops, baseline);

    /* ── Візуалізація ── */
    print_chart();
    log_chart();

    printf("\n");
    print_separator();
    printf("  Пояснення методів синхронізації:\n\n");
    printf("  • omp_lock    — OpenMP-мʼютекс (omp_lock_t).\n");
    printf("                  Кожен потік виконує: lock → ++counter → unlock.\n");
    printf("                  При T потоках доступ серіалізується:\n");
    printf("                  прискорення обмежене конкуренцією.\n\n");
    printf("  • semaphore   — POSIX binary semaphore (sem_t, init=1).\n");
    printf("                  sem_wait (P) зменшує лічильник до 0 (блокує).\n");
    printf("                  sem_post (V) збільшує до 1 (розблоковує).\n");
    printf("                  Ефективно ≈ мʼютекс, але з явною P/V семантикою;\n");
    printf("                  sem_post може викликати будь-який потік.\n\n");
    printf("  • local_array — кожен потік пише лише у свій вирівняний слот\n");
    printf("                  (64-байт padding → нема false sharing).\n");
    printf("                  Жодних блокувань під час лічби; фінальна сума —\n");
    printf("                  один послідовний прохід. Лінійне прискорення.\n\n");
    printf("  • CV%%         — коефіцієнт варіації: стабільність між прогонами.\n");
    printf("  • CtxSw       — перемикання контексту вол./прим. за %d прогонів.\n",
           N_RUNS);
    print_separator();
    printf("\n");

    flush_buf_free();
    log_close();
    return 0;
}
