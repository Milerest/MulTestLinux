/*
 * prod_cons_omp.c
 *
 * Багатопотокова черга «Виробник–Споживач» з використанням OpenMP.
 *
 * Модель:
 *   N/2 виробників генерують елементи і кладуть їх у кільцеву чергу;
 *   N/2 споживачів забирають елементи і обчислюють контрольну суму.
 *   Перевірка коректності: сума_спожитого == N*(N-1)/2.
 *
 * Три методи синхронізації:
 *   1) omp-lock   — omp_lock_t охороняє чергу; потоки крутяться (spinning)
 *                   при повній або порожній черзі.
 *   2) semaphore  — POSIX sem_t блокує виробника (черга повна) і споживача
 *                   (черга порожня), звільняючи CPU під час очікування.
 *   3) lock-free  — MPMC кільцева черга Д. В'юкова: CAS-операції без
 *                   м'ютексів, мінімальна затримка на передачу елементу.
 *
 * Конфігурації потоків: {1, 2, 4, 6, 8}
 *   1 потік  → послідовний режим (без черги, baseline)
 *   N потоків → N/2 виробників + N/2 споживачів
 *
 * Бенчмарк:
 *   N_RUNS=20 прогонів × кожна конфігурація × кожен метод.
 *   Перед кожним прогоном кеш процесора очищується буфером ~48 МБ.
 *
 * Perf-подібна статистика (без root):
 *   — wall time (середній, мін., макс., σ, CV%)
 *   — CPU time та ефективність ядер
 *   — контекстні перемикання та page faults (getrusage)
 *   — прискорення відносно послідовного режиму
 *
 * Логування:
 *   pc_bench_РРРРММДД_ГГХВСС.log — інформація про CPU, детальний perf
 *   рядок на конфігурацію, підсумок методу, ASCII-діаграма.
 *
 * Збірка (Linux, GCC ≥ 9):
 *   gcc -O2 -fopenmp -Wall -Wextra -D_GNU_SOURCE -o prod_cons_omp prod_cons_omp.c -lm
 *
 * Запуск:
 *   ./prod_cons_omp [кількість_елементів]
 */

#define _GNU_SOURCE

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

#define DEFAULT_N_ITEMS     10000000L   /* 10 млн елементів за замовчуванням  */
#define QUEUE_CAP           4096        /* розмір кільцевої черги (степінь 2) */
#define QUEUE_MASK          (QUEUE_CAP - 1)
#define N_RUNS              20          /* прогонів на кожну конфігурацію     */
#define LOG_PATH_LEN        256
#define BAR_WIDTH           36
#define SPEEDUP_WARN_THR    0.85        /* поріг попередження про прискорення */

/* Розмір буфера для очищення кешу: перевищує L3 більшості CPU */
#define CACHE_FLUSH_MB      48
#define CACHE_FLUSH_SIZE    ((size_t)(CACHE_FLUSH_MB) * 1024 * 1024)

/* Конфігурації: загальна кількість OMP-потоків */
static const int CFG_THREADS[] = {1, 2, 4, 6, 8};
#define NUM_CFGS    (int)(sizeof(CFG_THREADS) / sizeof(CFG_THREADS[0]))
#define MAX_RESULTS (NUM_CFGS * 3)      /* 3 методи                           */

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
    log_write(LOG_INFO, "──── Апаратне забезпечення ────────────────────");
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
            { strncpy(model, val, sizeof(model) - 1); got_model = 1; }
        if (!got_cache && strncmp(line, "cache size", 10) == 0)
            { strncpy(cache, val, sizeof(cache) - 1); got_cache = 1; }
        if (!got_freq  && strncmp(line, "cpu MHz",    7) == 0) {
            snprintf(freq, sizeof(freq), "%.0f МГц", atof(val));
            got_freq = 1;
        }
        if (got_model && got_cache && got_freq) break;
    }
    fclose(f);

    log_write(LOG_INFO, "Модель CPU:    %s", model);
    log_write(LOG_INFO, "Частота CPU:   %s", freq);
    log_write(LOG_INFO, "Кеш (L2/L3):   %s", cache);
    log_write(LOG_INFO, "Буфер flush:   %d МБ (очищення між прогонами)", CACHE_FLUSH_MB);
    log_write(LOG_INFO, "Прогонів:      %d × кожна конфігурація",        N_RUNS);
    log_write(LOG_INFO, "Розмір черги:  %d слотів (кільцевий буфер)",   QUEUE_CAP);
    log_write(LOG_INFO, "───────────────────────────────────────────────");
}

/* ── Відкриття / закриття лог-файлу ──────────────────────────────────────── */

static int log_open(long n, int max_threads) {
    time_t t      = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(g_log_path, sizeof(g_log_path), "pc_bench_%Y%m%d_%H%M%S.log", tm);
    g_log = fopen(g_log_path, "w");
    if (!g_log) {
        fprintf(stderr, "WARN: не вдалося створити лог-файл '%s'\n", g_log_path);
        return -1;
    }
    char ts[32];
    current_timestamp(ts, sizeof(ts));
    fprintf(g_log,
        "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(g_log,
        "║  Виробник-Споживач OpenMP бенчмарк — сесія %s  ║\n", ts);
    fprintf(g_log,
        "╚══════════════════════════════════════════════════════════════╝\n\n");
    fflush(g_log);
    log_write(LOG_INFO, "Запуск програми");
    log_write(LOG_INFO, "Елементів:     %ld", n);
    log_write(LOG_INFO, "Макс. потоків: %d", max_threads);
    log_write(LOG_INFO, "Прогонів:      %d", N_RUNS);
    log_write(LOG_INFO, "Розмір черги:  %d слотів", QUEUE_CAP);
    return 0;
}

static void log_close(void) {
    if (!g_log) return;
    log_write(LOG_INFO, "Програму завершено успішно");
    fprintf(g_log, "\n══════════════════════════ EOF ══════════════════════════════\n");
    fclose(g_log);
    g_log = NULL;
}

/* ── Очищення кешу між прогонами ─────────────────────────────────────────── *
 *                                                                              *
 * Записуємо і читаємо буфер, більший за L3-кеш, щоб витіснити робочі дані   *
 * бенчмарку. Без цього перші прогони будуть «холодними», а решта — «теплими».*
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
    double wall_time;   /* реальний час (секунди)                       */
    double cpu_time;    /* час CPU (CLOCK_PROCESS_CPUTIME_ID)           */
    double throughput;  /* пропускна здатність (елементів/с)            */
    long   nvcsw;       /* добровільні перемикання контексту            */
    long   nivcsw;      /* примусові перемикання контексту              */
    long   minflt;      /* незначні відмови сторінок                    */
    long   majflt;      /* значні відмови сторінок                      */
    long   sum_result;  /* контрольна сума спожитих елементів           */
} run_stat_t;

typedef struct {
    double mean_wall, stddev_wall, min_wall, max_wall, cv_wall;
    double mean_cpu, cpu_eff;
    double mean_tput;
    double mean_speedup;
    long   total_nvcsw, total_nivcsw, total_minflt, total_majflt;
    int    valid;       /* 1 якщо всі контрольні суми пройшли           */
} agg_stat_t;

typedef struct {
    char   method[80];
    int    threads;
    double throughput;
    double speedup;
} bench_result_t;

static bench_result_t g_results[MAX_RESULTS];
static int            g_result_count = 0;

/* ── ASCII-діаграма прискорення ──────────────────────────────────────────── */

static void build_bar(char *buf, int bsz, int bar_len) {
    const char BLOCK[] = "\xe2\x96\x88";
    const int  BLEN    = 3;
    int pos = 0;
    for (int i = 0; i < BAR_WIDTH && pos + BLEN < bsz; i++) {
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
            strncpy(prev_method, r->method, sizeof(prev_method) - 1);
            fprintf(out, "\n  [%s]\n", r->method);
        }
        int bar_len = (int)((r->speedup / max_sp) * BAR_WIDTH);
        if (bar_len < 1) bar_len = 1;
        build_bar(bar_buf, sizeof(bar_buf), bar_len);

        if (r->threads <= 1) {
            fprintf(out, "  %2dп(посл.)  [%s] %.2fx\n",
                    r->threads, bar_buf, r->speedup);
        } else {
            int np = r->threads / 2, nc = r->threads - np;
            fprintf(out, "  %2dп(%d+%d)   [%s] %.2fx\n",
                    r->threads, np, nc, bar_buf, r->speedup);
        }
    }
    fprintf(out, "\n%s\n", sep);
}

static void print_chart(void) { render_chart(stdout); }
static void log_chart(void) {
    if (!g_log) return;
    log_write(LOG_INFO, "──── ASCII-діаграма прискорення ────────────────");
    render_chart(g_log);
}

/* ── Допоміжні функції виводу ────────────────────────────────────────────── */

static void print_sep(void) {
    printf("─────────────────────────────────────────────────────────────────\n");
}

static double now_sec(void)     { return omp_get_wtime(); }
static long   expected_sum(long n) { return n * (n - 1) / 2; }

/* Формат конфігурації: "1(посл.)" або "4(2+2)" */
static void cfg_label(int t, char *buf, int bsz) {
    if (t <= 1)
        snprintf(buf, (size_t)bsz, "1(посл.)");
    else {
        int np = t / 2, nc = t - np;
        snprintf(buf, (size_t)bsz, "%d(%d+%d)", t, np, nc);
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * МЕТОД 0 — Послідовний (без черги, baseline)
 *
 * Виробник записує елемент у volatile-змінну; споживач одразу читає.
 * Це симулює мінімальний передачі через однослотову «чергу» без синхронізації
 * і дає реальний upper-bound на пропускну здатність.
 * ════════════════════════════════════════════════════════════════════════════ */

static run_stat_t run_sequential(long n_items) {
    run_stat_t rs;
    memset(&rs, 0, sizeof(rs));

    struct rusage   ru0, ru1;
    struct timespec cpu0, cpu1;
    getrusage(RUSAGE_SELF, &ru0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu0);

    double ts = now_sec();
    volatile long slot = 0;
    long sum = 0;
    for (long i = 0; i < n_items; i++) {
        slot = i;       /* виробник кладе елемент   */
        sum += slot;    /* споживач бере елемент     */
    }
    rs.wall_time  = now_sec() - ts;
    rs.sum_result = sum;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu1);
    getrusage(RUSAGE_SELF, &ru1);

    rs.cpu_time  = (cpu1.tv_sec  - cpu0.tv_sec)
                 + (cpu1.tv_nsec - cpu0.tv_nsec) * 1e-9;
    rs.throughput = (rs.wall_time > 0.0) ? n_items / rs.wall_time : 0.0;
    rs.nvcsw  = ru1.ru_nvcsw  - ru0.ru_nvcsw;
    rs.nivcsw = ru1.ru_nivcsw - ru0.ru_nivcsw;
    rs.minflt = ru1.ru_minflt - ru0.ru_minflt;
    rs.majflt = ru1.ru_majflt - ru0.ru_majflt;
    return rs;
}

/* ════════════════════════════════════════════════════════════════════════════
 * МЕТОД 1 — OMP-замок (omp_lock_t + активне очікування)
 *
 * Кільцева черга захищена одним omp_lock_t.
 * При повній черзі виробник крутиться (spin) відпускаючи замок між спробами.
 * При порожній черзі споживач аналогічно крутиться.
 * Низькі затримки на незаблокованих операціях; невигідний при незбалансованих
 * швидкостях виробника/споживача (марна трата CPU).
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    long         buf[QUEUE_CAP];
    volatile int head, tail, count;
    omp_lock_t   lock;
} lk_queue_t;

static lk_queue_t g_lkq;

static void lkq_init(lk_queue_t *q) {
    q->head = q->tail = q->count = 0;
    omp_init_lock(&q->lock);
}
static void lkq_destroy(lk_queue_t *q) { omp_destroy_lock(&q->lock); }

/* Повертає 1 при успіху, 0 якщо черга повна */
static int lkq_enqueue(lk_queue_t *q, long item) {
    omp_set_lock(&q->lock);
    if (q->count >= QUEUE_CAP) { omp_unset_lock(&q->lock); return 0; }
    q->buf[q->tail] = item;
    q->tail = (q->tail + 1) & QUEUE_MASK;
    q->count++;
    omp_unset_lock(&q->lock);
    return 1;
}

/* Повертає 1 при успіху, 0 якщо черга порожня */
static int lkq_dequeue(lk_queue_t *q, long *item) {
    omp_set_lock(&q->lock);
    if (q->count <= 0) { omp_unset_lock(&q->lock); return 0; }
    *item = q->buf[q->head];
    q->head = (q->head + 1) & QUEUE_MASK;
    q->count--;
    omp_unset_lock(&q->lock);
    return 1;
}

static run_stat_t run_lock(long n_items, int total_threads) {
    if (total_threads <= 1) return run_sequential(n_items);

    int np = total_threads / 2;
    int nc = total_threads - np;
    long g_next_item = 0, g_done_prod = 0, g_total_sum = 0;

    lkq_init(&g_lkq);

    run_stat_t rs;
    memset(&rs, 0, sizeof(rs));
    struct rusage   ru0, ru1;
    struct timespec cpu0, cpu1;
    getrusage(RUSAGE_SELF, &ru0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu0);
    double ts = now_sec();

    #pragma omp parallel num_threads(total_threads) \
        shared(np, nc, g_next_item, g_done_prod, g_total_sum, n_items, g_lkq)
    {
        int tid = omp_get_thread_num();

        if (tid < np) {
            /* ── Виробник — статичний розподіл діапазону ── */
            long chunk = (n_items + np - 1) / np;
            long start = tid * chunk;
            long end   = start + chunk;
            if (end > n_items) end = n_items;
            for (long item = start; item < end; item++) {
                while (!lkq_enqueue(&g_lkq, item)) { /* spin */ }
            }
            /* Останній виробник надсилає poison-pill кожному споживачу */
            long done = __atomic_fetch_add(&g_done_prod, 1L, __ATOMIC_ACQ_REL) + 1L;
            if (done == (long)np) {
                for (int i = 0; i < nc; i++)
                    while (!lkq_enqueue(&g_lkq, -1L)) { /* spin */ }
            }
        } else {
            /* ── Споживач ── */
            long local_sum = 0;
            for (;;) {
                long item;
                if (lkq_dequeue(&g_lkq, &item)) {
                    if (item < 0L) break;   /* poison pill */
                    local_sum += item;
                }
            }
            __atomic_fetch_add(&g_total_sum, local_sum, __ATOMIC_RELAXED);
        }
    }

    rs.wall_time = now_sec() - ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu1);
    getrusage(RUSAGE_SELF, &ru1);
    lkq_destroy(&g_lkq);

    rs.cpu_time   = (cpu1.tv_sec  - cpu0.tv_sec)
                  + (cpu1.tv_nsec - cpu0.tv_nsec) * 1e-9;
    rs.throughput = (rs.wall_time > 0.0) ? n_items / rs.wall_time : 0.0;
    rs.sum_result = g_total_sum;
    rs.nvcsw  = ru1.ru_nvcsw  - ru0.ru_nvcsw;
    rs.nivcsw = ru1.ru_nivcsw - ru0.ru_nivcsw;
    rs.minflt = ru1.ru_minflt - ru0.ru_minflt;
    rs.majflt = ru1.ru_majflt - ru0.ru_majflt;
    return rs;
}

/* ════════════════════════════════════════════════════════════════════════════
 * МЕТОД 2 — Семафор (POSIX sem_t + omp_lock_t)
 *
 * sem_empty: лічить порожні слоти (початок = QUEUE_CAP).
 * sem_full:  лічить заповнені слоти (початок = 0).
 * Виробник виконує sem_wait(&sem_empty) → блокується при повній черзі,
 * звільняючи CPU. Споживач виконує sem_wait(&sem_full) → блокується при
 * порожній. Два окремі замки для голови і хвоста знижують конкуренцію.
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    long       buf[QUEUE_CAP];
    int        head, tail;
    omp_lock_t prod_lock;   /* захищає tail */
    omp_lock_t cons_lock;   /* захищає head */
    sem_t      sem_empty;   /* кількість порожніх слотів */
    sem_t      sem_full;    /* кількість заповнених слотів */
} sm_queue_t;

static sm_queue_t g_smq;

static void smq_init(sm_queue_t *q) {
    q->head = q->tail = 0;
    omp_init_lock(&q->prod_lock);
    omp_init_lock(&q->cons_lock);
    sem_init(&q->sem_empty, 0, QUEUE_CAP);
    sem_init(&q->sem_full,  0, 0);
}

static void smq_destroy(sm_queue_t *q) {
    omp_destroy_lock(&q->prod_lock);
    omp_destroy_lock(&q->cons_lock);
    sem_destroy(&q->sem_empty);
    sem_destroy(&q->sem_full);
}

static void smq_enqueue(sm_queue_t *q, long item) {
    sem_wait(&q->sem_empty);
    omp_set_lock(&q->prod_lock);
    q->buf[q->tail] = item;
    q->tail = (q->tail + 1) & QUEUE_MASK;
    omp_unset_lock(&q->prod_lock);
    sem_post(&q->sem_full);
}

static long smq_dequeue(sm_queue_t *q) {
    sem_wait(&q->sem_full);
    omp_set_lock(&q->cons_lock);
    long item = q->buf[q->head];
    q->head = (q->head + 1) & QUEUE_MASK;
    omp_unset_lock(&q->cons_lock);
    sem_post(&q->sem_empty);
    return item;
}

static run_stat_t run_sem(long n_items, int total_threads) {
    if (total_threads <= 1) return run_sequential(n_items);

    int np = total_threads / 2;
    int nc = total_threads - np;
    long g_next_item = 0, g_done_prod = 0, g_total_sum = 0;

    smq_init(&g_smq);

    run_stat_t rs;
    memset(&rs, 0, sizeof(rs));
    struct rusage   ru0, ru1;
    struct timespec cpu0, cpu1;
    getrusage(RUSAGE_SELF, &ru0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu0);
    double ts = now_sec();

    #pragma omp parallel num_threads(total_threads) \
        shared(np, nc, g_next_item, g_done_prod, g_total_sum, n_items, g_smq)
    {
        int tid = omp_get_thread_num();

        if (tid < np) {
            /* ── Виробник ── */
            for (;;) {
                long item = __atomic_fetch_add(&g_next_item, 1L, __ATOMIC_RELAXED);
                if (item >= n_items) break;
                smq_enqueue(&g_smq, item);
            }
            long done = __atomic_fetch_add(&g_done_prod, 1L, __ATOMIC_ACQ_REL) + 1L;
            if (done == (long)np) {
                for (int i = 0; i < nc; i++)
                    smq_enqueue(&g_smq, -1L);
            }
        } else {
            /* ── Споживач ── */
            long local_sum = 0;
            for (;;) {
                long item = smq_dequeue(&g_smq);
                if (item < 0L) break;
                local_sum += item;
            }
            __atomic_fetch_add(&g_total_sum, local_sum, __ATOMIC_RELAXED);
        }
    }

    rs.wall_time = now_sec() - ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu1);
    getrusage(RUSAGE_SELF, &ru1);
    smq_destroy(&g_smq);

    rs.cpu_time   = (cpu1.tv_sec  - cpu0.tv_sec)
                  + (cpu1.tv_nsec - cpu0.tv_nsec) * 1e-9;
    rs.throughput = (rs.wall_time > 0.0) ? n_items / rs.wall_time : 0.0;
    rs.sum_result = g_total_sum;
    rs.nvcsw  = ru1.ru_nvcsw  - ru0.ru_nvcsw;
    rs.nivcsw = ru1.ru_nivcsw - ru0.ru_nivcsw;
    rs.minflt = ru1.ru_minflt - ru0.ru_minflt;
    rs.majflt = ru1.ru_majflt - ru0.ru_majflt;
    return rs;
}

/* ════════════════════════════════════════════════════════════════════════════
 * МЕТОД 3 — Без замку: MPMC черга Д. В'юкова (GCC atomic built-ins)
 *
 * Кожен слот містить sequence-число. Виробник CAS-ує enq_pos, читає seq,
 * якщо diff==0 — слот вільний. Споживач аналогічно CAS-ує deq_pos.
 * Немає м'ютексів: тільки атомарні операції та spinning при повній/порожній.
 * Кожен cell вирівняний на окремий рядок кешу (64 байти) — нема false sharing.
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    long data;
    int  seq;
    char _pad[52];              /* разом з data+seq = 64 байти (cache line) */
} __attribute__((aligned(64))) lf_cell_t;

typedef struct {
    lf_cell_t    cell[QUEUE_CAP];   /* 4096 × 64 = 256 КБ                   */
    volatile int enq_pos;
    char         _p0[60];           /* enq_pos на окремому cache line        */
    volatile int deq_pos;
    char         _p1[60];           /* deq_pos на окремому cache line        */
} lf_queue_t;

static lf_queue_t g_lfq;

static void lfq_init(lf_queue_t *q) {
    for (int i = 0; i < QUEUE_CAP; i++) {
        __atomic_store_n(&q->cell[i].seq, i, __ATOMIC_RELAXED);
        q->cell[i].data = 0;
    }
    __atomic_store_n(&q->enq_pos, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&q->deq_pos, 0, __ATOMIC_RELAXED);
}

/* Повертає 1 при успіху, 0 якщо черга повна */
static int lfq_enqueue(lf_queue_t *q, long data) {
    lf_cell_t *cell;
    int pos = __atomic_load_n(&q->enq_pos, __ATOMIC_RELAXED);
    for (;;) {
        cell = &q->cell[pos & QUEUE_MASK];
        int seq  = __atomic_load_n(&cell->seq, __ATOMIC_ACQUIRE);
        int diff = seq - pos;
        if (diff == 0) {
            if (__atomic_compare_exchange_n(&q->enq_pos, &pos, pos + 1,
                                            1,
                                            __ATOMIC_RELAXED,
                                            __ATOMIC_RELAXED))
                break;              /* слот зайнятий нами */
        } else if (diff < 0) {
            return 0;               /* черга повна */
        } else {
            /* наш pos застарів — перезавантажуємо */
            pos = __atomic_load_n(&q->enq_pos, __ATOMIC_RELAXED);
        }
    }
    cell->data = data;
    __atomic_store_n(&cell->seq, pos + 1, __ATOMIC_RELEASE);
    return 1;
}

/* Повертає 1 при успіху, 0 якщо черга порожня */
static int lfq_dequeue(lf_queue_t *q, long *data) {
    lf_cell_t *cell;
    int pos = __atomic_load_n(&q->deq_pos, __ATOMIC_RELAXED);
    for (;;) {
        cell = &q->cell[pos & QUEUE_MASK];
        int seq  = __atomic_load_n(&cell->seq, __ATOMIC_ACQUIRE);
        int diff = seq - (pos + 1);
        if (diff == 0) {
            if (__atomic_compare_exchange_n(&q->deq_pos, &pos, pos + 1,
                                            1,
                                            __ATOMIC_RELAXED,
                                            __ATOMIC_RELAXED))
                break;              /* слот зайнятий нами */
        } else if (diff < 0) {
            return 0;               /* черга порожня */
        } else {
            pos = __atomic_load_n(&q->deq_pos, __ATOMIC_RELAXED);
        }
    }
    *data = cell->data;
    __atomic_store_n(&cell->seq, pos + QUEUE_CAP, __ATOMIC_RELEASE);
    return 1;
}

static run_stat_t run_lockfree(long n_items, int total_threads) {
    if (total_threads <= 1) return run_sequential(n_items);

    int np = total_threads / 2;
    int nc = total_threads - np;
    long g_next_item = 0, g_done_prod = 0, g_total_sum = 0;

    lfq_init(&g_lfq);

    run_stat_t rs;
    memset(&rs, 0, sizeof(rs));
    struct rusage   ru0, ru1;
    struct timespec cpu0, cpu1;
    getrusage(RUSAGE_SELF, &ru0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu0);
    double ts = now_sec();

    #pragma omp parallel num_threads(total_threads) \
        shared(np, nc, g_next_item, g_done_prod, g_total_sum, n_items, g_lfq)
    {
        int tid = omp_get_thread_num();

        if (tid < np) {
            /* ── Виробник ── */
            for (;;) {
                long item = __atomic_fetch_add(&g_next_item, 1L, __ATOMIC_RELAXED);
                if (item >= n_items) break;
                while (!lfq_enqueue(&g_lfq, item)) { /* spin */ }
            }
            long done = __atomic_fetch_add(&g_done_prod, 1L, __ATOMIC_ACQ_REL) + 1L;
            if (done == (long)np) {
                for (int i = 0; i < nc; i++)
                    while (!lfq_enqueue(&g_lfq, -1L)) { /* spin */ }
            }
        } else {
            /* ── Споживач ── */
            long local_sum = 0;
            for (;;) {
                long item;
                if (lfq_dequeue(&g_lfq, &item)) {
                    if (item < 0L) break;
                    local_sum += item;
                }
            }
            __atomic_fetch_add(&g_total_sum, local_sum, __ATOMIC_RELAXED);
        }
    }

    rs.wall_time = now_sec() - ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu1);
    getrusage(RUSAGE_SELF, &ru1);

    rs.cpu_time   = (cpu1.tv_sec  - cpu0.tv_sec)
                  + (cpu1.tv_nsec - cpu0.tv_nsec) * 1e-9;
    rs.throughput = (rs.wall_time > 0.0) ? n_items / rs.wall_time : 0.0;
    rs.sum_result = g_total_sum;
    rs.nvcsw  = ru1.ru_nvcsw  - ru0.ru_nvcsw;
    rs.nivcsw = ru1.ru_nivcsw - ru0.ru_nivcsw;
    rs.minflt = ru1.ru_minflt - ru0.ru_minflt;
    rs.majflt = ru1.ru_majflt - ru0.ru_majflt;
    return rs;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Агрегований бенчмарк для одного методу
 * ════════════════════════════════════════════════════════════════════════════ */

typedef run_stat_t (*run_func_t)(long, int);

static void run_benchmark(const char *name, run_func_t func,
                          long n_items, double baseline_tput) {
    /* ── Лог-рядок про метод ── */
    log_write(LOG_INFO, "════ Метод: %-40s ════", name);

    printf("\n  Метод: %s\n", name);
    printf("  %-12s  %-13s  %-10s  %-10s  %-9s  %-9s  %-8s  %-6s  %-9s  %s\n",
           "Конфіг", "Мел./с", "Сер.час с",
           "Відхил. с", "Min с", "Max с",
           "Прискор.", "CV%", "CtxSw", "Сума");
    printf("  %-12s  %-13s  %-10s  %-10s  %-9s  %-9s  %-8s  %-6s  %-9s  %s\n",
           "────────────", "─────────────", "──────────",
           "──────────", "─────────", "─────────",
           "────────", "──────", "─────────", "────");

    char log_line[768];
    int  log_pos  = 0;
    int  has_warn = 0;
    run_stat_t runs[N_RUNS];

    for (int ci = 0; ci < NUM_CFGS; ci++) {
        int t = CFG_THREADS[ci];
        char label[24];
        cfg_label(t, label, sizeof(label));

        printf("  [%s] прогрес: ", label);
        fflush(stdout);

        for (int r = 0; r < N_RUNS; r++) {
            flush_cache();
            runs[r] = func(n_items, t);
            printf("."); fflush(stdout);
        }
        printf(" OK\n");

        /* ── Агрегування статистики ── */
        agg_stat_t agg;
        memset(&agg, 0, sizeof(agg));
        agg.min_wall = runs[0].wall_time;
        agg.max_wall = runs[0].wall_time;
        agg.valid    = 1;
        long exp     = expected_sum(n_items);

        for (int r = 0; r < N_RUNS; r++) {
            agg.mean_wall    += runs[r].wall_time;
            agg.mean_cpu     += runs[r].cpu_time;
            agg.mean_tput    += runs[r].throughput;
            agg.total_nvcsw  += runs[r].nvcsw;
            agg.total_nivcsw += runs[r].nivcsw;
            agg.total_minflt += runs[r].minflt;
            agg.total_majflt += runs[r].majflt;
            if (runs[r].wall_time < agg.min_wall) agg.min_wall = runs[r].wall_time;
            if (runs[r].wall_time > agg.max_wall) agg.max_wall = runs[r].wall_time;
            if (runs[r].sum_result != exp)         agg.valid    = 0;
        }
        agg.mean_wall /= N_RUNS;
        agg.mean_cpu  /= N_RUNS;
        agg.mean_tput /= N_RUNS;

        for (int r = 0; r < N_RUNS; r++) {
            double d = runs[r].wall_time - agg.mean_wall;
            agg.stddev_wall += d * d;
        }
        agg.stddev_wall  = sqrt(agg.stddev_wall / N_RUNS);
        agg.cv_wall      = (agg.mean_wall > 0.0)
                           ? (agg.stddev_wall / agg.mean_wall * 100.0) : 0.0;
        agg.mean_speedup = (baseline_tput > 0.0)
                           ? (agg.mean_tput / baseline_tput) : 1.0;
        agg.cpu_eff      = (agg.mean_wall > 0.0 && t > 0)
                           ? (agg.mean_cpu / (agg.mean_wall * (double)t) * 100.0)
                           : 0.0;

        const char *vstr = agg.valid ? "✓ OK" : "✗ ПОМИЛКА";

        printf("  %-12s  %-13.3f  %-10.4f  %-10.4f  %-9.4f  %-9.4f"
               "  %-8.3f  %-6.2f  %ld/%-5ld  %s\n",
               label, agg.mean_tput / 1e6,
               agg.mean_wall, agg.stddev_wall, agg.min_wall, agg.max_wall,
               agg.mean_speedup, agg.cv_wall,
               agg.total_nvcsw, agg.total_nivcsw, vstr);

        /* ── Детальний perf-рядок у лог ── */
        log_write(LOG_INFO,
            "[%s | %s]  тпут=%.3f Мел/с  "
            "час=%.4f±%.4fс [%.4f–%.4f]  "
            "прискор=%.3fx  CPU-ефект=%.1f%%  CV=%.2f%%  "
            "ctx_sw=%ld/%ld  page_faults=%ld+%ld  "
            "сума=%s",
            name, label,
            agg.mean_tput / 1e6,
            agg.mean_wall, agg.stddev_wall, agg.min_wall, agg.max_wall,
            agg.mean_speedup, agg.cpu_eff, agg.cv_wall,
            agg.total_nvcsw, agg.total_nivcsw,
            agg.total_minflt, agg.total_majflt,
            agg.valid ? "✓ правильно" : "✗ ПОМИЛКА!");

        /* Компактний підсумковий рядок методу */
        log_pos += snprintf(log_line + log_pos,
                            sizeof(log_line) - (size_t)log_pos,
                            "%s=%.3fx(%.2fМел/с)",
                            label, agg.mean_speedup, agg.mean_tput / 1e6);
        if (ci < NUM_CFGS - 1)
            log_pos += snprintf(log_line + log_pos,
                                sizeof(log_line) - (size_t)log_pos, " | ");

        if (t >= 4 && agg.mean_speedup < SPEEDUP_WARN_THR) has_warn = 1;

        if (!agg.valid) {
            printf("  *** УВАГА: контрольна сума не збіглась! ***\n");
            log_write(LOG_ERROR,
                "Перевірка суми провалена для методу [%s], конфіг [%s]!",
                name, label);
        }

        /* Зберігаємо для ASCII-діаграми */
        if (g_result_count < MAX_RESULTS) {
            bench_result_t *res = &g_results[g_result_count++];
            strncpy(res->method, name, sizeof(res->method) - 1);
            res->threads    = t;
            res->throughput = agg.mean_tput;
            res->speedup    = agg.mean_speedup;
        }
    }

    log_write(LOG_INFO, "ПІДСУМОК %-42s → %s", name, log_line);
    if (has_warn)
        log_write(LOG_WARN,
            "%-42s → прискорення < %.2f при ≥4 потоках: "
            "можлива конкуренція або недостатньо ядер",
            name, SPEEDUP_WARN_THR);
}

/* ── Точка входу ─────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    long n_items = DEFAULT_N_ITEMS;
    if (argc >= 2) {
        long a = atol(argv[1]);
        if (a > 0) n_items = a;
        else fprintf(stderr,
             "ERROR: кількість елементів має бути > 0, "
             "використовую %ld\n", DEFAULT_N_ITEMS);
    }

    int max_threads = omp_get_max_threads();

    if (log_open(n_items, max_threads) == 0)
        printf("  Лог-файл: %s\n", g_log_path);

    log_cpu_info();

    if (flush_buf_init() != 0) {
        fprintf(stderr,
            "WARN: не вдалося виділити %d МБ для очищення кешу\n",
            CACHE_FLUSH_MB);
        log_write(LOG_WARN,
            "Буфер очищення кешу недоступний — результати можуть бути завищені");
    } else {
        log_write(LOG_INFO,
            "Буфер очищення кешу виділено: %d МБ", CACHE_FLUSH_MB);
    }

    /* ── Заголовок ── */
    printf("\n");
    print_sep();
    printf("  Черга «Виробник–Споживач»  ─  OpenMP бенчмарк\n");
    print_sep();
    printf("  Елементів:      %ld\n", n_items);
    printf("  Прогонів:       %d (з очищенням кешу %d МБ перед кожним)\n",
           N_RUNS, CACHE_FLUSH_MB);
    printf("  Макс. потоків:  %d\n", max_threads);
    printf("  Конфігурації:   ");
    for (int i = 0; i < NUM_CFGS; i++) {
        char lbl[24];
        cfg_label(CFG_THREADS[i], lbl, sizeof(lbl));
        printf("%s", lbl);
        if (i < NUM_CFGS - 1) printf(" | ");
    }
    printf("\n");
    printf("  Черга:          %d слотів (кільцевий буфер)\n", QUEUE_CAP);
    printf("  Перевірка:      сума ≡ N×(N−1)/2 = %ld\n", expected_sum(n_items));
    print_sep();

    /* ── Вимірювання базової пропускної здатності ── */
    printf("\n  Вимірювання базової продуктивності (послідовний, %d прогонів)...\n",
           N_RUNS);
    printf("  прогрес: ");
    fflush(stdout);

    double bl_arr[N_RUNS];
    for (int r = 0; r < N_RUNS; r++) {
        flush_cache();
        run_stat_t rs = run_lock(n_items, 2);
        bl_arr[r] = rs.throughput;
        printf("."); fflush(stdout);
    }
    printf(" OK\n");

    double baseline_tput = 0.0;
    for (int r = 0; r < N_RUNS; r++) baseline_tput += bl_arr[r];
    baseline_tput /= N_RUNS;

    printf("  Базова продуктивність (середня): %.3f Мел./с\n", baseline_tput / 1e6);
    log_write(LOG_INFO,
        "Базова продуктивність (послідовний, N=%d): %.3f Мел./с",
        N_RUNS, baseline_tput / 1e6);

    /* ── Бенчмарк трьох методів ── */
    run_benchmark("omp-lock (spinning)",      run_lock,     n_items, baseline_tput);
    run_benchmark("semaphore (POSIX sem_t)",  run_sem,      n_items, baseline_tput);
    run_benchmark("lock-free (MPMC В'юкова)", run_lockfree, n_items, baseline_tput);

    /* ── ASCII-діаграма ── */
    print_chart();
    log_chart();

    /* ── Пояснення ── */
    printf("\n");
    print_sep();
    printf("  Пояснення:\n");
    printf("  • omp-lock   — omp_lock_t захищає всю чергу; потоки крутяться (spin)\n");
    printf("                 при повній/порожній черзі — просто, мала затримка, але\n");
    printf("                 марнує CPU при незбалансованих швидкостях.\n");
    printf("  • semaphore  — sem_t блокує потік (повна/порожня), звільняючи CPU;\n");
    printf("                 більше перемикань контексту, але вигідно при дисбалансі.\n");
    printf("  • lock-free  — MPMC черга В'юкова: CAS без м'ютексів, мінімальна\n");
    printf("                 затримка передачі; спін при повній/порожній черзі.\n");
    printf("  • Прискорення — відносно послідовного режиму (без черги).\n");
    printf("  • N(П+С)     — N потоків: П виробників + С споживачів.\n");
    printf("  • CV%%        — коефіцієнт варіації: стабільність між прогонами.\n");
    printf("  • CtxSw      — перемикання контексту вол./прим. за %d прогонів.\n",
           N_RUNS);
    printf("  • ✓/✗       — перевірка коректності (сума спожитого).\n");
    print_sep();
    printf("\n");

    flush_buf_free();
    log_close();
    return 0;
}
