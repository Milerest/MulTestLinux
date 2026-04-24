/*
 * sort_omp.c
 *
 * Багатопотокове сортування масиву цілих чисел з використанням OpenMP.
 *
 * Методи сортування та синхронізація (omp_lock_t як mutex):
 *   1) Merge Sort  — Lock-Striping: масив замків (по одному на merge-слот)
 *                    захищає запис у спільний буфер tmp під час фази злиття.
 *   2) Quicksort   — черга задач (LIFO-стек) захищена єдиним omp_lock_t.
 *                    Потоки конкурують за задачі; pending-лічильник під тим
 *                    самим замком визначає умову завершення роботи.
 *
 * Бенчмарк:
 *   Конфігурації потоків: 1 2 4 6 8
 *   Кожна конфігурація:   20 прогонів з очищенням кешу (~48 МБ)
 *   Базовий час:          20 прогонів Merge Sort / 1 потік
 *
 * Статистика:
 *   — середній / мін / макс / stddev wall-time
 *   — CPU-time (CLOCK_PROCESS_CPUTIME_ID) і ефективність ядер
 *   — добровільні / примусові перемикання контексту (getrusage)
 *   — minor / major page faults
 *   — коефіцієнт варіації CV%
 *
 * Perf stat:
 *   Якщо `perf` доступний, для кожної конфігурації виконується один
 *   додатковий прогін під `perf stat`, результат записується у лог.
 *   Бінарник викликає сам себе з прапором --single-run для цих вимірювань.
 *
 * Логування:
 *   Файл sort_bench_YYYYMMDD_HHMMSS.log (українська мова):
 *     — конфігурація CPU
 *     — інформаційний рядок на кожний метод
 *     — детальний perf-рядок на кожну конфігурацію потоків
 *     — ASCII-діаграма прискорення
 *
 * Збірка (Linux / WSL, GCC):
 *   gcc -O2 -fopenmp -Wall -Wextra -o sort_omp sort_omp.c -lm
 *
 * Запуск:
 *   ./sort_omp [розмір_масиву]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>

/* ══════════════════════════════════════════════════════════════════════
 *  Константи
 * ══════════════════════════════════════════════════════════════════════ */

#define DEFAULT_N           10000000L   /* 10 млн елементів                    */
#define N_RUNS              20          /* прогонів на конфігурацію            */
#define CACHE_FLUSH_MB      48          /* МБ для очищення кешу                */
#define CACHE_FLUSH_SIZE    ((size_t)(CACHE_FLUSH_MB) * 1024 * 1024)
#define LOG_PATH_LEN        256
#define BAR_WIDTH           36
#define SPEEDUP_WARN_THR    0.9
#define MAX_RESULTS         32
#define QS_THRESHOLD        8192        /* нижче → serial qsort                */
#define MAX_QS_TASKS        65536       /* макс. задач у черзі quicksort       */

/* Фіксовані конфігурації потоків (1 2 4 6 8) */
static const int THREAD_COUNTS[] = {1, 2, 4, 6, 8};
#define NUM_CONFIGS ((int)(sizeof(THREAD_COUNTS) / sizeof(THREAD_COUNTS[0])))

/* ══════════════════════════════════════════════════════════════════════
 *  Логування
 * ══════════════════════════════════════════════════════════════════════ */

typedef enum { LOG_INFO, LOG_WARN, LOG_ERROR } log_level_t;

static FILE *g_log = NULL;
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

/* ══════════════════════════════════════════════════════════════════════
 *  Інформація про CPU і кеш
 * ══════════════════════════════════════════════════════════════════════ */

static void log_cpu_info(void) {
    log_write(LOG_INFO, "──── Апаратне забезпечення ─────────────────────────────────────────");
    log_write(LOG_INFO, "Логічних ядер (OpenMP max): %d", omp_get_max_threads());

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
        if (!got_freq  && strncmp(line, "cpu MHz",    7) == 0)
            { snprintf(freq, sizeof(freq), "%.0f МГц", atof(val)); got_freq = 1; }
        if (got_model && got_cache && got_freq) break;
    }
    fclose(f);

    log_write(LOG_INFO, "Модель CPU  : %s", model);
    log_write(LOG_INFO, "Частота CPU : %s", freq);
    log_write(LOG_INFO, "Кеш (L2/L3) : %s", cache);
    log_write(LOG_INFO, "Буфер flush : %d МБ (очищення кешу між прогонами)", CACHE_FLUSH_MB);
    log_write(LOG_INFO, "Прогонів    : %d × %d конфіг × 2 методи = %d ітерацій",
              N_RUNS, NUM_CONFIGS, N_RUNS * NUM_CONFIGS * 2);
    log_write(LOG_INFO, "────────────────────────────────────────────────────────────────────");
}

/* ══════════════════════════════════════════════════════════════════════
 *  Очищення кешу між прогонами
 *
 *  Записуємо та зчитуємо буфер, що перевищує L3-кеш (~48 МБ),
 *  щоб витіснити дані попереднього прогону. Без цього перший прогін
 *  буде «холодним», решта — «теплими», що спотворює stddev.
 * ══════════════════════════════════════════════════════════════════════ */

static char *g_flush_buf = NULL;

static int flush_buf_init(void) {
    g_flush_buf = (char *)malloc(CACHE_FLUSH_SIZE);
    if (!g_flush_buf) return -1;
    memset(g_flush_buf, 0xAB, CACHE_FLUSH_SIZE);
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

/* ══════════════════════════════════════════════════════════════════════
 *  Структури статистики
 * ══════════════════════════════════════════════════════════════════════ */

typedef struct {
    double wall_time;
    double cpu_time;
    long   nvcsw, nivcsw, minflt, majflt;
} run_stat_t;

typedef struct {
    double mean_wall, stddev_wall, min_wall, max_wall, cv_wall;
    double mean_cpu, cpu_eff, mean_speedup;
    long   total_nvcsw, total_nivcsw, total_minflt, total_majflt;
} agg_stat_t;

typedef struct {
    char   method[80];
    int    threads;
    double elapsed, speedup;
} bench_result_t;

static bench_result_t g_results[MAX_RESULTS];
static int            g_result_count = 0;

/* ══════════════════════════════════════════════════════════════════════
 *  ASCII-діаграма прискорення
 * ══════════════════════════════════════════════════════════════════════ */

static void build_bar(char *buf, int buf_size, int bar_len) {
    const char BLK[] = "\xe2\x96\x88";
    int pos = 0;
    for (int i = 0; i < BAR_WIDTH && pos + 3 < buf_size; i++) {
        if (i < bar_len) { memcpy(buf + pos, BLK, 3); pos += 3; }
        else              { buf[pos++] = ' '; }
    }
    buf[pos] = '\0';
}

static void render_chart(FILE *out) {
    if (!g_result_count) return;
    double max_sp = 1.0;
    for (int i = 0; i < g_result_count; i++)
        if (g_results[i].speedup > max_sp) max_sp = g_results[i].speedup;

    char sep[74]; memset(sep, '-', 72); sep[72] = '\0';
    fprintf(out, "\n%s\n", sep);
    fprintf(out, "  ДІАГРАМА ПРИСКОРЕННЯ  (1 \xe2\x96\x88 = %.2fx)\n", max_sp / BAR_WIDTH);
    fprintf(out, "%s\n", sep);

    char prev[80] = "";
    char bar_buf[BAR_WIDTH * 3 + 4];
    for (int i = 0; i < g_result_count; i++) {
        bench_result_t *r = &g_results[i];
        if (strcmp(r->method, prev) != 0) {
            strncpy(prev, r->method, sizeof(prev) - 1);
            fprintf(out, "\n  [%s]\n", r->method);
        }
        int blen = (int)((r->speedup / max_sp) * BAR_WIDTH);
        if (blen < 1) blen = 1;
        build_bar(bar_buf, sizeof(bar_buf), blen);
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

/* ══════════════════════════════════════════════════════════════════════
 *  Відкриття / закриття лог-файлу
 * ══════════════════════════════════════════════════════════════════════ */

static int log_open(long n, const char *perf_status) {
    time_t t      = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(g_log_path, sizeof(g_log_path), "sort_bench_%Y%m%d_%H%M%S.log", tm);
    g_log = fopen(g_log_path, "w");
    if (!g_log) {
        fprintf(stderr, "WARN: не вдалося створити лог-файл '%s'\n", g_log_path);
        return -1;
    }
    char ts[32];
    current_timestamp(ts, sizeof(ts));
    fprintf(g_log,
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  Сортування OpenMP бенчмарк — сесія %s  ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n\n",
        ts);
    log_write(LOG_INFO, "Запуск програми");
    log_write(LOG_INFO, "Розмір масиву : %ld елементів (%.1f МБ)",
              n, (double)n * sizeof(int) / (1024.0 * 1024.0));
    log_write(LOG_INFO, "Прогонів      : %d", N_RUNS);
    log_write(LOG_INFO, "Конфігурації  : 1 2 4 6 8 потоків");
    log_write(LOG_INFO, "Perf stat     : %s", perf_status);
    return 0;
}

static void log_close(void) {
    if (!g_log) return;
    log_write(LOG_INFO, "Програму завершено успішно");
    fprintf(g_log, "\n══════════════════════════ EOF ══════════════════════════════\n");
    fclose(g_log);
    g_log = NULL;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Допоміжні функції виводу
 * ══════════════════════════════════════════════════════════════════════ */

static void print_separator(void) {
    printf("─────────────────────────────────────────────────────────────────\n");
}

static void print_header(void) {
    printf("\n");
    print_separator();
    printf("  Паралельне сортування масиву  ─  OpenMP бенчмарк\n");
    print_separator();
}

/* ══════════════════════════════════════════════════════════════════════
 *  Масив: генерація, скидання, перевірка
 * ══════════════════════════════════════════════════════════════════════ */

static int *g_src_array  = NULL;   /* оригінальний (незасортований)  */
static int *g_work_array = NULL;   /* робоча копія (сортується)       */
static int *g_tmp_array  = NULL;   /* буфер для merge-sort             */

static void array_generate(long n, unsigned seed) {
    srand(seed);
    for (long i = 0; i < n; i++) g_src_array[i] = rand();
}

static void array_reset(long n) {
    memcpy(g_work_array, g_src_array, (size_t)n * sizeof(int));
}

static int array_is_sorted(const int *arr, long n) {
    for (long i = 1; i < n; i++)
        if (arr[i] < arr[i - 1]) return 0;
    return 1;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Метод 1 — Паралельний Merge Sort з omp_lock_t (lock striping)
 *
 *  Фаза 1 (параллельна): кожен потік сортує свою частку через qsort().
 *  Фаза 2 (параллельна): дерево злиття за рівнями (log₂(nthreads) рівнів).
 *    Масив замків locks[] (розміром nthreads) — «Lock Striping»:
 *    кожна операція злиття захоплює locks[m % nthreads].
 *    Оскільки на одному рівні операції злиття незалежні (різні регіони
 *    tmp), вони завжди отримують різні замки → повний паралелізм.
 *    Замок захищає: memcpy + merge-запис у спільний буфер tmp.
 *    У складніших схемах (cross-level parallelism) замки були б
 *    суворо необхідними для запобігання перегонам запису.
 * ══════════════════════════════════════════════════════════════════════ */

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void sort_merge(int *arr, long n, int nthreads, int *tmp) {
    long chunk = (n + nthreads - 1) / nthreads;

    /* Масив замків: один на потенційний паралельний слот злиття */
    omp_lock_t *locks = (omp_lock_t *)malloc((size_t)nthreads * sizeof(omp_lock_t));
    if (!locks) {
        fprintf(stderr, "ERROR: malloc locks\n");
        return;
    }
    for (int i = 0; i < nthreads; i++) omp_init_lock(&locks[i]);

    /* ── Фаза 1: паралельне локальне сортування ─────────────── */
    #pragma omp parallel for num_threads(nthreads) schedule(static, 1)
    for (int t = 0; t < nthreads; t++) {
        long lo = (long)t * chunk;
        long hi = lo + chunk;
        if (hi > n) hi = n;
        if (lo < n) qsort(arr + lo, (size_t)(hi - lo), sizeof(int), cmp_int);
    }

    /* ── Фаза 2: паралельне дерево злиття ───────────────────── */
    for (long stride = chunk; stride < n; stride *= 2) {
        long stride2 = stride * 2;
        long nmerges = (n + stride2 - 1) / stride2;

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
        for (long m = 0; m < nmerges; m++) {
            long lo  = m * stride2;
            long mid = lo + stride;  if (mid > n) mid = n;
            long hi  = mid + stride; if (hi  > n) hi  = n;
            if (mid >= n) continue;

            /* Lock striping: вибір замка за індексом merge-слота */
            omp_lock_t *lk = &locks[m % (long)nthreads];
            omp_set_lock(lk);

            /* Копія лівої половини у спільний буфер (захищена замком) */
            memcpy(tmp + lo, arr + lo, (size_t)(mid - lo) * sizeof(int));
            long i = lo, j = mid, k = lo;
            while (i < mid && j < hi)
                arr[k++] = (tmp[i] <= arr[j]) ? tmp[i++] : arr[j++];
            while (i < mid) arr[k++] = tmp[i++];

            omp_unset_lock(lk);
        }
        /* Неявний бар'єр на кінці #pragma omp parallel for */
    }

    for (int i = 0; i < nthreads; i++) omp_destroy_lock(&locks[i]);
    free(locks);
}

/* ══════════════════════════════════════════════════════════════════════
 *  Метод 2 — Паралельний Quicksort (LIFO-черга задач) з omp_lock_t
 *
 *  Спільна черга задач (масив qs_task_t) захищена одним omp_lock_t.
 *  Кожен потік:
 *    а) блокує замок і витягує задачу зі стека;
 *    б) якщо len ≤ QS_THRESHOLD → qsort() (листова задача);
 *    в) інакше → 3-way partition, кладе дві підзадачі у стек.
 *
 *  Лічильник pending (під тим самим замком):
 *    iniціально = 1.  Розбиття: −1 + 2 = +1.  Лист: −1.
 *    Коли pending == 0 → черга порожня → завершення.
 *
 *  3-way partition (Dutch National Flag) коректно обробляє дублікати:
 *  рівні pivot елементи залишаються на місці, рекурсія лише на
 *  елементах, строго менших / більших за pivot.
 * ══════════════════════════════════════════════════════════════════════ */

typedef struct { int *arr; long n; } qs_task_t;

static qs_task_t  *g_qs_tasks   = NULL;
static int         g_qs_top     = 0;
static int         g_qs_pending = 0;
static omp_lock_t  g_qs_lock;

static int median3(int a, int b, int c) {
    if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
    if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
    return c;
}

/* 3-way partition: arr[0..lt) < pivot, arr[lt..gt) == pivot, arr[gt..n) > pivot */
static void partition3(int *arr, long n, long *lt_out, long *gt_out) {
    int pivot = median3(arr[0], arr[n / 2], arr[n - 1]);
    long lt = 0, eq = 0, gt = n;
    while (eq < gt) {
        if (arr[eq] < pivot) {
            int t = arr[lt]; arr[lt] = arr[eq]; arr[eq] = t;
            lt++; eq++;
        } else if (arr[eq] > pivot) {
            gt--;
            int t = arr[eq]; arr[eq] = arr[gt]; arr[gt] = t;
        } else {
            eq++;
        }
    }
    *lt_out = lt;
    *gt_out = gt;
}

static void sort_quick(int *arr, long n, int nthreads, int *tmp) {
    (void)tmp;  /* quicksort не потребує зовнішнього буфера */

    omp_init_lock(&g_qs_lock);
    g_qs_top     = 0;
    g_qs_pending = 1;
    g_qs_tasks[g_qs_top++] = (qs_task_t){arr, n};

    #pragma omp parallel num_threads(nthreads)
    {
        for (;;) {
            qs_task_t t;
            int have_task = 0;

            /* ── Критична секція: витяг задачі ── */
            omp_set_lock(&g_qs_lock);
            if (g_qs_top > 0) {
                t = g_qs_tasks[--g_qs_top];
                have_task = 1;
            } else if (g_qs_pending == 0) {
                omp_unset_lock(&g_qs_lock);
                break;          /* всі задачі виконано */
            }
            omp_unset_lock(&g_qs_lock);

            if (!have_task) {
                /* Черга тимчасово порожня, але є задачі в обробці */
                struct timespec ts = {0, 500};
                nanosleep(&ts, NULL);
                continue;
            }

            if (t.n <= QS_THRESHOLD) {
                /* ── Листова задача: серійне сортування ── */
                if (t.n > 1) qsort(t.arr, (size_t)t.n, sizeof(int), cmp_int);
                omp_set_lock(&g_qs_lock);
                g_qs_pending--;
                omp_unset_lock(&g_qs_lock);
            } else {
                /* ── Розбиття → підзадачі ── */
                long lt, gt;
                partition3(t.arr, t.n, &lt, &gt);
                /* pivot-рівні елементи arr[lt..gt) вже на фінальних позиціях */

                int n_left  = (lt > 1)       ? 1 : 0;
                int n_right = (t.n - gt > 1) ? 1 : 0;

                omp_set_lock(&g_qs_lock);
                if (g_qs_top + n_left + n_right <= MAX_QS_TASKS) {
                    if (n_left)
                        g_qs_tasks[g_qs_top++] = (qs_task_t){t.arr,      lt       };
                    if (n_right)
                        g_qs_tasks[g_qs_top++] = (qs_task_t){t.arr + gt, t.n - gt };
                    /*
                     * net зміна pending:
                     *   −1 (батьківська задача знята) + n_left + n_right (нові)
                     * Приклади:
                     *   0 дочірніх → −1  (рівні елементи закрили весь масив)
                     *   1 дочірня  →  0  (одна сторона тривіальна)
                     *   2 дочірніх → +1  (обидві сторони потребують роботи)
                     */
                    g_qs_pending += (n_left + n_right) - 1;
                    omp_unset_lock(&g_qs_lock);
                } else {
                    /* Черга переповнена: серійне сортування обох частин */
                    g_qs_pending--;
                    omp_unset_lock(&g_qs_lock);
                    if (lt > 1)
                        qsort(t.arr,      (size_t)lt,        sizeof(int), cmp_int);
                    if (t.n - gt > 1)
                        qsort(t.arr + gt, (size_t)(t.n - gt), sizeof(int), cmp_int);
                }
            }
        }
    }

    omp_destroy_lock(&g_qs_lock);
}

/* ══════════════════════════════════════════════════════════════════════
 *  Perf stat (опційний)
 * ══════════════════════════════════════════════════════════════════════ */

static int g_perf_available = 0;

static void check_perf_available(void) {
    g_perf_available = (system("which perf > /dev/null 2>&1") == 0);
}

/* Один представницький прогін під `perf stat`; результат → лог */
static void run_perf_stat(const char *method_name, int method_id,
                          int nthreads, long n, const char *exe) {
    if (!g_perf_available || !g_log) return;

    char cmd[768];
    snprintf(cmd, sizeof(cmd),
        "perf stat"
        " -e cycles,instructions,cache-misses,cache-references"
        ",branch-misses,branch-instructions"
        " -- %s --single-run %d %d %ld 2>&1",
        exe, method_id, nthreads, n);

    FILE *fp = popen(cmd, "r");
    if (!fp) {
        log_write(LOG_WARN, "popen perf stat не вдалося для [%s | %d п]",
                  method_name, nthreads);
        return;
    }
    log_write(LOG_INFO, "  ── perf stat [%s | %d потоків] ──────────────────────",
              method_name, nthreads);
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        line[strcspn(line, "\n")] = '\0';
        if (line[0] != '\0') log_write(LOG_INFO, "    %s", line);
    }
    pclose(fp);
}

/* ══════════════════════════════════════════════════════════════════════
 *  Функція запуску бенчмарку
 * ══════════════════════════════════════════════════════════════════════ */

typedef void (*sort_func_t)(int *, long, int, int *);

static void run_benchmark(const char *name, sort_func_t func, int method_id,
                          long n, double baseline_time, char **argv) {
    printf("\n  Метод: %s\n", name);
    log_write(LOG_INFO,
              "══ Метод: %-55s ══", name);
    log_write(LOG_INFO,
              "Синхронізація: omp_lock_t (mutex)  |  Прогонів: %d  |  Конфігурацій: %d",
              N_RUNS, NUM_CONFIGS);

    /* ── Заголовок таблиці ── */
    printf("  %-6s  %-10s  %-10s  %-10s  %-10s  %-8s  %-6s  %-9s\n",
           "Потоки", "Сер.час с", "Відхил. с", "Min с", "Max с",
           "Прискор.", "CV%", "CtxSw");
    printf("  %-6s  %-10s  %-10s  %-10s  %-10s  %-8s  %-6s  %-9s\n",
           "──────", "──────────", "──────────", "──────────", "──────────",
           "────────", "────", "─────────");

    char log_line[640];
    int  log_pos = 0, has_warn = 0;
    run_stat_t runs[N_RUNS];

    for (int ci = 0; ci < NUM_CONFIGS; ci++) {
        int t = THREAD_COUNTS[ci];
        printf("  [%2dп] прогрес: ", t); fflush(stdout);

        for (int r = 0; r < N_RUNS; r++) {
            /* Очищення кешу перед кожним прогоном */
            flush_cache();
            array_reset(n);

            struct rusage   ru0, ru1;
            struct timespec cpu0, cpu1;
            getrusage(RUSAGE_SELF, &ru0);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu0);

            double t_start = omp_get_wtime();
            func(g_work_array, n, t, g_tmp_array);
            double elapsed = omp_get_wtime() - t_start;

            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu1);
            getrusage(RUSAGE_SELF, &ru1);

            double cpu_sec = (cpu1.tv_sec  - cpu0.tv_sec)
                           + (cpu1.tv_nsec - cpu0.tv_nsec) * 1e-9;

            runs[r].wall_time = elapsed;
            runs[r].cpu_time  = cpu_sec;
            runs[r].nvcsw     = ru1.ru_nvcsw  - ru0.ru_nvcsw;
            runs[r].nivcsw    = ru1.ru_nivcsw - ru0.ru_nivcsw;
            runs[r].minflt    = ru1.ru_minflt - ru0.ru_minflt;
            runs[r].majflt    = ru1.ru_majflt - ru0.ru_majflt;

            printf("."); fflush(stdout);
        }
        printf(" OK\n");

        /* ── Агрегування статистики ── */
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
        }
        agg.mean_wall /= N_RUNS;
        agg.mean_cpu  /= N_RUNS;

        for (int r = 0; r < N_RUNS; r++) {
            double d = runs[r].wall_time - agg.mean_wall;
            agg.stddev_wall += d * d;
        }
        agg.stddev_wall  = sqrt(agg.stddev_wall / N_RUNS);
        agg.cv_wall      = agg.mean_wall > 0.0
                           ? agg.stddev_wall / agg.mean_wall * 100.0 : 0.0;
        agg.mean_speedup = baseline_time > 0.0
                           ? baseline_time / agg.mean_wall : 1.0;
        agg.cpu_eff      = (agg.mean_wall > 0.0 && t > 0)
                           ? agg.mean_cpu / (agg.mean_wall * t) * 100.0 : 0.0;

        /* ── Вивід таблиці ── */
        printf("  %-6d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-8.3f  %-6.2f  %ld/%ld\n",
               t,
               agg.mean_wall, agg.stddev_wall, agg.min_wall, agg.max_wall,
               agg.mean_speedup, agg.cv_wall,
               agg.total_nvcsw, agg.total_nivcsw);

        /* ── Детальний perf-рядок у лог ── */
        log_write(LOG_INFO,
                  "[%s | %2dп]  "
                  "час=%.4f±%.4fс [%.4f–%.4f]  "
                  "прискор=%.3fx  CPU-ефект=%.1f%%  CV=%.2f%%  "
                  "ctx_sw=%ld/%ld  page_faults=%ld+%ld",
                  name, t,
                  agg.mean_wall, agg.stddev_wall, agg.min_wall, agg.max_wall,
                  agg.mean_speedup, agg.cpu_eff, agg.cv_wall,
                  agg.total_nvcsw, agg.total_nivcsw,
                  agg.total_minflt, agg.total_majflt);

        /* ── Perf stat (один представницький прогін) ── */
        run_perf_stat(name, method_id, t, n, argv[0]);

        /* ── Компактний підсумок для методу ── */
        log_pos += snprintf(log_line + log_pos,
                            sizeof(log_line) - (size_t)log_pos,
                            "%dп=%.3fx(±%.3fs,CV=%.1f%%)",
                            t, agg.mean_speedup, agg.stddev_wall, agg.cv_wall);
        if (ci < NUM_CONFIGS - 1)
            log_pos += snprintf(log_line + log_pos,
                                sizeof(log_line) - (size_t)log_pos, " | ");

        if (t >= 2 && agg.mean_speedup < SPEEDUP_WARN_THR) has_warn = 1;

        /* ── Зберігаємо для діаграми ── */
        if (g_result_count < MAX_RESULTS) {
            bench_result_t *res = &g_results[g_result_count++];
            strncpy(res->method, name, sizeof(res->method) - 1);
            res->threads = t;
            res->elapsed = agg.mean_wall;
            res->speedup = agg.mean_speedup;
        }
    }

    log_write(LOG_INFO, "ПІДСУМОК %-55s → %s", name, log_line);
    if (has_warn)
        log_write(LOG_WARN,
                  "%-55s → прискорення < %.1f при ≥2 потоках: "
                  "можлива конкуренція або недостатньо ядер",
                  name, SPEEDUP_WARN_THR);
}

/* ══════════════════════════════════════════════════════════════════════
 *  Точка входу
 * ══════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {

    /* ── Режим --single-run (для perf stat) ───────────────────── *
     *  ./sort_omp --single-run <method_id> <threads> <n>          *
     *  Виконує один прогін сортування і завершує роботу.          *
     * ─────────────────────────────────────────────────────────── */
    if (argc >= 5 && strcmp(argv[1], "--single-run") == 0) {
        int  method_id = atoi(argv[2]);
        int  nthreads  = atoi(argv[3]);
        long n         = atol(argv[4]);

        if (flush_buf_init() != 0) return 1;

        int *arr = (int *)malloc((size_t)n * sizeof(int));
        int *tmp = (int *)malloc((size_t)n * sizeof(int));
        if (!arr || !tmp) { free(arr); free(tmp); flush_buf_free(); return 1; }

        srand(0x2B2B2B2B);
        for (long i = 0; i < n; i++) arr[i] = rand();
        flush_cache();

        if (method_id == 0) {
            sort_merge(arr, n, nthreads, tmp);
        } else {
            g_qs_tasks = (qs_task_t *)malloc((size_t)MAX_QS_TASKS * sizeof(qs_task_t));
            if (g_qs_tasks) sort_quick(arr, n, nthreads, NULL);
            free(g_qs_tasks); g_qs_tasks = NULL;
        }

        free(arr); free(tmp);
        flush_buf_free();
        return 0;
    }

    /* ── Основний режим ─────────────────────────────────────────── */
    long n = DEFAULT_N;
    if (argc >= 2) {
        long arg = atol(argv[1]);
        if (arg > 0) n = arg;
        else fprintf(stderr, "ERROR: розмір масиву > 0, використовую %ld\n",
                     DEFAULT_N);
    }

    check_perf_available();
    const char *perf_status = g_perf_available
        ? "доступний (perf stat на 1 прогін / конфігурацію → лог)"
        : "недоступний (тільки внутрішня статистика)";

    if (log_open(n, perf_status) == 0)
        printf("  Лог-файл: %s\n", g_log_path);

    log_cpu_info();

    /* Виділення буфера для очищення кешу */
    if (flush_buf_init() != 0) {
        fprintf(stderr, "WARN: не вдалося виділити %d МБ для очищення кешу\n",
                CACHE_FLUSH_MB);
        log_write(LOG_WARN,
                  "Буфер очищення кешу недоступний — результати можуть бути завищені");
    } else {
        log_write(LOG_INFO, "Буфер очищення кешу виділено: %d МБ", CACHE_FLUSH_MB);
    }

    /* Виділення масивів */
    g_src_array  = (int *)malloc((size_t)n * sizeof(int));
    g_work_array = (int *)malloc((size_t)n * sizeof(int));
    g_tmp_array  = (int *)malloc((size_t)n * sizeof(int));
    g_qs_tasks   = (qs_task_t *)malloc((size_t)MAX_QS_TASKS * sizeof(qs_task_t));

    if (!g_src_array || !g_work_array || !g_tmp_array || !g_qs_tasks) {
        fprintf(stderr, "ERROR: недостатньо пам'яті\n");
        log_write(LOG_ERROR, "Не вдалося виділити пам'ять для масивів/черги");
        return 1;
    }

    log_write(LOG_INFO, "Пам'ять: src + work + tmp = %.1f МБ; черга задач: %.1f МБ",
              (double)(3 * n) * sizeof(int) / (1024.0 * 1024.0),
              (double)MAX_QS_TASKS * sizeof(qs_task_t) / (1024.0 * 1024.0));

    /* Генерація вихідного масиву (один раз) */
    array_generate(n, 0xDEADBEEFU);

    /* ── Вивід шапки ── */
    print_header();
    printf("  Розмір масиву:  %ld елементів (%.1f МБ)\n",
           n, (double)n * sizeof(int) / (1024.0 * 1024.0));
    printf("  Прогонів:       %d (з очищенням кешу %d МБ перед кожним)\n",
           N_RUNS, CACHE_FLUSH_MB);
    printf("  Конфігурації:   ");
    for (int i = 0; i < NUM_CONFIGS; i++)
        printf("%d%s", THREAD_COUNTS[i], i < NUM_CONFIGS - 1 ? " " : "\n");
    printf("  OpenMP ядер:    %d\n", omp_get_max_threads());
    printf("  Perf stat:      %s\n", perf_status);
    print_separator();

    /* ── Базовий час (merge sort, 1 потік) ── */
    printf("\n  Вимірювання базового часу"
           " (1 потік, merge sort, %d прогонів)...\n", N_RUNS);
    double baseline_times[N_RUNS];
    printf("  прогрес: ");
    for (int r = 0; r < N_RUNS; r++) {
        flush_cache();
        array_reset(n);
        double ts = omp_get_wtime();
        sort_merge(g_work_array, n, 1, g_tmp_array);
        baseline_times[r] = omp_get_wtime() - ts;
        printf("."); fflush(stdout);
    }
    printf(" OK\n");

    double baseline = 0.0;
    for (int r = 0; r < N_RUNS; r++) baseline += baseline_times[r];
    baseline /= N_RUNS;
    printf("  Базовий час (середній): %.4f с\n", baseline);
    log_write(LOG_INFO, "Базовий час (1 потік, merge sort, N=%d): %.6f с",
              N_RUNS, baseline);

    /* Перевірка коректності merge sort */
    if (!array_is_sorted(g_work_array, n)) {
        fprintf(stderr, "ERROR: merge sort некоректний!\n");
        log_write(LOG_ERROR, "Перевірка коректності merge sort: ПРОВАЛЕНА");
        return 1;
    }
    log_write(LOG_INFO, "Перевірка коректності merge sort: OK");

    /* Перевірка коректності quicksort (невеликий масив) */
    {
        long nq = 200000L;
        int *qa = (int *)malloc((size_t)nq * sizeof(int));
        if (qa) {
            srand(0x1234ABCDU);
            for (long i = 0; i < nq; i++) qa[i] = rand();
            sort_quick(qa, nq, 2, NULL);
            int ok = array_is_sorted(qa, nq);
            log_write(ok ? LOG_INFO : LOG_ERROR,
                      "Перевірка коректності quicksort (%ld ел.): %s",
                      nq, ok ? "OK" : "ПРОВАЛЕНА");
            if (!ok) { free(qa); return 1; }
            free(qa);
        }
    }

    /* ── Бенчмарк двох методів ── */
    run_benchmark(
        "Merge Sort (omp_lock_t — lock striping на буфер tmp)",
        sort_merge, 0, n, baseline, argv);

    run_benchmark(
        "Quicksort  (omp_lock_t — захист черги задач LIFO)",
        sort_quick, 1, n, baseline, argv);

    /* ── ASCII-діаграма ── */
    print_chart();
    log_chart();

    /* ── Пояснення ── */
    printf("\n");
    print_separator();
    printf("  Пояснення:\n");
    printf("  • Merge Sort  — Фаза 1: кожен потік сортує qsort() свою частку.\n");
    printf("    Фаза 2: паралельне дерево злиття. Lock Striping: locks[m%%T]\n");
    printf("    захищає запис у спільний буфер tmp — mutex-патерн OpenMP.\n");
    printf("  • Quicksort   — LIFO-стек задач захищений omp_lock_t.\n");
    printf("    Потоки конкурують за задачі (push/pop під замком).\n");
    printf("    Pending-лічильник під тим самим замком = умова завершення.\n");
    printf("    3-way partition коректно обробляє масиви з дублікатами.\n");
    printf("  • CV%%         — коефіцієнт варіації: стабільність між прогонами.\n");
    printf("  • CtxSw      — перемикання контексту вол./прим. за %d прогонів.\n",
           N_RUNS);
    print_separator();
    printf("\n");

    /* ── Звільнення ресурсів ── */
    free(g_src_array);
    free(g_work_array);
    free(g_tmp_array);
    free(g_qs_tasks);
    flush_buf_free();
    log_close();
    return 0;
}
