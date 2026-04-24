/*
 * monte_carlo_omp.c
 *
 * Багатопотокове обчислення числа π методом Монте-Карло з використанням OpenMP.
 *
 * Метод: випадкові точки кидаються в одиничний квадрат [0,1]×[0,1].
 * Якщо точка потрапила в чверть одиничного кола (x²+y² ≤ 1):
 *
 *   π ≈ 4 × (кількість влучень) / (загальна кількість точок)
 *
 * Три варіанти синхронізації потоків:
 *   1) reduction  — автоматична редукція OpenMP (рекомендований спосіб)
 *   2) atomic     — кожен потік інкрементує спільний лічильник через atomic
 *   3) critical   — кожен потік накопичує локально, підсумок через critical
 *
 * Бенчмарк:
 *   Кожна конфігурація (метод × кількість потоків) запускається N_RUNS=20 разів.
 *   Перед кожним прогоном кеш процесора очищується великим буфером (~CACHE_FLUSH_MB),
 *   щоб результати не залежали від «теплого» кешу.
 *
 * Perf-подібна статистика (без root/perf):
 *   — середній і мінімальний/максимальний час (wall time)
 *   — стандартне відхилення часу і коефіцієнт варіації (CV%)
 *   — час CPU (CLOCK_PROCESS_CPUTIME_ID) і ефективність використання ядер
 *   — добровільні/примусові перемикання контексту (getrusage)
 *   — незначні/значні відмови сторінок (page faults)
 *
 * Логування:
 *   Файл mc_bench_РРРРММДД_ГГХВСС.log з інформацією про CPU/кеш,
 *   компактним рядком результатів на метод та ASCII-діаграмою.
 *
 * Збірка (Linux, GCC):
 *   gcc -O2 -fopenmp -Wall -o monte_carlo_omp monte_carlo_omp.c -lm
 *
 * Запуск:
 *   ./monte_carlo_omp [кількість_вибірок]
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

/* ── Константи ────────────────────────────────────────────────────────────── */

#define DEFAULT_SAMPLES     100000000L  /* 100 млн вибірок за замовчуванням    */
#define N_RUNS              20          /* кількість повторень кожного прогону  */
#define MAX_THREAD_STEPS    7           /* до 2^6 = 64 потоків у бенчмарку     */
#define LOG_PATH_LEN        256
#define MAX_RESULTS         ((MAX_THREAD_STEPS + 2) * 3)
#define BAR_WIDTH           36
#define SPEEDUP_WARN_THR    0.9

/* Розмір буфера для очищення кешу: має перевищувати L3-кеш процесора.
 * 48 МБ покривають більшість сучасних десктопних CPU (L3 ≤ 32 МБ).    */
#define CACHE_FLUSH_MB      48
#define CACHE_FLUSH_SIZE    ((size_t)(CACHE_FLUSH_MB) * 1024 * 1024)

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

    log_write(LOG_INFO, "Модель CPU:    %s", model);
    log_write(LOG_INFO, "Частота CPU:   %s", freq);
    log_write(LOG_INFO, "Кеш (L2/L3):   %s", cache);
    log_write(LOG_INFO, "Буфер flush:   %d МБ (очищення кешу між прогонами)", CACHE_FLUSH_MB);
    log_write(LOG_INFO, "Прогонів:      %d × кожна конфігурація", N_RUNS);
    log_write(LOG_INFO, "───────────────────────────────────────────────");
}

/* ── Очищення кешу між прогонами ─────────────────────────────────────────── *
 *                                                                              *
 * Записуємо і зчитуємо буфер, більший за L3-кеш, щоб витіснити робочі дані   *
 * бенчмарку. Без цього кроку перший прогін буде «холодним», а наступні 19 —   *
 * «теплими», що занижує стандартне відхилення і спотворює порівняння.         *
 * volatile + компіляторний бар'єр запобігають оптимізації.                    *
 * ─────────────────────────────────────────────────────────────────────────── */

static char *g_flush_buf = NULL;

static int flush_buf_init(void) {
    g_flush_buf = (char *)malloc(CACHE_FLUSH_SIZE);
    if (!g_flush_buf) return -1;
    memset(g_flush_buf, 0, CACHE_FLUSH_SIZE); /* торкаємося пам'яті одразу */
    return 0;
}

static void flush_buf_free(void) {
    free(g_flush_buf);
    g_flush_buf = NULL;
}

static void flush_cache(void) {
    if (!g_flush_buf) return;
    volatile char *p = (volatile char *)g_flush_buf;
    /* Прохід запису — витісняємо поточні дані */
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64)
        p[i] = (char)(i & 0xFF);
    /* Прохід читання — заповнюємо кеш нерелевантними даними */
    volatile long acc = 0;
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64)
        acc += p[i];
    (void)acc;
    /* Компіляторний бар'єр пам'яті */
    __asm__ volatile("" ::: "memory");
}

/* ── Структури статистики ─────────────────────────────────────────────────── */

/* Результат одного прогону */
typedef struct {
    double wall_time;   /* реальний час (секунди)                 */
    double cpu_time;    /* час CPU усіх потоків (секунди)         */
    double pi;          /* оцінка π                               */
    long   nvcsw;       /* добровільні перемикання контексту      */
    long   nivcsw;      /* примусові перемикання контексту        */
    long   minflt;      /* незначні відмови сторінок (page fault) */
    long   majflt;      /* значні відмови сторінок                */
} run_stat_t;

/* Агрегована статистика по N_RUNS прогонах */
typedef struct {
    double mean_wall;       /* середній реальний час               */
    double stddev_wall;     /* стандартне відхилення реального часу*/
    double min_wall;        /* мінімум                             */
    double max_wall;        /* максимум                            */
    double cv_wall;         /* коефіцієнт варіації, %              */
    double mean_cpu;        /* середній час CPU                    */
    double cpu_eff;         /* ефективність використання ядер, %   */
    double mean_speedup;    /* середнє прискорення                 */
    double mean_pi;         /* середня оцінка π                    */
    double mean_error;      /* середня абсолютна похибка           */
    long   total_nvcsw;     /* добровільних перемикань за N_RUNS   */
    long   total_nivcsw;    /* примусових перемикань за N_RUNS     */
    long   total_minflt;    /* незначних page faults за N_RUNS     */
    long   total_majflt;    /* значних page faults за N_RUNS       */
} agg_stat_t;

/* Зведений результат для ASCII-діаграми */
typedef struct {
    char   method[80];
    int    threads;
    double pi;
    double error;
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
    fprintf(out, "  ДІАГРАМА ПРИСКОРЕННЯ  (1 \xe2\x96\x88 = %.2fx)\n", max_sp / BAR_WIDTH);
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

/* ── Відкриття/закриття лог-файлу ────────────────────────────────────────── */

static int log_open(long n, int max_threads) {
    time_t t      = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(g_log_path, sizeof(g_log_path), "mc_bench_%Y%m%d_%H%M%S.log", tm);
    g_log = fopen(g_log_path, "w");
    if (!g_log) {
        fprintf(stderr, "WARN: не вдалося створити лог-файл '%s'\n", g_log_path);
        return -1;
    }
    char ts[32];
    current_timestamp(ts, sizeof(ts));
    fprintf(g_log,
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  Монте-Карло OpenMP бенчмарк — сесія %s  ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n",
            ts);
    fflush(g_log);
    log_write(LOG_INFO, "Запуск програми");
    log_write(LOG_INFO, "Вибірок:       %ld", n);
    log_write(LOG_INFO, "Макс. потоків: %d", max_threads);
    log_write(LOG_INFO, "Прогонів:      %d", N_RUNS);
    log_write(LOG_INFO, "Істинне π:     %.10f", M_PI);
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
    printf("  Метод Монте-Карло для обчислення π  ─  OpenMP бенчмарк\n");
    print_separator();
}

static double now_sec(void) { return omp_get_wtime(); }

/* ── Метод 1: reduction ───────────────────────────────────────────────────── */
double mc_pi_reduction(long n, int nthreads) {
    long inside = 0;
    #pragma omp parallel num_threads(nthreads) reduction(+:inside)
    {
        unsigned int seed = (unsigned int)(omp_get_thread_num() * 2654435761U + 1);
        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            if (x * x + y * y <= 1.0) inside++;
        }
    }
    return 4.0 * (double)inside / (double)n;
}

/* ── Метод 2: atomic ─────────────────────────────────────────────────────── */
double mc_pi_atomic(long n, int nthreads) {
    long inside = 0;
    #pragma omp parallel num_threads(nthreads)
    {
        unsigned int seed = (unsigned int)(omp_get_thread_num() * 2654435761U + 1);
        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            if (x * x + y * y <= 1.0) {
                #pragma omp atomic
                inside++;
            }
        }
    }
    return 4.0 * (double)inside / (double)n;
}

/* ── Метод 3: critical ────────────────────────────────────────────────────── */
double mc_pi_critical(long n, int nthreads) {
    long inside = 0;
    #pragma omp parallel num_threads(nthreads)
    {
        unsigned int seed = (unsigned int)(omp_get_thread_num() * 2654435761U + 1);
        long local_inside = 0;
        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            if (x * x + y * y <= 1.0) local_inside++;
        }
        #pragma omp critical
        { inside += local_inside; }
    }
    return 4.0 * (double)inside / (double)n;
}

/* ── Функція запуску бенчмарку ───────────────────────────────────────────── *
 *                                                                              *
 * Для кожної конфігурації потоків виконує N_RUNS прогонів:                    *
 *   1) flush_cache() — очищуємо кеш "холодними" даними                        *
 *   2) знімаємо getrusage/clock_gettime до і після                            *
 *   3) обчислюємо агреговану статистику                                        *
 * ─────────────────────────────────────────────────────────────────────────── */

typedef double (*mc_func_t)(long, int);

static void run_benchmark(const char *name, mc_func_t func,
                          long n, int *thread_counts, int num_configs,
                          double baseline_time) {
    /* Заголовок таблиці */
    printf("\n  Метод: %s\n", name);
    printf("  %-6s  %-12s  %-9s  %-9s  %-9s  %-9s  %-8s  %-6s  %-5s\n",
           "Потоки", "π (серед.)", "Сер.час с",
           "Відхил. с", "Min с", "Max с",
           "Прискор.", "CV%", "CtxSw");
    printf("  %-6s  %-12s  %-9s  %-9s  %-9s  %-9s  %-8s  %-6s  %-5s\n",
           "──────", "──────────", "─────────",
           "─────────", "─────", "─────",
           "────────", "────", "─────");

    /* Компактний рядок для лога */
    char log_line[640];
    int  log_pos  = 0;
    int  has_warn = 0;

    run_stat_t runs[N_RUNS];

    for (int ci = 0; ci < num_configs; ci++) {
        int t = thread_counts[ci];

        /* Прогрес-рядок під час виконання */
        printf("  [%2dп] прогрес: ", t);
        fflush(stdout);

        for (int r = 0; r < N_RUNS; r++) {
            /* ── Очищення кешу ── */
            flush_cache();

            /* ── Знімаємо стан до ── */
            struct rusage    ru0;
            struct timespec  cpu0;
            getrusage(RUSAGE_SELF, &ru0);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu0);

            /* ── Власне обчислення ── */
            double t_start = now_sec();
            double pi_est  = func(n, t);
            double elapsed = now_sec() - t_start;

            /* ── Знімаємо стан після ── */
            struct rusage   ru1;
            struct timespec cpu1;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu1);
            getrusage(RUSAGE_SELF, &ru1);

            double cpu_sec = (cpu1.tv_sec  - cpu0.tv_sec) +
                             (cpu1.tv_nsec - cpu0.tv_nsec) * 1e-9;

            runs[r].wall_time = elapsed;
            runs[r].cpu_time  = cpu_sec;
            runs[r].pi        = pi_est;
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
            agg.mean_pi      += runs[r].pi;
            agg.total_nvcsw  += runs[r].nvcsw;
            agg.total_nivcsw += runs[r].nivcsw;
            agg.total_minflt += runs[r].minflt;
            agg.total_majflt += runs[r].majflt;
            if (runs[r].wall_time < agg.min_wall) agg.min_wall = runs[r].wall_time;
            if (runs[r].wall_time > agg.max_wall) agg.max_wall = runs[r].wall_time;
        }
        agg.mean_wall /= N_RUNS;
        agg.mean_cpu  /= N_RUNS;
        agg.mean_pi   /= N_RUNS;

        for (int r = 0; r < N_RUNS; r++) {
            double d = runs[r].wall_time - agg.mean_wall;
            agg.stddev_wall += d * d;
        }
        agg.stddev_wall  = sqrt(agg.stddev_wall / N_RUNS);
        agg.cv_wall      = (agg.mean_wall > 0.0)
                           ? (agg.stddev_wall / agg.mean_wall * 100.0) : 0.0;
        agg.mean_speedup = (baseline_time > 0.0)
                           ? (baseline_time / agg.mean_wall) : 1.0;
        agg.mean_error   = fabs(agg.mean_pi - M_PI);
        /* CPU efficiency: скільки відсотків від ідеального паралелізму */
        agg.cpu_eff      = (agg.mean_wall > 0.0 && t > 0)
                           ? (agg.mean_cpu / (agg.mean_wall * t) * 100.0) : 0.0;

        /* ── Таблиця ── */
        printf("  %-6d  %-12.9f  %-9.4f  %-9.4f  %-9.4f  %-9.4f  %-8.3f  %-6.2f  %ld/%ld\n",
               t, agg.mean_pi,
               agg.mean_wall, agg.stddev_wall, agg.min_wall, agg.max_wall,
               agg.mean_speedup, agg.cv_wall,
               agg.total_nvcsw, agg.total_nivcsw);

        /* Детальний рядок perf у лог */
        log_write(LOG_INFO,
                  "[%s | %2dп]  π=%.9f  час=%.4f±%.4fс [%.4f–%.4f]  "
                  "прискор=%.3fx  CPU-ефект=%.1f%%  CV=%.2f%%  "
                  "ctx_sw=%ld/%ld  page_faults=%ld+%ld",
                  name, t,
                  agg.mean_pi, agg.mean_wall, agg.stddev_wall,
                  agg.min_wall, agg.max_wall,
                  agg.mean_speedup, agg.cpu_eff, agg.cv_wall,
                  agg.total_nvcsw, agg.total_nivcsw,
                  agg.total_minflt, agg.total_majflt);

        /* Накопичуємо компактний рядок-підсумок для методу */
        log_pos += snprintf(log_line + log_pos,
                            sizeof(log_line) - (size_t)log_pos,
                            "%dп=%.3fx(±%.3fs,CV=%.1f%%)",
                            t, agg.mean_speedup, agg.stddev_wall, agg.cv_wall);
        if (ci < num_configs - 1)
            log_pos += snprintf(log_line + log_pos,
                                sizeof(log_line) - (size_t)log_pos, " | ");

        if (t >= 2 && agg.mean_speedup < SPEEDUP_WARN_THR) has_warn = 1;

        /* Зберігаємо для візуалізатора */
        if (g_result_count < MAX_RESULTS) {
            bench_result_t *res = &g_results[g_result_count++];
            strncpy(res->method, name, sizeof(res->method) - 1);
            res->threads = t;
            res->pi      = agg.mean_pi;
            res->error   = agg.mean_error;
            res->elapsed = agg.mean_wall;
            res->speedup = agg.mean_speedup;
        }
    }

    /* ── Один підсумковий рядок на метод ── */
    log_write(LOG_INFO, "ПІДСУМОК %-40s → %s", name, log_line);
    if (has_warn)
        log_write(LOG_WARN,
                  "%-40s → прискорення < %.1f при ≥2 потоках: "
                  "можлива конкуренція або мало ядер CPU",
                  name, SPEEDUP_WARN_THR);
}

/* ── Точка входу ─────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    long n = DEFAULT_SAMPLES;
    if (argc >= 2) {
        long arg = atol(argv[1]);
        if (arg > 0) n = arg;
        else fprintf(stderr, "ERROR: вибірок має бути > 0, використовую %ld\n",
                     DEFAULT_SAMPLES);
    }

    int max_threads = omp_get_max_threads();

    if (log_open(n, max_threads) == 0)
        printf("  Лог-файл: %s\n", g_log_path);

    log_cpu_info();

    /* Ініціалізуємо буфер очищення кешу */
    if (flush_buf_init() != 0) {
        fprintf(stderr, "WARN: не вдалося виділити %d МБ для очищення кешу\n",
                CACHE_FLUSH_MB);
        log_write(LOG_WARN, "Буфер очищення кешу недоступний — результати можуть бути завищені");
    } else {
        log_write(LOG_INFO, "Буфер очищення кешу виділено: %d МБ", CACHE_FLUSH_MB);
    }

    /* Список конфігурацій потоків: степені двійки + 3/4*max + max */
    int thread_counts[MAX_THREAD_STEPS + 2];
    int num_configs = 0;

    for (int t = 1; t <= max_threads && num_configs < MAX_THREAD_STEPS; t *= 2)
        thread_counts[num_configs++] = t;

    if (max_threads > 2) {
        int tq = max_threads * 3 / 4;
        if (tq % 2 != 0) tq++;
        if (tq > 1 && tq < max_threads) {
            int dup = 0;
            for (int i = 0; i < num_configs; i++)
                if (thread_counts[i] == tq) { dup = 1; break; }
            if (!dup && num_configs < MAX_THREAD_STEPS + 1) {
                int pos = num_configs;
                for (int i = 0; i < num_configs; i++)
                    if (thread_counts[i] > tq) { pos = i; break; }
                memmove(&thread_counts[pos+1], &thread_counts[pos],
                        (size_t)(num_configs - pos) * sizeof(int));
                thread_counts[pos] = tq;
                num_configs++;
            }
        }
    }
    if (thread_counts[num_configs-1] != max_threads &&
        num_configs < MAX_THREAD_STEPS + 2)
        thread_counts[num_configs++] = max_threads;

    print_header();
    printf("  Вибірок:        %ld\n", n);
    printf("  Прогонів:       %d (з очищенням кешу %d МБ перед кожним)\n",
           N_RUNS, CACHE_FLUSH_MB);
    printf("  Макс. потоків:  %d\n", max_threads);
    printf("  Конфігурації:   ");
    for (int i = 0; i < num_configs; i++)
        printf("%d%s", thread_counts[i], i < num_configs-1 ? " " : "\n");
    printf("  Істинне π:      %.9f\n", M_PI);
    print_separator();

    /* Базовий час — теж з очищенням, щоб бути чесними */
    printf("\n  Вимірювання базового часу (1 потік, reduction, %d прогонів)...\n",
           N_RUNS);
    double baseline_times[N_RUNS];
    printf("  прогрес: ");
    for (int r = 0; r < N_RUNS; r++) {
        flush_cache();
        double ts = now_sec();
        mc_pi_reduction(n, 1);
        baseline_times[r] = now_sec() - ts;
        printf("."); fflush(stdout);
    }
    printf(" OK\n");

    double baseline = 0.0;
    for (int r = 0; r < N_RUNS; r++) baseline += baseline_times[r];
    baseline /= N_RUNS;

    printf("  Базовий час (середній): %.4f с\n", baseline);
    log_write(LOG_INFO, "Базовий час (1 потік, N=%d): %.4f с", N_RUNS, baseline);

    /* ── Бенчмарк трьох методів ── */
    run_benchmark("reduction (рекомендується)",
                  mc_pi_reduction, n, thread_counts, num_configs, baseline);

    run_benchmark("atomic (інкремент у циклі)",
                  mc_pi_atomic, n, thread_counts, num_configs, baseline);

    run_benchmark("critical (локальне накопичення + critical)",
                  mc_pi_critical, n, thread_counts, num_configs, baseline);

    /* ── Візуалізація ── */
    print_chart();
    log_chart();

    printf("\n");
    print_separator();
    printf("  Пояснення:\n");
    printf("  • reduction — OpenMP ділить лічильник на приватні копії;\n");
    printf("                немає конкуренції у циклі → максимальна швидкість.\n");
    printf("  • atomic    — атомарний інкремент у кожній ітерації циклу;\n");
    printf("                висока конкуренція потоків → повільніше.\n");
    printf("  • critical  — локальна сума, один critical-блок наприкінці;\n");
    printf("                серіалізація мінімальна → близько до reduction.\n");
    printf("  • CV%%        — коефіцієнт варіації: стабільність між прогонами.\n");
    printf("  • CtxSw     — перемикання контексту вол./прим. за %d прогонів.\n",
           N_RUNS);
    print_separator();
    printf("\n");

    flush_buf_free();
    log_close();
    return 0;
}
