#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <fcntl.h>
#include <unistd.h>
#include <libgen.h>
#include <termios.h>
#include <errno.h>
#include <sys/time.h>

#include "common.h"
#include "monitor.h"
#include "optimizer.h"
#include "transfer.h"

// --- グローバル変数の実体定義 ---
int cur_thread_gl = 1;
int opt_algo_gl = 0;
int opt_dur_gl = 5;
int stop_thread_inc_gl = 0;
bool measure_transaction_latency_gl = false;
int g_sftp_cmd_get_count = 0;
int sftp_cmd_worst_case_latency = 0;
int exp_opt_h_gl = 0;
volatile int active_thread_count_gl = 0;
int transfer_complete_gl = 0;
int flag_nr_conn_changed_gl = 0;
int g_thread_adjust_state = 0;
thread_data_t *thread_data = NULL;
int chunk_count_gl = 0;
int chunk_last_count_gl;
off_t fake_file_size_gl = 0;
off_t real_file_size_gl = 0;

pthread_mutex_t transfer_mutex;
sem_t sem_for_maxstartups;
sem_t sem_for_optimization;

// --- このファイル内でのみ使用するヘルパー関数 ---
static off_t parse_file_size(const char *str);
static char *split_user_host_path(const char *s, char **userp, char **hostp, char **pathp);

int main(int argc, char *argv[]) {
    int num_threads = 16;
    off_t chunk_size = (16 << 20);
    
    int opt;
    while ((opt = getopt(argc, argv, "n:a:e:s:f:c:t:h:")) != -1) {
        switch (opt) {
            case 'n': num_threads = atoi(optarg); cur_thread_gl = num_threads; break;
            case 'a': opt_algo_gl = atoi(optarg); break;
            case 'e': stop_thread_inc_gl = atoi(optarg); break;
            case 's': opt_dur_gl = atoi(optarg); break;
            case 'f': fake_file_size_gl = parse_file_size(optarg); break;
            case 'c': measure_transaction_latency_gl = true; g_sftp_cmd_get_count = atoi(optarg); break;
            case 't': sftp_cmd_worst_case_latency = atoi(optarg); break;
            case 'h': exp_opt_h_gl = atoi(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [options] <local_file> <user@host:remote_file>\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (optind + 2 != argc) {
        fprintf(stderr, "Usage: %s [options] <local_file> <user@host:remote_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *local_file = argv[optind];
    char *user = NULL, *host = NULL, *remote_file = NULL;
    char *input_str = split_user_host_path(argv[optind + 1], &user, &host, &remote_file);
    if (!input_str) {
        exit(EXIT_FAILURE);
    }
    const char *passphrase_ptr = getenv("SSH_AUTH_PASSPHRASE");

    int local_fd = open(local_file, O_RDONLY);
    if (local_fd < 0) {
        perror("Failed to open local file");
        free(input_str);
        exit(EXIT_FAILURE);
    }
    real_file_size_gl = lseek(local_fd, 0, SEEK_END);
    close(local_fd);
    
    off_t file_size_to_use = (fake_file_size_gl != 0) ? fake_file_size_gl : real_file_size_gl;
    if (file_size_to_use > 0 && chunk_size > 0) {
        chunk_last_count_gl = (file_size_to_use + chunk_size - 1) / chunk_size;
    } else {
        chunk_last_count_gl = 0;
    }

    thread_data = calloc(MAX_THREADS, sizeof(thread_data_t));
    if (thread_data == NULL) {
        fprintf(stderr, "Failed to allocate memory for thread data\n");
        free(input_str);
        exit(EXIT_FAILURE);
    }

    pthread_mutex_init(&transfer_mutex, NULL);
    sem_init(&sem_for_maxstartups, 0, 8);
    sem_init(&sem_for_optimization, 0, cur_thread_gl);

    // fprintf(stderr, "Starting transfer...\n");
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    pthread_t monitor_thread, optimizer_thread, transfer_manager;
    
    transfer_args_t trans_args = {
        .local_file = local_file,
        .remote_file = remote_file,
        .host = host,
        .user = user,
        .passphrase = passphrase_ptr,
        .chunk_size = chunk_size
    };

    pthread_create(&monitor_thread, NULL, monitor_transfer, NULL);
    if (opt_algo_gl > 0) {
        pthread_create(&optimizer_thread, NULL, dynamic_transfer_threads_management, NULL);
    }
    pthread_create(&transfer_manager, NULL, transfer_manager_thread, &trans_args);

    pthread_join(transfer_manager, NULL);
    gettimeofday(&end_time, NULL);

    // --- 最終レポートの計算と表示 ---
    transfer_complete_gl = 1; // 他のスレッドに完了を通知

    // 最後の正確な統計情報を取得
    mscptest_stats final_stats;
    final_stats.done = 0;
    for (int i = 0; i < MAX_THREADS; i++) {
        if(thread_data[i].copied_bytes > 0) {
            final_stats.done += thread_data[i].copied_bytes;
        }
    }
    final_stats.total = (fake_file_size_gl != 0) ? fake_file_size_gl : real_file_size_gl;
    
    double total_elapsed_time = (end_time.tv_sec - start_time.tv_sec) + 
                                (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    double final_throughput = (total_elapsed_time > 0) ? 
                              ((double)final_stats.done / 1024.0 / 1024.0 / total_elapsed_time) : 0;
    double final_done_gb = (double)final_stats.done / 1000000000.0;

    pthread_join(monitor_thread, NULL); // monitorスレッドが画面クリアするのを待つ
    if (opt_algo_gl > 0) {
        sem_post(&sem_for_optimization);
        pthread_join(optimizer_thread, NULL);
    }
    
    if (!measure_transaction_latency_gl) {   // -c オプションを使ったときは、最終表示をみせない
        fprintf(stdout, "\n====================== Final Report ======================\n");
        fprintf(stdout, " Total Transferred: %.2f GB (%zu bytes)\n", final_done_gb, final_stats.done);
        fprintf(stdout, " Total File Size  : %.2f GB (%zu bytes)\n", (double)final_stats.total / 1000000000.0, final_stats.total);
        fprintf(stdout, " Total Elapsed Time: %.3f seconds\n", total_elapsed_time);
        fprintf(stdout, " Average Throughput: %.2f MB/s\n", final_throughput);
        if(final_stats.total > 0) {
            fprintf(stdout, " Completion Ratio : %.2f%%\n", 100.0 * final_stats.done / final_stats.total);
        }
        fprintf(stdout, "==========================================================\n");
    }

    pthread_mutex_destroy(&transfer_mutex);
    sem_destroy(&sem_for_maxstartups);
    sem_destroy(&sem_for_optimization);
    
    free(thread_data);
    free(input_str);

    return 0;
}

// (以下、parse_file_size, split_user_host_path の実装は変更なし)
static off_t parse_file_size(const char *str) {
    char unit = '\0';
    long long size_val = 0;
    sscanf(str, "%lld%c", &size_val, &unit);
    off_t size = size_val;
    switch (unit) {
        case 'G': case 'g': size *= 1024 * 1024 * 1024; break;
        case 'M': case 'm': size *= 1024 * 1024; break;
        case 'K': case 'k': size *= 1024; break;
    }
    return size;
}

static char *strip_brackets(char *host) {
    size_t len = strlen(host);
    if (len > 1 && host[0] == '[' && host[len - 1] == ']') {
        host[len - 1] = '\0';
        return host + 1;
    }
    return host;
}

static char *split_user_host_path(const char *s, char **userp, char **hostp, char **pathp) {
    char *tmp, *cp, *user = NULL, *host = NULL, *path = NULL;
    if (!(tmp = strdup(s))) {
        perror("strdup");
        return NULL;
    }
    path = tmp;
    cp = strrchr(tmp, ':');
    if (cp != NULL) {
        char *bracket = strrchr(tmp, ']');
        if (bracket == NULL || cp > bracket) {
            *cp = '\0';
            path = cp + 1;
        }
    }
    host = tmp;
    cp = strchr(tmp, '@');
    if (cp != NULL) {
        *cp = '\0';
        user = tmp;
        host = cp + 1;
    }
    *userp = user;
    *hostp = host ? strip_brackets(host) : NULL;
    *pathp = path;
    return tmp;
}
