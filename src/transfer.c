#include "transfer.h"
#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <errno.h>
#include <sys/stat.h>

#define min(a, b) (((a) > (b)) ? (b) : (a))

// --- このファイル内でのみ使用する関数のプロトタイプ宣言 ---
static void *worker_thread_func(void *arg);
static void perform_sftp_transfer_for_chunk(sftp_session sftp, ssh_session session, thread_data_t *data, int chunk_num);
static ssize_t read_to_buf(void *buffer, size_t count, void *user_data);
static int nr_cpus();
static int set_thread_affinity(pthread_t tid, int core);

/**
 * @brief ワーカースレッドの生成・監視・終了を管理するスレッド
 */
void *transfer_manager_thread(void *arg) {
    transfer_args_t *args = (transfer_args_t *)arg;
    pthread_t worker_threads[MAX_THREADS];
    int created_thread_count = 0;
    int num_cpus = nr_cpus();

    while (chunk_count_gl < chunk_last_count_gl) {
        while (created_thread_count < cur_thread_gl && created_thread_count < MAX_THREADS) {
            int i = created_thread_count;
            thread_data[i].local_file = args->local_file;
            thread_data[i].remote_file = args->remote_file;
            thread_data[i].host = args->host;
            thread_data[i].user = args->user;
            thread_data[i].key_path = NULL;
            thread_data[i].passphrase = args->passphrase;
            thread_data[i].chunk_size = args->chunk_size;
            thread_data[i].copied_bytes = 0;
            thread_data[i].latency_count = 0;
            thread_data[i].started_flag = 0;

            if (pthread_create(&worker_threads[i], NULL, worker_thread_func, (void *)&thread_data[i]) != 0) {
                perror("pthread_create for worker failed");
            } else {
                __sync_fetch_and_add(&active_thread_count_gl, 1);
                if (num_cpus > 0) {
                    set_thread_affinity(worker_threads[i], i % num_cpus);
                }
                created_thread_count++;
            }
        }
        usleep(100000);
    }

    for (int i = 0; i < created_thread_count; i++) {
        pthread_join(worker_threads[i], NULL);
    }

    return NULL;
}


/**
 * @brief ワーカースレッドのエントリーポイント
 */
static void *worker_thread_func(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    ssh_session session = init_ssh_session(data->host, data->user, data->passphrase);

    sem_wait(&sem_for_maxstartups);
    connect_ssh_session(session);
    authenticate_ssh_session(session, data->key_path, data->passphrase);
    sftp_session sftp = init_sftp_session(session);
    sem_post(&sem_for_maxstartups);

    while (1) {
        int my_chunk_num;

        pthread_mutex_lock(&transfer_mutex);
        if (chunk_count_gl < chunk_last_count_gl) {
            my_chunk_num = chunk_count_gl++;
        } else {
            pthread_mutex_unlock(&transfer_mutex);
            break;
        }
        pthread_mutex_unlock(&transfer_mutex);
        
        sem_wait(&sem_for_optimization);
        perform_sftp_transfer_for_chunk(sftp, session, data, my_chunk_num);
        sem_post(&sem_for_optimization);
    }

    sftp_free(sftp);
    ssh_disconnect(session);
    ssh_free(session);
    
    __sync_fetch_and_sub(&active_thread_count_gl, 1);
    
    return NULL;
}


/**
 * @brief 1チャンク分のデータ転送を実際に行う内部関数
 */
static void perform_sftp_transfer_for_chunk(sftp_session sftp, ssh_session session, thread_data_t *data, int chunk_num) {
    sftp_file file = sftp_open(sftp, data->remote_file, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
    if (file == NULL) {
        fprintf(stderr, "\nThread %ld: sftp_open failed: %s\n", (long)pthread_self(), ssh_get_error(session));
        return;
    }

    int local_fd = open(data->local_file, O_RDONLY);
    if (local_fd < 0) {
        fprintf(stderr, "\nThread %ld: open local file failed: %s\n", (long)pthread_self(), strerror(errno));
        sftp_close(file);
        return;
    }

    off_t file_size = (fake_file_size_gl != 0) ? fake_file_size_gl : real_file_size_gl;
    off_t start_offset = data->chunk_size * chunk_num;
    off_t chunk_len = (chunk_num != chunk_last_count_gl - 1) ? data->chunk_size : (file_size - start_offset);

    if (chunk_len <= 0) {
        close(local_fd);
        sftp_close(file);
        return;
    }

    lseek(local_fd, start_offset, SEEK_SET);
    sftp_seek64(file, start_offset);

    ssize_t remaind = chunk_len, thrown = chunk_len;
    struct {
        uint32_t id;
        ssize_t len;
        struct timeval start_time;
    } reqs[DEFAULT_NR_AHEAD];
    
    int idx;
    for (idx = 0; idx < DEFAULT_NR_AHEAD && thrown > 0; idx++) {
        reqs[idx].len = min(thrown, DEFAULT_BUF_SZ);
        reqs[idx].len = sftp_async_write(file, read_to_buf, reqs[idx].len, &local_fd, &reqs[idx].id);
        if (measure_transaction_latency_gl) {
            gettimeofday(&reqs[idx].start_time, NULL);
        }

        if (reqs[idx].len < 0) {
            fprintf(stderr, "\nsftp_async_write: %s\n", ssh_get_error(session));
            goto cleanup;
        }
        thrown -= reqs[idx].len;
    }

    for (idx = 0; remaind > 0; idx = (idx + 1) % DEFAULT_NR_AHEAD) {
        ssize_t transferred_len = reqs[idx].len;
        if (sftp_async_write_end(file, reqs[idx].id, 1) != SSH_OK) {
            fprintf(stderr, "\nsftp_async_write_end: %s\n", ssh_get_error(session));
            goto cleanup;
        }

        // ---- 追加: cmd_time_usec の記録 ----
        if (measure_transaction_latency_gl) {
            struct timeval current_time;
            gettimeofday(&current_time, NULL);
            int cmd_time_usec = (current_time.tv_sec - reqs[idx].start_time.tv_sec) * 1000000 +
                        (current_time.tv_usec - reqs[idx].start_time.tv_usec);

            int pos = data->latency_count % RING_BUF;
            data->a_latency_usec_buffer[pos] = cmd_time_usec;
            data->a_transferred_bytes_buffer[pos] = transferred_len;
            data->latency_count++;
        }
        // ------------------------------


        remaind -= transferred_len;
        // --- 修正点: アトミックな加算に変更 ---
        __sync_fetch_and_add(&data->copied_bytes, transferred_len);

        if (thrown > 0) {
            reqs[idx].len = min(thrown, DEFAULT_BUF_SZ);
            reqs[idx].len = sftp_async_write(file, read_to_buf, reqs[idx].len, &local_fd, &reqs[idx].id);
            if (measure_transaction_latency_gl) {
                gettimeofday(&reqs[idx].start_time, NULL);
            }

            if (reqs[idx].len < 0) {
                fprintf(stderr, "\nsftp_async_write: %s\n", ssh_get_error(session));
                goto cleanup;
            }
            thrown -= reqs[idx].len;
        }
    }

cleanup:
    close(local_fd);
    sftp_close(file);
}

// (以下、read_to_buf, nr_cpus, set_thread_affinity の実装は変更なし)
static ssize_t read_to_buf(void *buffer, size_t count, void *user_data) {
    int fd = *(int *)user_data;
    return read(fd, buffer, count);
}
static int nr_cpus() {
    cpu_set_t cpu_set;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set) == 0) return CPU_COUNT(&cpu_set);
    return -1;
}
static int set_thread_affinity(pthread_t tid, int core) {
    cpu_set_t target_cpu_set;
    CPU_ZERO(&target_cpu_set);
    CPU_SET(core, &target_cpu_set);
    if (pthread_setaffinity_np(tid, sizeof(target_cpu_set), &target_cpu_set) < 0) {
        fprintf(stderr, "Failed to set thread affinity for core %d\n", core);
        return -1;
    }
    return 0;
}
