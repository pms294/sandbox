#include "optimizer.h"
#include "monitor.h" // mscptest_get_stats を使うため

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>


// --- このファイル内でのみ使用するマクロとstatic変数 ---
#define EPSILON 1e-9
#define max(a, b) (((a) > (b)) ? (a) : (a))
#define min(a, b) (((a) < (b)) ? (a) : (b))


// ベイズ最適化用
#define MAX_DATA 100
#define CONNECTION_MIN 1
#define CONNECTION_MAX 50
#define ITERATIONS 10

typedef struct {
    double X[MAX_DATA];
    double Y[MAX_DATA];
    int size;
} DataSet;

static pthread_mutex_t bo_lock = PTHREAD_MUTEX_INITIALIZER;

// アルゴリズムの状態フラグ
static bool MN_algorithm_end_flag = false;
static bool HC_algorithm_end_flag = false;
static bool HC_algorithm_direction_flag = false; // 0: 増加, 1: 減少
static bool GD_algorithm_direction_flag = false; // 0: 増加, 1: 減少
static int learning_factor = 1;
static bool learning_factor_reset = false;
static double delta = 1.0;
static int temp = 0;
static int steady_cnt = 0; // 適応的調整アルゴリズム用のカウンタ

// --- プロトタイプ宣言 ---
static void threads_manage_BO();
static void *thread_manage_sftp_cmd();
static void GD_algorithm_3(double prev_thr, double cur_thr, int prev_conn, int cur_conn);
static void adaptive_adjustment(double prev_thr, double cur_thr);


// --- 最適化アルゴリズムの実装 ---

static void simple_increase() {
    if (stop_thread_inc_gl > 0 && cur_thread_gl >= stop_thread_inc_gl) {
        return;
    }
    if (cur_thread_gl < MAX_THREADS) {
        cur_thread_gl++;
    }
}

static void MN_algorithm(double prev_thr, double cur_thr) {
    if (MN_algorithm_end_flag) return;

    if (prev_thr > cur_thr && cur_thread_gl > 1) {
        MN_algorithm_end_flag = true;
    } else if (cur_thread_gl < MAX_THREADS) {
        cur_thread_gl++;
    }
}

static void HC_algorithm(double prev_thr, double cur_thr) {
    if (HC_algorithm_end_flag) return;

    double threshold = 0.03;
    double rate_of_change = (fabs(cur_thr) > EPSILON) ? (cur_thr - prev_thr) / cur_thr : 0.0;

    if (!HC_algorithm_direction_flag) { // 増加方向
        if (rate_of_change > threshold && cur_thread_gl < MAX_THREADS) {
            cur_thread_gl++;
        } else if (rate_of_change < -threshold && cur_thread_gl > 1) {
            cur_thread_gl--;
            HC_algorithm_direction_flag = true;
        } else {
            HC_algorithm_end_flag = true;
        }
    } else { // 減少方向
        if (rate_of_change > threshold && cur_thread_gl > 1) {
            cur_thread_gl--;
        } else if (rate_of_change < -threshold && cur_thread_gl < MAX_THREADS) {
            cur_thread_gl++;
            HC_algorithm_direction_flag = false;
        } else {
            HC_algorithm_end_flag = true;
        }
    }
}

static void GD_algorithm(double prev_thr, double cur_thr, int prev_conn, int cur_conn) {
    if (cur_conn == prev_conn) return;

    double gradient = (double)(cur_thr - prev_thr) / (cur_conn - prev_conn);
    if (!learning_factor_reset && fabs(prev_thr) > EPSILON) {
        delta = gradient / prev_thr;
    } else {
        learning_factor = 1;
        learning_factor_reset = false;
        delta = 1.0;
    }
    
    temp = (int)ceil(fabs(delta) * learning_factor);
    if (temp == 0) temp = 1;

    if (!GD_algorithm_direction_flag) { // 増加方向
        if (delta > 0.03) {
            cur_thread_gl += temp;
            learning_factor *= 2;
        } else {
            GD_algorithm_direction_flag = true;
            learning_factor = 1;
            cur_thread_gl -= temp;
            learning_factor *= 2;
        }
    } else { // 減少方向
        if (delta < -0.03) {
            cur_thread_gl -= temp;
            learning_factor *= 2;
        } else {
            GD_algorithm_direction_flag = false;
            learning_factor = 1;
            cur_thread_gl += temp;
            learning_factor *= 2;
        }
    }
    if (cur_thread_gl < 1) cur_thread_gl = 1;
    if (cur_thread_gl > MAX_THREADS) cur_thread_gl = MAX_THREADS;
}

static void adaptive_adjustment(double prev_thr, double cur_thr) {
    double threshold = 0.03; // 3%
    double rate_of_change = (fabs(prev_thr) > EPSILON) ? (cur_thr - prev_thr) / prev_thr : 0.0;

    if (rate_of_change > threshold) {
        if (cur_thread_gl < MAX_THREADS) cur_thread_gl++;
        steady_cnt = 0;
    } else if (rate_of_change < -threshold) {
        if (cur_thread_gl > 1) cur_thread_gl--;
        steady_cnt = 0;
    } else {
        steady_cnt++;
        if (steady_cnt >= 5) {
            if (cur_thread_gl < MAX_THREADS) cur_thread_gl++;
            steady_cnt = 0;
        }
    }
}

// --- ベイズ最適化関連のヘルパー関数 ---

static double rbf_kernel(double x1, double x2, double length_scale) {
    double diff = x1 - x2;
    return exp(-(diff * diff) / (2 * length_scale * length_scale));
}

static void compute_kernel_matrix(double *X, gsl_matrix *K, int n, double length_scale) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            gsl_matrix_set(K, i, j, rbf_kernel(X[i], X[j], length_scale));
        }
    }
}

static double predict(double x_new, double *X, double *Y, int n, double length_scale, double *sigma) {
    gsl_set_error_handler_off();
    if (n < 2) {
        return rand() % (CONNECTION_MAX - CONNECTION_MIN + 1) + CONNECTION_MIN;
    }

    gsl_matrix *K = gsl_matrix_alloc(n, n);
    compute_kernel_matrix(X, K, n, length_scale);

    double jitter = 1e-6;
    for (int i = 0; i < n; i++) {
        gsl_matrix_set(K, i, i, gsl_matrix_get(K, i, i) + jitter);
    }

    gsl_matrix *K_copy = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(K_copy, K);
    gsl_permutation *p = gsl_permutation_alloc(n);
    int signum;

    if (gsl_linalg_LU_decomp(K_copy, p, &signum) != 0) {
        fprintf(stderr, "Error: Kernel matrix is singular. Skipping update.\n");
        gsl_matrix_free(K); gsl_matrix_free(K_copy); gsl_permutation_free(p);
        return 0.0;
    }

    gsl_matrix *K_inv = gsl_matrix_alloc(n, n);
    gsl_linalg_LU_invert(K_copy, p, K_inv);

    gsl_vector *k_star = gsl_vector_alloc(n);
    for (int i = 0; i < n; i++) {
        gsl_vector_set(k_star, i, rbf_kernel(X[i], x_new, length_scale));
    }

    gsl_vector *alpha = gsl_vector_alloc(n);
    gsl_vector_view Y_vec = gsl_vector_view_array(Y, n);
    gsl_blas_dgemv(CblasNoTrans, 1.0, K_inv, &Y_vec.vector, 0.0, alpha);

    double mu = 0.0;
    gsl_blas_ddot(k_star, alpha, &mu);

    gsl_vector *v = gsl_vector_alloc(n);
    gsl_linalg_LU_solve(K_copy, p, k_star, v);

    double var_reduction = 0.0;
    gsl_blas_ddot(k_star, v, &var_reduction);
    *sigma = rbf_kernel(x_new, x_new, length_scale) - var_reduction;
    
    gsl_matrix_free(K); gsl_matrix_free(K_inv); gsl_matrix_free(K_copy);
    gsl_vector_free(k_star); gsl_vector_free(alpha); gsl_vector_free(v);
    gsl_permutation_free(p);

    return mu;
}

static double expected_improvement(double mu, double sigma, double f_max) {
    if (sigma < 1e-9) return 0.0;
    double Z = (mu - f_max) / sigma;
    return (mu - f_max) * gsl_cdf_ugaussian_P(Z) + sigma * gsl_ran_ugaussian_pdf(Z);
}

static int select_best_connection(double *X, double *Y, int n, double length_scale) {
    double best_ei = -1.0;
    int best_conn = CONNECTION_MIN;
    double f_max = -1e9;
    for (int i = 0; i < n; i++) {
        if (Y[i] > f_max) f_max = Y[i];
    }

    for (int c = CONNECTION_MIN; c <= CONNECTION_MAX; c++) {
        double sigma = 0.0, mu = 0.0;
        mu = predict(c, X, Y, n, length_scale, &sigma);
        if (sigma < 0) sigma = 0;

        double ei = expected_improvement(mu, sqrt(sigma), f_max);
        if (ei > best_ei) {
            best_ei = ei;
            best_conn = c;
        }
    }
    return best_conn;
}

static void threads_manage_BO() {
    DataSet data;
    memset(&data, 0, sizeof(DataSet));
    double cur_thr = 0;
    size_t prev_total = 0;
    double length_scale = 5.0;
    int time_count = 1;

    mscptest_stats stats;
    mscptest_get_stats(&stats);
    prev_total = stats.done;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        sleep(opt_dur_gl);
        mscptest_get_stats(&stats);

        pthread_mutex_lock(&bo_lock);
        cur_thr = (double)(stats.done - prev_total) / opt_dur_gl;
        prev_total = stats.done;

        if (data.size < MAX_DATA) {
            data.X[data.size] = cur_thread_gl;
            data.Y[data.size] = cur_thr;
            data.size++;
        }
        pthread_mutex_unlock(&bo_lock);

        int before_thread = cur_thread_gl;
        if (data.size < 3) {
            cur_thread_gl = rand() % (CONNECTION_MAX - CONNECTION_MIN + 1) + CONNECTION_MIN;
        } else {
            cur_thread_gl = select_best_connection(data.X, data.Y, data.size, length_scale);
        }
        
        int diff = cur_thread_gl - before_thread;
        if (diff > 0) {
            for(int i=0; i<diff; ++i) sem_post(&sem_for_optimization);
        }
        else if (diff < 0) {
            for(int i=0; i<-diff; ++i) sem_trywait(&sem_for_optimization);
        }

        fprintf(stderr, "LOG: BO iter=%d, threads=%d, throughput=%.4f Gbps\n", 
                iter + 1, cur_thread_gl, cur_thr * 8 / 1e9);
        time_count++;
    }
    // After optimization, continue monitoring
    while(!transfer_complete_gl) {
        sleep(opt_dur_gl);
        mscptest_get_stats(&stats);
        cur_thr = (double)(stats.done - prev_total) / opt_dur_gl;
        prev_total = stats.done;
        fprintf(stderr, "LOG: time=%ds, threads=%d, throughput=%.4f Gbps\n", 
                time_count * opt_dur_gl, cur_thread_gl, cur_thr * 8 / 1e9);
        time_count++;
    }
}

// --- SFTPコマンド遅延ベースの最適化 ---
static void GD_algorithm_3(double prev_thr, double cur_thr, int prev_conn, int cur_conn) {
    if (cur_conn == prev_conn) {
        return;
    }
    double gradient = (cur_thr - prev_thr) / (cur_conn - prev_conn);
    if (!learning_factor_reset && fabs(prev_thr) > EPSILON) {
        delta = gradient / prev_thr;
    } else {
        learning_factor = 1;
        learning_factor_reset = 0;
        delta = 1;
    }

    if (!GD_algorithm_direction_flag) {
        if (delta > 0.02) {
            temp = (int)ceil(delta * learning_factor);
            if (cur_thread_gl + temp < MAX_THREADS) {
                cur_thread_gl += temp;
                test_change_flag_gl = (cur_thread_gl >= active_thread_count_gl) ? 1 : 3;
            }
            learning_factor *= 2;
        } else {
            GD_algorithm_direction_flag = 1;
            learning_factor = 1;
            temp = (int)ceil(-delta * learning_factor);
            if (cur_thread_gl > temp) {
                cur_thread_gl -= temp;
                test_change_flag_gl = 2;
            }
            learning_factor *= 2;
        }
    } else {
        if (delta < -0.02) {
            temp = (int)ceil(-delta * learning_factor);
            if (cur_thread_gl > temp) {
                cur_thread_gl -= temp;
                test_change_flag_gl = 2;
            }
            learning_factor *= 2;
        } else {
            GD_algorithm_direction_flag = 0;
            learning_factor = 1;
            temp = (int)ceil(delta * learning_factor);
            if (cur_thread_gl + temp < MAX_THREADS) {
                cur_thread_gl += temp;
                test_change_flag_gl = (cur_thread_gl >= active_thread_count_gl) ? 1 : 3;
            }
            learning_factor *= 2;
        }
    }
}

static void *thread_manage_sftp_cmd() {
    struct timeval start_time, current_time, prev_time;
    gettimeofday(&start_time, NULL);
    prev_time = start_time;
    
    int prev_latency_count[MAX_THREADS] = {0};
    double prev_thr = 0;
    size_t prev_total_bytes = 0;
    int prev_conn = 1, cur_conn = 1;
    int opt_dur_dyna_us = 100000;

    while (!transfer_complete_gl) {
        if (test_change_flag_gl == 1) { // スレッドが増加した場合
            while (flag_nr_conn_changed_gl) {
                usleep(100);
            }
        } else {
            usleep(opt_dur_dyna_us);
        }
        test_change_flag_gl = 0;
        if (transfer_complete_gl) break;

        gettimeofday(&current_time, NULL);
        
        size_t total_copied_bytes = 0;
        size_t total_latency_sum = 0;
        int sample_count = 0;
        int max_latency = 0;

        for (int i = 0; i < active_thread_count_gl; ++i) {
            total_copied_bytes += thread_data[i].copied_bytes;
            int count = thread_data[i].latency_count;
            int new_data_count = count - prev_latency_count[i];
            
            if (new_data_count > 0) {
                for (int j = 0; j < new_data_count; ++j) {
                    int pos = (prev_latency_count[i] + j) % RING_BUF;
                    int latency = thread_data[i].latency_buffer[pos];
                    total_latency_sum += latency;
                    if (latency > max_latency) max_latency = latency;
                }
                sample_count += new_data_count;
            }
            prev_latency_count[i] = count;
        }

        if (sample_count < sftp_cmd_get_count_gl) {
            prev_time = current_time;
            prev_total_bytes = total_copied_bytes;
            continue;
        }

        size_t bytes_sent = total_copied_bytes - prev_total_bytes;
        double time_delta = (current_time.tv_sec - prev_time.tv_sec) + (current_time.tv_usec - prev_time.tv_usec) / 1000000.0;
        double throughput_Gbps = (time_delta > 0) ? (double)bytes_sent * 8 / (time_delta * 1e9) : 0;
        
        opt_dur_dyna_us = max(10000, max_latency * 2);
        
        int old_thread_gl = cur_thread_gl;
        cur_conn = old_thread_gl;

        if (cur_thread_gl < 2) {
            cur_thread_gl++;
            test_change_flag_gl = (cur_thread_gl >= active_thread_count_gl) ? 1 : 3;
        } else {
            GD_algorithm_3(prev_thr, throughput_Gbps, prev_conn, cur_conn);
        }
        
        int diff = cur_thread_gl - old_thread_gl;
        if (diff > 0) {
            for(int i=0; i<diff; ++i) sem_post(&sem_for_optimization);
        }
        else if (diff < 0) {
            for(int i=0; i<-diff; ++i) sem_trywait(&sem_for_optimization);
        }

        fprintf(stderr, "LOG: time=%ds, threads=%d, throughput=%.4f Gbps, latency=%d ms\n", 
                (int)(current_time.tv_sec - start_time.tv_sec), cur_thread_gl, throughput_Gbps, max_latency);


        prev_thr = throughput_Gbps;
        prev_conn = cur_conn;
        prev_total_bytes = total_copied_bytes;
        prev_time = current_time;
    }
    return NULL;
}


/**
 * @brief 最適化アルゴリズムを実行し、スレッド数を動的に調整する
 */
void *dynamic_transfer_threads_management(void *arg) {
    if (opt_algo_gl == 5) {
        threads_manage_BO();
        return NULL;
    }
    if (sftp_cmd_metric_flag_gl) {
        thread_manage_sftp_cmd();
        return NULL;
    }

    double prev_thr = 0, cur_thr = 0;
    int prev_conn = cur_thread_gl;
    size_t prev_bytes = 0;
    int time_count = 1;

    mscptest_stats stats;
    mscptest_get_stats(&stats);
    prev_bytes = stats.done;

    while (!transfer_complete_gl) {
        sleep(opt_dur_gl);
        if (transfer_complete_gl) break;

        mscptest_get_stats(&stats);
        size_t current_bytes = stats.done;
        
        cur_thr = (double)(current_bytes - prev_bytes) / opt_dur_gl;
        prev_bytes = current_bytes;

        int old_thread_gl = cur_thread_gl;
        int cur_conn = old_thread_gl;

        switch (opt_algo_gl) {
            case 1: simple_increase(); break;
            case 2: MN_algorithm(prev_thr, cur_thr); break;
            case 3: HC_algorithm(prev_thr, cur_thr); break;
            case 4: GD_algorithm(prev_thr, cur_thr, prev_conn, cur_conn); break;
            case 6: adaptive_adjustment(prev_thr, cur_thr); break;
            default: break;
        }
        
        int new_thread_gl = cur_thread_gl;

        if (new_thread_gl > old_thread_gl) {
            for (int i = 0; i < (new_thread_gl - old_thread_gl); i++) {
                sem_post(&sem_for_optimization);
            }
        } else if (new_thread_gl < old_thread_gl) {
            for (int i = 0; i < (old_thread_gl - new_thread_gl); i++) {
                sem_trywait(&sem_for_optimization);
            }
        }
        
        fprintf(stderr, "LOG: time=%ds, threads=%d, throughput=%.4f Gbps\n", 
                time_count * opt_dur_gl, cur_thread_gl, cur_thr * 8 / 1e9);
        
        time_count++;
        prev_thr = cur_thr;
        prev_conn = cur_conn;
    }
    return NULL;
}
