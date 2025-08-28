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
static void adaptive_GD_algorithm(double prev_thr, double cur_thr, int prev_conn, int cur_conn) {
    if (cur_conn == prev_conn || fabs(cur_thr - prev_thr) < 0.50) {
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
            }
            learning_factor *= 2;
        } else {
            GD_algorithm_direction_flag = 1;
            learning_factor = 1;
            temp = (int)ceil(-delta * learning_factor);
            if (cur_thread_gl > temp) {
                cur_thread_gl -= temp;
            }
            learning_factor *= 2;
        }
    } else {
        if (delta < -0.02) {
            temp = (int)ceil(-delta * learning_factor);
            if (cur_thread_gl > temp) {
                cur_thread_gl -= temp;
            }
            learning_factor *= 2;
        } else {
            GD_algorithm_direction_flag = 0;
            learning_factor = 1;
            temp = (int)ceil(delta * learning_factor);
            if (cur_thread_gl + temp < MAX_THREADS) {
                cur_thread_gl += temp;
            }
            learning_factor *= 2;
        }
    }
}

static void *thread_manage_sftp_cmd() {

    // ループを開始する前に、変数を初期化する
    struct timeval start_time, current_time, prev_time;
    gettimeofday(&start_time, NULL);
    prev_time = start_time;
    
    int prev_latency_count[MAX_THREADS] = {0};
    double prev_thr_Gbps = 0, cur_thr_Gbps = 0;
    size_t prev_total_sent_bytes = 0;
    int prev_conn = 1, cur_conn = 1;
    int adaptive_sleep_usec = 100000; // 初期値100ms
    int elapsed_time_usec = 0;
    int stedy_state_cnt = 0;
    int stedy_state_check = 0;


    while (!transfer_complete_gl) {

        // g_thread_adjust_state == 0   スレッドの初期条件
        // g_thread_adjust_state == 1   スレッドが増加したとき
        // g_thread_adjust_state == 2   スレッドが減ったとき
        // g_thread_adjust_state == 3   スレッドが変わらなかったとき, コネクションが確立したとき
        // g_thread_adjust_state == 4   スレッドの初期条件の後、1回だけスレッドを増やしたとき

        if (g_thread_adjust_state == 0) {
            usleep(1000000); // 1秒待つ   コネクションの確立の過渡期、コネクションの増減を行わない
        }
        if (g_thread_adjust_state == 1) {
            usleep(100000); // 100m秒待つ     コネクションの確立の過渡期、コネクションの増減を行わない
        }
        if (g_thread_adjust_state == 2) {
            usleep(100000); // 100m秒待つ     コネクションの確立の過渡期、コネクションの増減を行わない
        }
        if (g_thread_adjust_state == 3) {
            usleep(adaptive_sleep_usec); // コネクション確立の過渡期を過ぎた後、コネクションを変化させる可能性あり
        }
        if (g_thread_adjust_state == 4) { 
            usleep(adaptive_sleep_usec); // コネクション確立の過渡期を過ぎた後、コネクションを変化させる可能性あり
        }

        
        usleep(adaptive_sleep_usec);
        if (transfer_complete_gl) break;

        gettimeofday(&current_time, NULL);
        
        // 収集するデータの初期化
        size_t total_sent_bytes = 0;
        size_t total_latency_usec_sum = 0;
        int total_sample_count = 0;

        int max_latency_usec = 0;
        int ave_latency_usec = 0;

        // リングバッファから遅延データを収集
        for (int thread_i = 0; thread_i < active_thread_count_gl; ++thread_i) {
            total_sent_bytes += thread_data[thread_i].copied_bytes;
            int now_latency_count_i = thread_data[thread_i].latency_count;
            int new_data_count_i = now_latency_count_i - prev_latency_count[thread_i];
            if (new_data_count_i > 0) {
                for (int j = 0; j < new_data_count_i; ++j) {
                    int pos = (prev_latency_count[thread_i] + j) % RING_BUF;
                    int latency_usec = thread_data[thread_i].a_latency_usec_buffer[pos];
                    total_latency_usec_sum += latency_usec;
                    if (latency_usec > max_latency_usec) max_latency_usec = latency_usec;
                }
                total_sample_count += new_data_count_i;
            }
            prev_latency_count[thread_i] = now_latency_count_i;
        }

        // サンプル数が十分でない場合はスキップ
        if (total_sample_count < g_sftp_cmd_get_count) {
            prev_time = current_time;
            prev_total_sent_bytes = total_sent_bytes;
            continue;
        }

        // 転送バイト数、転送時間、スループット、経過時間、平均遅延時間の計算
        size_t new_sent_bytes = total_sent_bytes - prev_total_sent_bytes;
        int time_delta_usec = (current_time.tv_sec - prev_time.tv_sec) * 1000000 + (current_time.tv_usec - prev_time.tv_usec);
        elapsed_time_usec = (current_time.tv_sec - start_time.tv_sec) * 1000000 + (current_time.tv_usec - start_time.tv_usec);
        ave_latency_usec = total_latency_usec_sum / total_sample_count;
        double thr_Gbps = (double) new_sent_bytes * 8 / 1000 / time_delta_usec;

        // 過渡状態の
        if (g_thread_adjust_state == 3 || g_thread_adjust_state == 4) {
            cur_thr_Gbps = thr_Gbps;
        }
        
        // データ取得時間を動的に調整
        adaptive_sleep_usec = min(100000, max_latency_usec * 2);

        
        // fprintf(stderr, "STATE=%d, stedy_cnt=%d, time=%d us, threads=%d, active=%d, throughput=%.4f Gbps, cur_thr=%.4f Gbps, max_latency=%d us, ave_latency=%d us, sent_bytes=%ld byte, time_delta=%d us, sample_count=%d\n", 
        //         g_thread_adjust_state, stedy_state_cnt, elapsed_time_usec, cur_thread_gl, active_thread_count_gl, thr_Gbps, cur_thr_Gbps,max_latency_usec, ave_latency_usec, new_sent_bytes, time_delta_usec, total_sample_count);

        printf("%d, %d, %.4f\n", 
                elapsed_time_usec, cur_thread_gl, thr_Gbps);


        // // 初期過渡状態、増加過渡状態、減少過渡状態　＝＞　スレッドの変化、増減判断なし
        if (g_thread_adjust_state == 0) {
            g_thread_adjust_state = 4;
            prev_total_sent_bytes = total_sent_bytes;
            prev_time = current_time;
            continue;
        }
        else if (g_thread_adjust_state == 1) {
            g_thread_adjust_state = 3;
            prev_total_sent_bytes = total_sent_bytes;
            prev_time = current_time;  
            continue;
        }
        else if (g_thread_adjust_state == 2) {
            g_thread_adjust_state = 3;
            prev_total_sent_bytes = total_sent_bytes;
            prev_time = current_time;
            continue;
        }

        // 増減をここで判断
        int old_thread_gl = cur_thread_gl;
        cur_conn = old_thread_gl;

        // 閾値内の状態 or コネクション確立時
        if (g_thread_adjust_state == 3) {
            adaptive_GD_algorithm(prev_thr_Gbps, cur_thr_Gbps, prev_conn, cur_conn);
        }
        else if (g_thread_adjust_state == 4) {
            cur_thread_gl++;
        }

        
        // スレッド数の変化に応じてセマフォを操作
        int diff = cur_thread_gl - old_thread_gl;
        if (diff == 0) {
            g_thread_adjust_state = 3;
            stedy_state_cnt++;
            if (stedy_state_check == 1) {
                cur_thread_gl--;
                stedy_state_check = 0;
                stedy_state_cnt = 0;
                // printf("deb 強制的に減らしました\n");
            }
            if (stedy_state_cnt >= 20) {
                cur_thread_gl++;
                stedy_state_check = 1;
                stedy_state_cnt = 0;
                // printf("deb 強制的に増やしました\n");
            }
        }
        else if (diff > 0) {
            for(int i=0; i<diff; ++i) sem_post(&sem_for_optimization);
            g_thread_adjust_state = 1;
            stedy_state_check = 0; //　強制的に減らすフラグをリセット
        }
        else if (diff < 0) {
            for(int i=0; i<-diff; ++i) sem_trywait(&sem_for_optimization);
            g_thread_adjust_state = 2;
            stedy_state_check = 0; //　強制的に減らすフラグをリセット
        }

        prev_thr_Gbps = cur_thr_Gbps;
        prev_conn = cur_conn;
        prev_total_sent_bytes = total_sent_bytes;
        prev_time = current_time;
        // printf("dev\n");
    }
    return NULL;
}


/**
 * @brief 最適化アルゴリズムを実行し、スレッド数を動的に調整する
 */
void *dynamic_transfer_threads_management(void *arg) {
    /* -c オプションだとこっちに行く（ただし、今のコードだと -a オプションも*/
    if (measure_transaction_latency_gl) {
        thread_manage_sftp_cmd();
        return NULL;
    }
    if (opt_algo_gl == 5) {
        threads_manage_BO();
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
