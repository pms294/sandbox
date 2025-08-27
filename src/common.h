#ifndef COMMON_H
#define COMMON_H

#include <libssh/libssh.h>
#include <libssh/sftp.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>

// --- 共通の定数 ---
#define DEFAULT_BUF_SZ   16384
#define DEFAULT_NR_AHEAD 32
#define MAX_THREADS      100
#define RING_BUF         10000

// --- 共通の構造体定義 ---

// 各転送スレッドのデータを保持する構造体
typedef struct {
  const char *local_file;
  const char *remote_file;
  const char *host;
  const char *user;
  const char *passphrase;
  const char *key_path;
  off_t chunk_size;

  size_t copied_bytes;
  pthread_t tid;

  uint32_t latency_buffer[RING_BUF];
  int len_buffer[RING_BUF];
  int latency_count;

  struct timeval conn_established_time;
  struct timeval first_transfer_time;
  int started_flag;
} thread_data_t;

// 全体の転送統計情報を保持する構造体
typedef struct {
  size_t total;
  size_t done;
} mscptest_stats;

// --- グローバル変数のextern宣言 ---
// 実体はmain.cで定義

// 状態管理
extern int transfer_complete_gl;
extern volatile int active_thread_count_gl;
extern int chunk_count_gl;
extern int chunk_last_count_gl;
extern off_t real_file_size_gl;
extern off_t fake_file_size_gl;

// 最適化アルゴリズム関連
extern int cur_thread_gl;
extern int sem_post_gl;
extern int sem_wait_gl;
extern int opt_algo_gl;
extern int opt_dur_gl;
extern int stop_thread_inc_gl;

// SFTPコマンド遅延測定用
extern bool measure_transaction_latency_gl;
extern int sftp_cmd_get_count_gl;
extern int sftp_cmd_worst_case_latency;

// 同期用
extern pthread_mutex_t transfer_mutex;
extern sem_t sem_for_maxstartups;
extern sem_t sem_for_optimization;

// ワーカースレッドの動的変更用フラグ
extern int flag_nr_conn_changed_gl;
extern int test_change_flag_gl;
extern int exp_opt_h_gl;

// 共有データ
extern thread_data_t *thread_data;

#endif  // COMMON_H
