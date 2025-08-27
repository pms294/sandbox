#include "monitor.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

/**
 * @brief 全体の転送統計情報を取得する
 * この関数はスレッドセーフではありませんが、モニター表示用のため許容します。
 * 最終的な正確な値はmainスレッドで取得します。
 */
void mscptest_get_stats(mscptest_stats *stats) {
    stats->done = 0;
    if (thread_data) {
        // active_thread_count_glは変動する可能性があるため、ループの前に値をコピーする
        int current_active_threads = active_thread_count_gl;
        for (int i = 0; i < current_active_threads; ++i) {
            stats->done += thread_data[i].copied_bytes;
        }
    }
    stats->total = (fake_file_size_gl != 0) ? fake_file_size_gl : real_file_size_gl;
}

/**
 * @brief 転送中の進捗を監視し、表示するスレッド
 * 最終レポートの表示はmainスレッドに責務を移譲しました。
 */
void *monitor_transfer(void *arg) {
    struct timeval last_time;
    gettimeofday(&last_time, NULL);
    
    mscptest_stats stats;
    size_t last_done = 0;

    // 転送が完了するまでループ
    while (!transfer_complete_gl) {
        usleep(200000); // 200ミリ秒ごとにチェック

        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        
        double elapsed = (current_time.tv_sec - last_time.tv_sec) + 
                         (current_time.tv_usec - last_time.tv_usec) / 1000000.0;
        
        // 画面の更新は1秒ごとに行う
        if (elapsed >= 1.0) {
            mscptest_get_stats(&stats);
            size_t now_done = stats.done;
            size_t total = stats.total > 0 ? stats.total : 1;
            size_t diff = now_done - last_done;
            
            double bps = (elapsed > 0) ? (diff / elapsed) : 0;
            double throughput_MB = bps / 1024.0 / 1024.0;
            int percent = (int)(100.0 * now_done / total);

            fprintf(stderr, "\r\033[K %3d%% | %7.2f MB/s | %zu / %zu bytes", 
                    percent, throughput_MB, now_done, total);
            fflush(stderr);

            last_time = current_time;
            last_done = now_done;
        }
    }
    // 最後の進捗表示をクリア
    fprintf(stderr, "\r\033[K");
    fflush(stderr);
    return NULL;
}
