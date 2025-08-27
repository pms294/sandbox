#ifndef MONITOR_H
#define MONITOR_H

#include "common.h"

/**
 * @brief 全体の転送統計情報を取得する
 */
void mscptest_get_stats(mscptest_stats *stats);

/**
 * @brief 転送状況を監視し、進捗を表示するスレッドのエントリーポイント
 */
void *monitor_transfer(void *arg);

#endif  // MONITOR_H
