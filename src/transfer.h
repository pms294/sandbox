#ifndef TRANSFER_H
#define TRANSFER_H

#include "common.h"

// transfer_manager_threadに渡す引数をまとめた構造体
typedef struct {
    const char *local_file;
    const char *remote_file;
    const char *host;
    const char *user;
    const char *passphrase;
    off_t chunk_size;
} transfer_args_t;

/**
 * @brief ワーカースレッドの生成・監視・終了を管理するスレッドのエントリーポイント
 */
void *transfer_manager_thread(void *arg);

#endif // TRANSFER_H
