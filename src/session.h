#ifndef SESSION_H
#define SESSION_H

#include "common.h"

/**
 * @brief SSHセッションを初期化する
 */
ssh_session init_ssh_session(const char *host, const char *user, const char *passphrase);

/**
 * @brief SSHサーバーに接続する
 */
void connect_ssh_session(ssh_session session);

/**
 * @brief 公開鍵認証を行う
 */
void authenticate_ssh_session(ssh_session session, const char *key_path, const char *passphrase);

/**
 * @brief SFTPセッションを初期化する
 */
sftp_session init_sftp_session(ssh_session session);

#endif  // SESSION_H
