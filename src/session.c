#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <libssh/callbacks.h> // <--- エラー修正のため追加

/**
 * @brief パスフレーズをキャッシュから提供するためのコールバック関数
 */
static int ssh_cache_passphrase(const char *prompt, char *buf, size_t len, int echo, int verify, void *userdata) {
    const char *passphrase = (const char *)userdata;
    if (!passphrase || strlen(passphrase) == 0) {
        return 1;
    }
    strncpy(buf, passphrase, len - 1);
    buf[len - 1] = '\0';
    return 0;
}

static struct ssh_callbacks_struct cb = {
    .auth_function = ssh_cache_passphrase,
    .userdata = NULL,
};

ssh_session init_ssh_session(const char *host, const char *user, const char *passphrase) {
    ssh_session session = ssh_new();
    if (session == NULL) {
        fprintf(stderr, "Error initializing SSH session\n");
        exit(EXIT_FAILURE);
    }

    cb.userdata = (void *)passphrase;
    ssh_callbacks_init(&cb);
    ssh_set_callbacks(session, &cb);

    ssh_options_set(session, SSH_OPTIONS_HOST, host);
    ssh_options_set(session, SSH_OPTIONS_USER, user);
    return session;
}

void connect_ssh_session(ssh_session session) {
    int rc = ssh_connect(session);
    if (rc != SSH_OK) {
        fprintf(stderr, "Error connecting to SSH server: %s\n", ssh_get_error(session));
        ssh_free(session);
        exit(EXIT_FAILURE);
    }
}

void authenticate_ssh_session(ssh_session session, const char *key_path, const char *passphrase) {
    int rc;
    if (key_path) {
        ssh_key key;
        rc = ssh_pki_import_privkey_file(key_path, passphrase, NULL, NULL, &key);
        if (rc != SSH_OK) {
            fprintf(stderr, "Error importing private key: %s\n", ssh_get_error(session));
            ssh_disconnect(session);
            ssh_free(session);
            pthread_exit(NULL);
        }
        rc = ssh_userauth_publickey(session, NULL, key);
        ssh_key_free(key);
    } else {
        rc = ssh_userauth_publickey_auto(session, NULL, NULL);
    }

    if (rc != SSH_AUTH_SUCCESS) {
        fprintf(stderr, "Authentication failed: %s\n", ssh_get_error(session));
        ssh_disconnect(session);
        ssh_free(session);
        pthread_exit(NULL);
    }
}

sftp_session init_sftp_session(ssh_session session) {
    sftp_session sftp = sftp_new(session);
    if (sftp == NULL) {
        fprintf(stderr, "Error initializing SFTP session: %s\n", ssh_get_error(session));
        ssh_disconnect(session);
        ssh_free(session);
        exit(EXIT_FAILURE);
    }

    int rc = sftp_init(sftp);
    if (rc != SSH_OK) {
        fprintf(stderr, "Error initializing SFTP channel: %d\n", sftp_get_error(sftp));
        sftp_free(sftp);
        ssh_disconnect(session);
        ssh_free(session);
        exit(EXIT_FAILURE);
    }
    return sftp;
}
