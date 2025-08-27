# コンパイラ
CC = gcc

# コンパイルフラグ
# -Isrc: srcディレクトリをインクルードパスに追加
CFLAGS = -Wall -g -fsanitize=address -D_GNU_SOURCE -Isrc

# リンカフラグ
# -L/usr/local/lib: /usr/local/lib をライブラリ検索パスに追加 (エラー修正)
LDFLAGS = -L/usr/local/lib -lssh -lpthread -lm -lgsl -lgslcblas -fsanitize=address

# --- ファイルとディレクトリ ---

# ディレクトリの定義
SRC_DIR = src
BUILD_DIR = build

# 実行ファイル名
EXEC = $(BUILD_DIR)/mscptest

# ソースファイルのベース名リスト
SRCS_BASE = main.c session.c transfer.c optimizer.c monitor.c

# ソースファイルのフルパスリスト (src/main.c, src/session.c, ...)
SRCS = $(addprefix $(SRC_DIR)/, $(SRCS_BASE))

# オブジェクトファイルのリスト (build/main.o, build/session.o, ...)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

# 依存関係ファイルのリスト (build/main.d, build/session.d, ...)
DEPS = $(patsubst %.c, $(BUILD_DIR)/%.d, $(SRCS_BASE))


# --- ターゲット ---

# デフォルトのターゲット
all: $(EXEC)

# 実行可能ファイルのリンク
$(EXEC): $(OBJS)
	@echo "Linking..."
	@$(CC) -o $@ $^ $(LDFLAGS)
	@echo "Build finished: $@"

# オブジェクトファイルのコンパイル
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c -o $@ $< -MMD -MP

# クリーンアップ
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR)

# .PHONY: ターゲット名がファイル名と衝突するのを防ぐ
.PHONY: all clean

# 生成された依存関係ファイルをインクルードする
-include $(DEPS)
