import tkinter as tk
from tkinter import messagebox
import random
import time
from datetime import datetime

BOARD_SIZE = 8  # ボードサイズ
SIMULATION_TIME = 0.1  # モンテカルロ法の計算時間（秒）

class Reversi:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("リバーシ")
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = 1  # 1: 黒, -1: 白

        # 棋譜を記録するリストを初期化
        self.move_history = []

        # 初期配置
        self.board[3][3] = -1
        self.board[4][4] = -1
        self.board[3][4] = 1
        self.board[4][3] = 1

        # キャンバスを作成
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg="green")
        self.canvas.pack()
        self.draw_board()

        # クリックイベント
        self.canvas.bind("<Button-1>", self.handle_click)

        # イベントループ開始
        self.root.mainloop()

    def draw_board(self):
        """ボードを描画する"""
        self.canvas.delete("all")
        cell_size = 400 // BOARD_SIZE

        # グリッド線の描画
        for i in range(BOARD_SIZE + 1):
            self.canvas.create_line(i * cell_size, 0, i * cell_size, 400, fill="black")
            self.canvas.create_line(0, i * cell_size, 400, i * cell_size, fill="black")

        # 石の描画
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] != 0:
                    x1 = col * cell_size + 5
                    y1 = row * cell_size + 5
                    x2 = (col + 1) * cell_size - 5
                    y2 = (row + 1) * cell_size - 5
                    color = "black" if self.board[row][col] == 1 else "white"
                    self.canvas.create_oval(x1, y1, x2, y2, fill=color)

    def handle_click(self, event):
        """クリックイベントを処理する"""
        if self.current_player != 1:  # 人間は黒（1）だけを操作する
            return

        cell_size = 400 // BOARD_SIZE
        col = event.x // cell_size
        row = event.y // cell_size

        if self.is_valid_move(row, col, self.current_player):
            self.make_move(row, col, self.current_player)
            self.current_player *= -1  # プレイヤー交代
            self.draw_board()

            # 白（後手）のターンを開始
            self.root.after(500, self.white_turn)

    def white_turn(self):
        """白（後手）のモンテカルロ法による選択"""
        if self.current_player == -1:
            start_time = time.time()
            best_move = self.monte_carlo_search(SIMULATION_TIME)
            end_time = time.time()

            print(f"白の計算時間: {end_time - start_time:.2f}秒")

            if best_move:
                row, col = best_move
                self.make_move(row, col, self.current_player)
                self.current_player *= -1  # プレイヤー交代
                self.draw_board()

            # ゲーム終了条件を確認
            if not self.has_valid_moves(self.current_player):
                self.current_player *= -1
                if not self.has_valid_moves(self.current_player):
                    self.end_game()

    def monte_carlo_search(self, time_limit):
        """モンテカルロ法で最適な手を探す"""
        valid_moves = self.get_valid_moves(self.current_player)
        if not valid_moves:
            return None

        move_scores = {move: 0 for move in valid_moves}
        simulations = {move: 0 for move in valid_moves}

        start_time = time.time()

        while time.time() - start_time < time_limit:
            move = random.choice(valid_moves)
            board_copy = [row[:] for row in self.board]
            self.make_move(move[0], move[1], self.current_player, board_copy)

            winner = self.simulate_random_game(board_copy, -self.current_player)
            if winner == self.current_player:
                move_scores[move] += 1
            simulations[move] += 1

        # 勝率の計算
        best_move = max(valid_moves, key=lambda m: move_scores[m] / (simulations[m] + 1e-6))
        return best_move

    def simulate_random_game(self, board, player):
        """ランダムにプレイアウトして勝者を返す"""
        current_player = player
        while self.has_valid_moves_in_board(board, 1) or self.has_valid_moves_in_board(board, -1):
            valid_moves = self.get_valid_moves_in_board(board, current_player)
            if valid_moves:
                move = random.choice(valid_moves)
                self.make_move(move[0], move[1], current_player, board)
            current_player *= -1

        black_score = sum(row.count(1) for row in board)
        white_score = sum(row.count(-1) for row in board)
        return 1 if black_score > white_score else -1 if white_score > black_score else 0

    def make_move(self, row, col, player, board=None):
        """石を配置し、挟んだ相手の石をひっくり返す"""
        board = board or self.board
        board[row][col] = player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            stones_to_flip = []
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if board[r][c] == 0:
                    break
                if board[r][c] == -player:
                    stones_to_flip.append((r, c))
                elif board[r][c] == player:
                    for rr, cc in stones_to_flip:
                        board[rr][cc] = player
                    break
                r += dr
                c += dc

    def is_valid_move(self, row, col, player):
        """有効な手かどうかを判定する"""
        return self.is_valid_move_in_board(self.board, row, col, player)

    def is_valid_move_in_board(self, board, row, col, player):
        """特定のボードで有効な手かどうかを判定"""
        if board[row][col] != 0:
            return False

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            has_opponent_between = False
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if board[r][c] == 0:
                    break
                if board[r][c] == -player:
                    has_opponent_between = True
                elif board[r][c] == player:
                    if has_opponent_between:
                        return True
                    break
                r += dr
                c += dc
        return False

    def get_valid_moves(self, player):
        """プレイヤーが置ける有効な手をすべて返す"""
        return self.get_valid_moves_in_board(self.board, player)

    def get_valid_moves_in_board(self, board, player):
        """特定のボードでプレイヤーが置ける有効な手を返す"""
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.is_valid_move_in_board(board, row, col, player):
                    valid_moves.append((row, col))
        return valid_moves

    def has_valid_moves(self, player):
        """プレイヤーに有効な手が残っているか確認"""
        return self.has_valid_moves_in_board(self.board, player)

    def has_valid_moves_in_board(self, board, player):
        """特定のボードでプレイヤーに有効な手が残っているか確認"""
        return any(self.is_valid_move_in_board(board, row, col, player) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE))

    def make_move(self, row, col, player, board=None):
        """石を配置し、挟んだ相手の石をひっくり返す"""
        board = board or self.board
        board[row][col] = player

        # 棋譜に記録
        if board is self.board:  # メインボードの場合のみ記録
            player_str = "黒" if player == 1 else "白"
            self.move_history.append(f"{player_str}: ({row + 1}, {col + 1})")

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            stones_to_flip = []
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if board[r][c] == 0:
                    break
                if board[r][c] == -player:
                    stones_to_flip.append((r, c))
                elif board[r][c] == player:
                    for rr, cc in stones_to_flip:
                        board[rr][cc] = player
                    break
                r += dr
                c += dc

    def end_game(self):
        """ゲーム終了時の処理"""
        black_score = sum(row.count(1) for row in self.board)
        white_score = sum(row.count(-1) for row in self.board)
        winner = "黒" if black_score > white_score else "白" if white_score > black_score else "引き分け"

        # 現在時刻を取得してファイル名に追加
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reversi_record_{timestamp}.txt"

        # 棋譜をファイルに保存
        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n".join(self.move_history))
            file.write(f"\n\n最終スコア:\n黒: {black_score}, 白: {white_score}\n勝者: {winner}")

        # メッセージボックスで結果を表示
        messagebox.showinfo(
            "ゲーム終了",
            f"黒: {black_score}, 白: {white_score}\n勝者: {winner}\n\n棋譜は '{filename}' に保存されました。"
        )
        self.root.quit()

# メイン処理
if __name__ == "__main__":
    Reversi()