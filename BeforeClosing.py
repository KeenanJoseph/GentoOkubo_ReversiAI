import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import time
import glob
import os

# --------------------------------------------------------------------------------------------------
# 盤面について
# --------------------------------------------------------------------------------------------------

def initial_board():
    board = [[0 for _ in range(8)] for _ in range(8)]
    board[3][3] = board[4][4] = -1
    board[3][4] = board[4][3] = 1
    return board

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1)
]

def is_on_board(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def apply_move(board, move, player):
    x, y = move
    board[x][y] = player

    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        stones_to_flip = []

        while is_on_board(nx, ny) and board[nx][ny] == -player:
            stones_to_flip.append((nx, ny))
            nx += dx
            ny += dy

        if is_on_board(nx, ny) and board[nx][ny] == player:
            for fx, fy in stones_to_flip:
                board[fx][fy] = player


# --------------------------------------------------------------------------------------------------
# 最後に、盤面表示
# 先手が●、後手が○、石がない場所は・
# 先手は、PredictMove、後手はランダム
# --------------------------------------------------------------------------------------------------

def print_board(board):
    symbols = {1: '●', -1: '○', 0: '・'}
    print("  1 2 3 4 5 6 7 8")
    for i, row in enumerate(board):
        print(f"{i+1} ", end='')
        print(' '.join(symbols[cell] for cell in row))
    print("\n")

# --------------------------------------------------------------------------------------------------
# 棋譜からデータセット作成
# --------------------------------------------------------------------------------------------------

def create_dataset_from_kifu(kifu_df):
    dataset = []
    board = initial_board()

    for idx, row in kifu_df.iterrows():
        player = 1 if row['プレイヤー'] == '黒' else -1
        move = (row['行'] - 1, row['列'] - 1)

        if not is_on_board(*move):
            print(f"警告: 範囲外の手を無視します 手番{idx+1}: {move}")
            continue

        board_copy = [cell for line in board for cell in line]
        move_index = move[0] * 8 + move[1]
        dataset.append((board_copy, move_index))

        apply_move(board, move, player)

    return dataset

# --------------------------------------------------------------------------------------------------
# データセットの定義
# 盤面を行列からリスト化
# 正解手（Move）をラベルに
# --------------------------------------------------------------------------------------------------

class ReversiDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, move = self.data[idx]
        return torch.tensor(board, dtype=torch.float32), torch.tensor(move, dtype=torch.long)

# --------------------------------------------------------------------------------------------------
# NNの定義
# Input(64次元) → Linear層(128次元) → ReLU → Linear層(64次元) → 出力
# 出力（64次元）	それぞれのマスに「打つべき確率（スコア）」を出す。
# --------------------------------------------------------------------------------------------------

class ReversiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------------------------------------------------------------------------------------
# 学習関数
# optimizer：Adam
# 　　　　　 過去の勾配を覚えておいて、その情報を使って、「どれくらい大きく/小さく」パラメータを動かすか調整
# loss_fn：CrossEntropyLoss
#          正解ラベルと予測確率のズレを測るもの
#          正しいマスのスコアが高くなるように、間違ったマスのスコアが低くなるように 調整してくれる損失関数
# --------------------------------------------------------------------------------------------------

def train(model, dataloader, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for boards, moves in dataloader:
            optimizer.zero_grad()
            outputs = model(boards)
            loss = loss_fn(outputs, moves)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


# --------------------------------------------------------------------------------------------------
# 有効手探索
# 打てるマスをチェック
# 隣が相手の石（-player）なら、そのままその方向にどんどん進む。
# 進んだ先に自分の石があれば、今調べてたマス (x, y) を有効手リストに追加
# そしてその方向のチェックは終了（break）。
# --------------------------------------------------------------------------------------------------

def get_valid_moves(board, player):
    moves = []
    for x in range(8):
        for y in range(8):
            if board[x][y] != 0:
                continue
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                while is_on_board(nx, ny) and board[nx][ny] == -player:
                    nx += dx
                    ny += dy
                    if is_on_board(nx, ny) and board[nx][ny] == player:
                        moves.append((x, y))
                        break
                if (x, y) in moves:
                    break
    return moves

# --------------------------------------------------------------------------------------------------
# シミュレーション
# 先手が●、後手が○、石がない場所は・
# 先手は、PredictMove、後手はランダム
# --------------------------------------------------------------------------------------------------

def simulate_game(model):
    board = initial_board()
    player = 1  # 黒スタート
    move_count = 0

    for turn in range(20):
        valid_moves = get_valid_moves(board, player)

        if not valid_moves:
            print(f"手番 {turn+1}: プレイヤー{player} パス")
            player *= -1
            continue

        if player == 1:
            board_flat = torch.tensor([cell for row in board for cell in row], dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                output = model(board_flat.unsqueeze(0))
                pred_move = output.argmax(dim=1).item()
                pred_row, pred_col = divmod(pred_move, 8)

            move = (pred_row, pred_col) if (pred_row, pred_col) in valid_moves else random.choice(valid_moves)
            move_type = "モデル手" if (pred_row, pred_col) in valid_moves else "ランダム手"
            print(f"黒の{move_type}: {move}")

        else:
            move = random.choice(valid_moves)
            print(f"白のランダム手: {move}")

        apply_move(board, move, player)

        print_board(board)
        time.sleep(1)

        player *= -1
        move_count += 1

    print(f"20手終了！総手数: {move_count}")


# --------------------------------------------------------------------------------------------------
# メイン関数
# --------------------------------------------------------------------------------------------------

def main():
    # 同じフォルダ内のxlsxファイルを自動取得
    xlsx_files = glob.glob(os.path.join(os.getcwd(), '*.xlsx'))

    if not xlsx_files:
        print("エラー: .xlsxファイルが見つかりませんでした。")
        return

    dataset = []
    for file_name in xlsx_files:
        try:
            kifu_df = pd.read_excel(file_name, sheet_name='棋譜')
            dataset.extend(create_dataset_from_kifu(kifu_df))
            print(f"{file_name} 読み込み成功！")
        except Exception as e:
            print(f"{file_name} 読み込み失敗: {e}")

    if not dataset:
        print("エラー: データが1件も作成できませんでした。")
        return

    train_dataset = ReversiDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = ReversiModel()

    train(model, train_loader, epochs=20)

    # モデル推論テスト
    board = initial_board()
    board_flat = torch.tensor([cell for row in board for cell in row], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(board_flat.unsqueeze(0))
        pred_move = output.argmax(dim=1).item()
        pred_row, pred_col = divmod(pred_move, 8)
        print(f'Predicted move: ({pred_row}, {pred_col})')

    simulate_game(model)

if __name__ == "__main__":
    main()