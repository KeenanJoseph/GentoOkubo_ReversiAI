import openpyxl
from tkinter import Tk, filedialog
import os

# ファイル選択ダイアログを表示
root = Tk()
root.withdraw()  # 余計なウィンドウを表示しない
input_file = filedialog.askopenfilename(
    title="テキストファイルを選択してください",
    filetypes=[("テキストファイル", "*.txt")]
)

if not input_file:
    print("ファイルが選択されませんでした。")
    exit()

# 出力ファイル名を作成（拡張子を.xlsxに変更）
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = os.path.join(os.path.dirname(input_file), f"{base_name}.xlsx")

# Excel用のワークブックを作成
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "棋譜"

# ヘッダーを追加
ws.append(["手番", "プレイヤー", "行", "列"])

# テキストデータを解析
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

turn = 1  # 手数のカウント
summary_ws = None  # 結果シート
for line in lines:
    line = line.strip()
    if ": " in line and "(" in line and ")" in line:  # 座標データを判定
        try:
            player, coords = line.split(": ")
            coords = coords.strip("()").split(", ")
            row, col = int(coords[0]), int(coords[1])
            ws.append([turn, player, row, col])
            turn += 1
        except (ValueError, IndexError):
            print(f"エラー: 行の解析に失敗しました: {line}")
    elif "最終スコア" in line or "勝者" in line:  # スコアや勝者の処理
        if summary_ws is None:
            summary_ws = wb.create_sheet("結果")
            summary_ws.append(["項目", "値"])
        if "最終スコア" in line:
            summary_ws.append(["最終スコア", line.replace("最終スコア:", "").strip()])
        elif "勝者" in line:
            summary_ws.append(["勝者", line.replace("勝者:", "").strip()])

# Excelファイルを保存
wb.save(output_file)
print(f"Excelファイルに保存しました: {output_file}")
