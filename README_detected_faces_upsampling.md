# detected_faces画像アップサンプリングスクリプト

このスクリプトは、detected_facesフォルダの顔検出済み画像をアップサンプリング（解像度を上げる）して、指定されたフォルダに保存します。

## 機能

- 顔検出済み画像の解像度を上げる（デフォルト: 512x512ピクセル）
- 高品質での保存（デフォルト: JPEG品質95）
- 複数の画像形式に対応（jpg, jpeg, png, bmp, tiff, tif）
- 上書きモード（既存フォルダを削除して新規作成）
- 詳細なログ出力

## 使用方法

### 基本的な使用方法

```bash
python upsample_detected_faces.py
```

デフォルト設定：
- 入力ディレクトリ: `detected_faces`
- 出力ディレクトリ: `detected_faces_upsampled`
- サイズ: 512x512ピクセル
- 品質: 95

### カスタム設定

```bash
# サイズと品質を指定
python upsample_detected_faces.py --width 1024 --height 1024 --quality 98

# 出力ディレクトリを指定
python upsample_detected_faces.py --output_dir my_detected_faces_upsampled

# 処理するファイル数を制限
python upsample_detected_faces.py --max_files 50

# 上書きしない（既存ファイルを保持）
python upsample_detected_faces.py --no_overwrite
```

### 全オプション

```bash
python upsample_detected_faces.py \
    --input_dir detected_faces \
    --output_dir detected_faces_upsampled \
    --width 512 \
    --height 512 \
    --quality 95 \
    --max_files 100 \
    --no_overwrite
```

## オプション説明

- `--input_dir`: 入力画像ディレクトリ（デフォルト: detected_faces）
- `--output_dir`: 出力画像ディレクトリ（デフォルト: detected_faces_upsampled）
- `--width`: 目標幅（デフォルト: 512）
- `--height`: 目標高さ（デフォルト: 512）
- `--quality`: JPEG品質 1-100（デフォルト: 95）
- `--max_files`: 処理する最大ファイル数（デフォルト: 全て）
- `--no_overwrite`: 既存ファイルを上書きしない

## 出力

- 処理された画像は指定された出力ディレクトリに保存されます
- 全ての画像は.jpg形式で保存されます
- ログファイル `detected_faces_upsampling.log` が作成されます
- コンソールにも処理状況が表示されます

## 注意事項

- 出力ディレクトリが存在する場合、`--no_overwrite`オプションを使用しない限り削除されます
- 元の画像ファイルは変更されません
- エラーが発生した画像はスキップされ、ログに記録されます
- アップサンプリングによりファイルサイズが大きくなります
- 高解像度画像のため、処理時間が長くなる可能性があります

## ファイル名の例

入力ファイル例：
- `000129_face0.jpg`
- `000098_face1.jpg`
- `000085_face6.jpg`

出力ファイル例：
- `000129_face0.jpg` (512x512にアップサンプリング)
- `000098_face1.jpg` (512x512にアップサンプリング)
- `000085_face6.jpg` (512x512にアップサンプリング)

## 解像度の比較

| 元のサイズ | アップサンプリング後 | ピクセル数増加 |
|------------|---------------------|----------------|
| 128×128 | 512×512 | 16倍 |
| 224×224 | 512×512 | 約5倍 |
| 256×256 | 512×512 | 4倍 | 