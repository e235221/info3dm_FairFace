# 画像ダウンサンプリングスクリプト

このスクリプトは、testフォルダの画像をダウンサンプリングして、指定されたフォルダに保存します。

## 機能

- 画像のリサイズ（デフォルト: 224x224ピクセル）
- JPEG品質の調整（デフォルト: 85）
- 複数の画像形式に対応（jpg, jpeg, png, bmp, tiff, tif）
- 上書きモード（既存フォルダを削除して新規作成）
- 詳細なログ出力

## 使用方法

### 基本的な使用方法

```bash
python downsample_test_images.py
```

デフォルト設定：
- 入力ディレクトリ: `test`
- 出力ディレクトリ: `test_downsampled`
- サイズ: 224x224ピクセル
- 品質: 85

### カスタム設定

```bash
# サイズと品質を指定
python downsample_test_images.py --width 128 --height 128 --quality 70

# 出力ディレクトリを指定
python downsample_test_images.py --output_dir my_downsampled_images

# 処理するファイル数を制限
python downsample_test_images.py --max_files 50

# 上書きしない（既存ファイルを保持）
python downsample_test_images.py --no_overwrite
```

### 全オプション

```bash
python downsample_test_images.py \
    --input_dir test \
    --output_dir test_downsampled \
    --width 224 \
    --height 224 \
    --quality 85 \
    --max_files 100 \
    --no_overwrite
```

## オプション説明

- `--input_dir`: 入力画像ディレクトリ（デフォルト: test）
- `--output_dir`: 出力画像ディレクトリ（デフォルト: test_downsampled）
- `--width`: 目標幅（デフォルト: 224）
- `--height`: 目標高さ（デフォルト: 224）
- `--quality`: JPEG品質 1-100（デフォルト: 85）
- `--max_files`: 処理する最大ファイル数（デフォルト: 全て）
- `--no_overwrite`: 既存ファイルを上書きしない

## 出力

- 処理された画像は指定された出力ディレクトリに保存されます
- 全ての画像は.jpg形式で保存されます
- ログファイル `downsampling.log` が作成されます
- コンソールにも処理状況が表示されます

## 注意事項

- 出力ディレクトリが存在する場合、`--no_overwrite`オプションを使用しない限り削除されます
- 元の画像ファイルは変更されません
- エラーが発生した画像はスキップされ、ログに記録されます 