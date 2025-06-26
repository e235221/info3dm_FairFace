# 画像処理コード実行マニュアル

## 📋 概要

このプロジェクトでは、画像の前処理について「前処理前」と「前処理後」の2つのアプローチを実装しています。複数のAIに相談した結果、意見が分かれたため、両方を実装して精度や評価を比較検証することにしました。

### AI相談結果
- **前処理前を推奨**: ChatGPT、Perplexity
- **前処理後を推奨**: Gemini、Claude
- **両方のメリット・デメリットを提示**: Copilot

## 🖥️ 動作環境

```
Python: 3.10.9
PIL: 9.4.0
```

## ⚠️ 重要な注意事項

**Pythonコマンド実行時の注意**
- `python3`と入力する際、大文字が含まれているとエラーが発生します
- 正しい例: `python3 BeforeLeftRightFlip.py` ✅
- 間違った例: `Python3 BeforeLeftRightFlip.py` ❌

```bash
# エラー例
(base) nagase@nagaseisshounoMacBook-Air info3dm_FairFace % Python3 BeforeLeftRightFlip.py
Traceback (most recent call last):
  File "/Users/nagase/info3dm_FairFace/BeforeLeftRightFlip.py", line 2, in <module>
    from PIL import Image, ImageOps
ModuleNotFoundError: No module named 'PIL'
```

## 🎨 画像加工機能

### 1. 色調変化 (color_change.py)

**設定箇所**: 6行目〜9行目
```python
SRC_DIR = 'AfterPretreatment'          # 元画像のディレクトリ
DST_DIR = 'AfterColorJitter'           # 色調変化画像の保存先
IMG_EXTS = ('.jpg', '.jpeg', '.png')   # 対象拡張子
HUE_RANGE = (-0.1, 0.1)               # 色相シフトの範囲
```

**カスタマイズ方法**:
- ディレクトリ名を変更したい場合: シングルクォーテーション内を変更
- 色相を調整したい場合: 括弧内の数値を変更
- 注意: 色相範囲は-0.1〜0.1と-10〜10で大きな違いは見られませんでした

### 2. コントラスト調整 (contrast.py)

**設定箇所**: 5行目〜8行目
```python
SRC_DIR = 'AfterPretreatment'          # 元画像のディレクトリ
DST_DIR = 'AfterContrast'              # コントラスト調整画像の保存先
IMG_EXTS = ('.jpg', '.jpeg', '.png')   # 対象拡張子
CONTRAST_RANGE = (10, 20)              # コントラスト係数の範囲
```

**推奨設定**:
- コントラスト範囲: 0.7〜1.5（白飛びを防ぐため）

**カスタマイズ方法**:
- ディレクトリ名を変更したい場合: シングルクォーテーション内を変更
- 係数を調整したい場合: 括弧内の数値を変更

### 3. 画像回転 (rotate.py)

**設定箇所**: 5行目〜8行目
```python
SRC_DIR = 'AfterPretreatment'          # 元画像のディレクトリ
DST_DIR = 'AfterRotate'                # 回転画像の保存先
IMG_EXTS = ('.jpg', '.jpeg', '.png')   # 対象拡張子
ANGLE_RANGE = (-15, 15)                # 回転角度の範囲（度）
```

**カスタマイズ方法**:
- ディレクトリ名を変更したい場合: シングルクォーテーション内を変更
- 回転角度を調整したい場合: 括弧内の数値を変更

## 🔄 左右反転機能

### 前処理前の左右反転
```bash
python3 BeforeLeftRightFlip.py
```
- 保存先: `BeforeLeftRightFlip` フォルダ

### 前処理後の左右反転
```bash
python3 AfterLeftRightFlip.py
```
- 保存先: `AfterLeftRightFlip` フォルダ

## 📈 アップサンプリング機能

### 前処理前のアップサンプリング

**ファイル**: `BeforePretreatment.py`
**配置場所**: `/Users/nagase/info3dm_FairFace/BeforePretreatment.py`

**実行コマンド**:
```bash
python3 BeforePretreatment.py --csv "test_outputs.csv" --input_dir "test" --output_dir "BeforePretreatment" --target_samples 100
```

**結果**: `BeforePretreatment` フォルダにアップサンプリング済み画像が保存されます

### 前処理後のアップサンプリング

**ファイル**: `AfterPretreatment.py`
**配置場所**: FairFace階層に`AfterPretreatment.py`を配置

**実行コマンド**:
```bash
python3 AfterPretreatment.py --input_dir detected_faces --output_dir AfterPretreatment --csv test_outputs.csv
```

**結果**: `AfterPretreatment` フォルダにアップサンプリング済み画像が保存されます

## 📁 ディレクトリ構造

```
/Users/nagase/info3dm_FairFace/
├── BeforePretreatment.py
├── AfterPretreatment.py
├── BeforeLeftRightFlip.py
├── AfterLeftRightFlip.py
├── color_change.py
├── contrast.py
├── rotate.py
├── test/                    # 入力画像
├── detected_faces/          # 検出された顔画像
├── BeforePretreatment/      # 前処理前アップサンプリング結果
├── AfterPretreatment/       # 前処理後アップサンプリング結果
├── BeforeLeftRightFlip/     # 前処理前左右反転結果
├── AfterLeftRightFlip/      # 前処理後左右反転結果
├── AfterColorJitter/        # 色調変化結果
├── AfterContrast/           # コントラスト調整結果
└── AfterRotate/             # 回転処理結果
```

## 🎯 使用目的

このシステムは、画像処理における前処理のタイミングによる効果の違いを検証し、より良い精度と評価結果を得るために開発されました。両方のアプローチを比較することで、最適な手法を選択できます。