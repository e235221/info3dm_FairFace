import os
import argparse
import logging
from pathlib import Path
from PIL import Image
import shutil


def setup_logging():
    """ロギングの設定を行う"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('detected_faces_upsampling.log')
        ]
    )


def upsample_image(image_path, output_path, target_size=(512, 512), quality=95):
    """
    画像をアップサンプリングします（解像度を上げます）。
    
    Args:
        image_path (str): 入力画像のパス
        output_path (str): 出力画像のパス
        target_size (tuple): 目標サイズ (width, height)
        quality (int): JPEG品質 (1-100)
    """
    try:
        # PILを使用して画像を読み込み
        with Image.open(image_path) as img:
            # RGBに変換（RGBAの場合は背景を白にする）
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # アップサンプリング（解像度を上げる）
            img_upsampled = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 出力ディレクトリを作成
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 画像を保存（高品質で保存）
            img_upsampled.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        return True
    except Exception as e:
        logging.error(f"画像処理エラー {image_path}: {str(e)}")
        return False


def upsample_detected_faces(input_dir, output_dir, target_size=(512, 512), quality=95, 
                           max_files=None, overwrite=True):
    """
    detected_facesフォルダ内の画像をアップサンプリングします（解像度を上げます）。
    
    Args:
        input_dir (str): 入力画像ディレクトリ（detected_faces）
        output_dir (str): 出力画像ディレクトリ
        target_size (tuple): 目標サイズ (width, height)
        quality (int): JPEG品質 (1-100)
        max_files (int): 処理する最大ファイル数（Noneの場合は全て）
        overwrite (bool): 既存ファイルを上書きするかどうか
    """
    # 出力ディレクトリが存在し、上書きが指定されている場合は削除
    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logging.info(f"既存の出力ディレクトリを削除します: {output_dir}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"出力ディレクトリを作成しました: {output_dir}")
    
    # 入力ディレクトリ内の画像ファイルを取得
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in os.listdir(input_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    # ファイル数を制限
    if max_files is not None:
        image_files = image_files[:max_files]
    
    logging.info(f"処理対象ファイル数: {len(image_files)}")
    logging.info(f"目標サイズ: {target_size}")
    logging.info(f"JPEG品質: {quality}")
    
    # 各画像をアップサンプリング
    processed_count = 0
    error_count = 0
    
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        
        # ファイル名の拡張子を.jpgに統一
        base_name = Path(filename).stem
        output_path = os.path.join(output_dir, f"{base_name}.jpg")
        
        logging.info(f"処理中 ({i}/{len(image_files)}): {filename}")
        
        if upsample_image(input_path, output_path, target_size, quality):
            processed_count += 1
        else:
            error_count += 1
    
    logging.info(f"\nアップサンプリング完了:")
    logging.info(f"成功: {processed_count} ファイル")
    logging.info(f"エラー: {error_count} ファイル")
    logging.info(f"出力先: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='detected_facesフォルダの画像をアップサンプリング')
    parser.add_argument('--input_dir', default='detected_faces', help='入力画像ディレクトリ')
    parser.add_argument('--output_dir', default='detected_faces_upsampled', help='出力画像ディレクトリ')
    parser.add_argument('--width', type=int, default=512, help='目標幅')
    parser.add_argument('--height', type=int, default=512, help='目標高さ')
    parser.add_argument('--quality', type=int, default=95, help='JPEG品質 (1-100)')
    parser.add_argument('--max_files', type=int, help='処理する最大ファイル数')
    parser.add_argument('--no_overwrite', action='store_true', help='既存ファイルを上書きしない')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # 入力ディレクトリの存在確認
    if not os.path.exists(args.input_dir):
        logging.error(f"入力ディレクトリが存在しません: {args.input_dir}")
        return
    
    target_size = (args.width, args.height)
    overwrite = not args.no_overwrite
    
    upsample_detected_faces(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=target_size,
        quality=args.quality,
        max_files=args.max_files,
        overwrite=overwrite
    )


if __name__ == "__main__":
    main() 