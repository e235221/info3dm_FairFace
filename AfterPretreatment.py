import os
import pandas as pd
import numpy as np
import shutil
import argparse
import logging
from pathlib import Path
import random


def setup_logging():
    """ロギングの設定を行う"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('upsampling.log')
        ]
    )


def upsample_images(input_dir, output_dir, csv_file, target_samples=100):
    """
    指定されたディレクトリ内の画像をアップサンプリングします。
    
    Args:
        input_dir (str): 入力画像ディレクトリ
        output_dir (str): 出力画像ディレクトリ
        csv_file (str): 画像のメタデータを含むCSVファイル
        target_samples (int): 各クラスの目標サンプル数
    """
    # 出力ディレクトリが存在する場合は削除
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logging.info(f"既存の出力ディレクトリを削除します: {output_dir}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir)
    logging.info(f"新しい出力ディレクトリを作成しました: {output_dir}")
    
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)
    
    # クラスごとの現在のサンプル数を表示
    class_counts = df['race'].value_counts()
    logging.info("\n各クラスの現在のサンプル数:")
    logging.info(f"\n{class_counts}")
    
    # アップサンプリング対象のクラスを取得
    target_classes = class_counts.index.tolist()
    logging.info("\nアップサンプリング対象のクラス:")
    logging.info(f"\n{target_classes}")
    
    # 各クラスに対してアップサンプリングを実行
    total_upsampled = 0
    for target_class in target_classes:
        logging.info(f"\nクラス {target_class} の処理中...")
        
        # 現在のクラスのサンプル数を取得
        current_samples = class_counts[target_class]
        logging.info(f"現在のサンプル数: {current_samples}")
        
        # 必要な複製数を計算
        needed_copies = target_samples - current_samples
        logging.info(f"必要な複製数: {needed_copies}")
        
        if needed_copies <= 0:
            logging.info(f"クラス {target_class} は既に十分なサンプル数があります。スキップします。")
            continue
        
        # 現在のクラスの画像ファイルを取得
        class_images = df[df['race'] == target_class]['face_name_align'].tolist()
        
        # ファイル名のみを抽出
        class_images = [os.path.basename(img_path) for img_path in class_images]
        
        # ランダムに画像を選択して複製
        for _ in range(needed_copies):
            # ランダムに画像を選択
            selected_image = random.choice(class_images)
            
            # 元の画像パス
            src_path = os.path.join(input_dir, selected_image)
            
            # 新しいファイル名を生成（元のファイル名に_dupを追加）
            base_name, ext = os.path.splitext(selected_image)
            new_filename = f"{base_name}_dup{ext}"
            dst_path = os.path.join(output_dir, new_filename)
            
            try:
                # 画像をコピー
                shutil.copy2(src_path, dst_path)
                total_upsampled += 1
            except FileNotFoundError:
                logging.warning(f"ファイルが見つかりません: {src_path}")
        
        logging.info(f"クラス {target_class} のアップサンプリング完了")
    
    # 最終的なクラスごとのサンプル数を表示
    final_counts = pd.Series([os.path.basename(f) for f in os.listdir(output_dir)]).value_counts()
    logging.info("\nアップサンプリング後の各クラスのサンプル数:")
    logging.info(f"\n{final_counts}")
    
    logging.info(f"\n合計 {total_upsampled} 個の画像をアップサンプリングしました")
    logging.info(f"アップサンプリング完了: {output_dir}に保存されました")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='画像のアップサンプリング')
    parser.add_argument('--input_dir', required=True, help='入力画像ディレクトリ')
    parser.add_argument('--output_dir', required=True, help='出力画像ディレクトリ')
    parser.add_argument('--csv', required=True, help='画像のメタデータを含むCSVファイル')
    parser.add_argument('--target_samples', type=int, default=100, help='各クラスの目標サンプル数')
    
    args = parser.parse_args()
    
    upsample_images(args.input_dir, args.output_dir, args.csv, args.target_samples) 
