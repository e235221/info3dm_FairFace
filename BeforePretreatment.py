import os
import pandas as pd
import numpy as np
import shutil
import argparse


def upsample_minority_classes(input_dir, output_dir, csv_file, target_samples=100):
    """
    testディレクトリの画像をアップサンプリングする関数
    
    Parameters:
    -----------
    input_dir : str
        入力画像のディレクトリ（testディレクトリ）
    output_dir : str
        アップサンプリング後の画像を保存するディレクトリ
    csv_file : str
        画像の属性情報が含まれるCSVファイル
    target_samples : int
        各クラスの目標サンプル数
    """
    # 出力ディレクトリが存在する場合は削除
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # 新しい出力ディレクトリを作成
    os.makedirs(output_dir)
    print(f"新しい出力ディレクトリを作成しました: {output_dir}")
    
    # CSVファイルから画像の属性情報を読み込み
    results = pd.read_csv(csv_file)
    
    # 各クラスのサンプル数をカウント
    class_counts = results['race'].value_counts()
    print("各クラスの現在のサンプル数:")
    print(class_counts)
    
    # 少数クラスの特定（サンプル数がtarget_samples未満のクラス）
    minority_classes = class_counts[class_counts < target_samples].index
    print("\nアップサンプリング対象のクラス:")
    print(minority_classes)
    
    for minority_class in minority_classes:
        print(f"\nクラス {minority_class} の処理中...")
        # 少数クラスの画像を取得
        minority_images = results[results['race'] == minority_class]['face_name_align'].tolist()
        
        # 必要な複製数を計算
        current_count = len(minority_images)
        needed_copies = target_samples - current_count
        
        if needed_copies > 0:
            print(f"現在のサンプル数: {current_count}")
            print(f"必要な複製数: {needed_copies}")
            
            # ランダムに画像を選択して複製
            images_to_copy = np.random.choice(minority_images, size=needed_copies, replace=True)
            
            for idx, img_path in enumerate(images_to_copy):
                # 元の画像のパスを構築
                base_name = os.path.basename(img_path)
                
                # ファイル名から元の画像名を抽出
                if '_face' in base_name:
                    # 前処理済み画像の場合（例：000116_face0.jpg → 000116.jpg）
                    original_name = base_name.split('_face')[0] + '.jpg'
                else:
                    # 元の画像の場合（例：000104.jpg）
                    original_name = base_name
                
                original_path = os.path.join(input_dir, original_name)
                
                # 新しいファイル名を生成
                name, ext = os.path.splitext(original_name)
                new_name = f"{name}_upsampled_{idx}{ext}"
                new_path = os.path.join(output_dir, new_name)
                
                try:
                    # 画像をコピー
                    shutil.copy2(original_path, new_path)
                except FileNotFoundError:
                    print(f"警告: ファイルが見つかりません: {original_path}")
                    continue
            
            print(f"クラス {minority_class} のアップサンプリング完了")
    
    # 最終的な各クラスのサンプル数を表示
    final_counts = results['race'].value_counts()
    print("\nアップサンプリング後の各クラスのサンプル数:")
    print(final_counts)
    print(f"\nアップサンプリング完了: {output_dir}に保存されました")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='testディレクトリの画像をアップサンプリング')
    parser.add_argument('--input_dir', type=str, default='test',
                        help='入力画像のディレクトリ（testディレクトリ）')
    parser.add_argument('--output_dir', type=str, default='upsampled_faces',
                        help='アップサンプリング後の画像を保存するディレクトリ')
    parser.add_argument('--csv', type=str, required=True,
                        help='画像の属性情報が含まれるCSVファイル')
    parser.add_argument('--target_samples', type=int, default=100,
                        help='各クラスの目標サンプル数')
    
    args = parser.parse_args()
    
    upsample_minority_classes(
        args.input_dir,
        args.output_dir,
        args.csv,
        args.target_samples
    )
