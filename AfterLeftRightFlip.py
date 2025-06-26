import os
import shutil
from PIL import Image, ImageOps

# 元画像ディレクトリ
SRC_DIR = 'AfterPretreatment'
# 反転画像保存先ディレクトリ
DST_DIR = 'AfterLeftRightFlip'

# 対象拡張子
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def augment_fliplr(src_dir, dst_dir):
    # 出力ディレクトリが存在する場合は削除
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
        print(f"既存の出力ディレクトリを削除しました: {dst_dir}")
    
    # 新しい出力ディレクトリを作成
    os.makedirs(dst_dir)
    print(f"新しい出力ディレクトリを作成しました: {dst_dir}")
    
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        if '_fliplr' in fname:
            continue  # 既に反転済み画像はスキップ
        src_path = os.path.join(src_dir, fname)
        name, ext = os.path.splitext(fname)
        flip_name = f"{name}_fliplr{ext}"
        flip_path = os.path.join(dst_dir, flip_name)
        
        try:
            with Image.open(src_path) as img:
                flipped_img = ImageOps.mirror(img)
                flipped_img.save(flip_path)
                print(f"保存: {flip_path}")
        except Exception as e:
            print(f"エラー: {src_path} -> {e}")


def main():
    augment_fliplr(SRC_DIR, DST_DIR)


if __name__ == '__main__':
    main() 

