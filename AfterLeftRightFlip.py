import os
from PIL import Image, ImageOps

# 元画像ディレクトリ
SRC_DIR = 'AfterPretreatment'
# 反転画像保存先ディレクトリ
DST_DIR = 'AfterLeftRightFlip'

# 対象拡張子
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def augment_fliplr(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        if '_fliplr' in fname:
            continue  # 既に反転済み画像はスキップ
        src_path = os.path.join(src_dir, fname)
        name, ext = os.path.splitext(fname)
        flip_name = f"{name}_fliplr{ext}"
        flip_path = os.path.join(dst_dir, flip_name)
        if os.path.exists(flip_path):
            continue  # 既に存在する場合はスキップ
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

