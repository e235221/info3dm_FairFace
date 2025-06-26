import os
import random
from PIL import Image, ImageEnhance

SRC_DIR = 'AfterPretreatment'  # 元画像のディレクトリ
DST_DIR = 'AfterContrast'  # コントラスト調整画像の保存先ディレクトリ
IMG_EXTS = ('.jpg', '.jpeg', '.png')  # 対象拡張子
CONTRAST_RANGE = (10, 20)  # コントラスト係数の範囲を指定


def augment_contrast(src_dir, dst_dir, contrast_range=CONTRAST_RANGE):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        src_path = os.path.join(src_dir, fname)
        name, ext = os.path.splitext(fname)
        contrast_name = f"{name}_contrast{ext}"
        contrast_path = os.path.join(dst_dir, contrast_name)
        factor = random.uniform(contrast_range[0], contrast_range[1])
        try:
            with Image.open(src_path) as img:
                enhancer = ImageEnhance.Contrast(img)
                contrast_img = enhancer.enhance(factor)
                contrast_img.save(contrast_path)
                print(f"保存: {contrast_path} (コントラスト係数: {factor:.2f})")
        except Exception as e:
            print(f"エラー: {src_path} -> {e}")


def main():s
    augment_contrast(SRC_DIR, DST_DIR)


if __name__ == '__main__':
    main() 
