import os
import random
from PIL import Image

SRC_DIR = 'AfterPretreatment'  # 元画像のディレクトリ
DST_DIR = 'AfterRotate'  # 回転画像の保存先ディレクトリ
IMG_EXTS = ('.jpg', '.jpeg', '.png')  # 対象拡張子
ANGLE_RANGE = (-15, 15)  # 回転角度の範囲（度）


def augment_rotate(src_dir, dst_dir, angle_range=ANGLE_RANGE):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        src_path = os.path.join(src_dir, fname)
        name, ext = os.path.splitext(fname)
        rotated_name = f"{name}_rotated{ext}"
        rotated_path = os.path.join(dst_dir, rotated_name)
        angle = random.uniform(angle_range[0], angle_range[1])
        try:
            with Image.open(src_path) as img:
                rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
                rotated_img.save(rotated_path)
                print(f"保存: {rotated_path} (角度: {angle:.2f}度)")
        except Exception as e:
            print(f"エラー: {src_path} -> {e}")


def main():
    augment_rotate(SRC_DIR, DST_DIR)


if __name__ == '__main__':
    main() 
