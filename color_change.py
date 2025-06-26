import os
import random
from PIL import Image
import numpy as np

SRC_DIR = 'AfterPretreatment'  # 元画像のディレクトリ
DST_DIR = 'AfterColorJitter'  # 色調変化画像の保存先ディレクトリ
IMG_EXTS = ('.jpg', '.jpeg', '.png')  # 対象拡張子
HUE_RANGE = (-0.1, 0.1)  # 色相シフトの範囲（-0.1～0.1）


def shift_hue(img, hue_shift):
    """PIL画像の色相をシフトする"""
    img = img.convert('RGB')
    arr = np.array(img).astype(np.uint8)
    hsv = np.array(Image.fromarray(arr).convert('HSV'))
    # 色相をシフト
    hsv[..., 0] = (hsv[..., 0].astype(int) + int(hue_shift * 255)) % 256
    img_shifted = Image.fromarray(hsv, 'HSV').convert('RGB')
    return img_shifted


def augment_color_jitter(src_dir, dst_dir, hue_range=HUE_RANGE):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        src_path = os.path.join(src_dir, fname)
        name, ext = os.path.splitext(fname)
        jitter_name = f"{name}_colorjitter{ext}"
        jitter_path = os.path.join(dst_dir, jitter_name)
        hue_shift = random.uniform(hue_range[0], hue_range[1])
        try:
            with Image.open(src_path) as img:
                jitter_img = shift_hue(img, hue_shift)
                jitter_img.save(jitter_path)
                print(f"保存: {jitter_path} (色相シフト: {hue_shift:.3f})")
        except Exception as e:
            print(f"エラー: {src_path} -> {e}")


def main():
    augment_color_jitter(SRC_DIR, DST_DIR)


if __name__ == '__main__':
    main() 
