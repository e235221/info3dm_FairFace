from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
import argparse

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def reverse_resized_rect(rect,resize_ratio):
    l = int(rect.left() / resize_ratio)
    t = int(rect.top() / resize_ratio)
    r = int(rect.right() / resize_ratio)
    b = int(rect.bottom() / resize_ratio)
    new_rect = dlib.rectangle(l,t,r,b)

    return [l,t,r,b] , new_rect

# def detect_face(image_paths, SAVE_DETECTED_AT, default_max_size=800, size=300, padding=0.25):
def detect_face(image_paths, SAVE_DETECTED_AT, default_max_size=800, size=300, padding=1.25):
    # dlib の CNN ベースの顔検出モデルと5点ランドマーク検出器を読み込む
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # 最大幅・高さの基準値
    rects = []   # 元の画像上での顔のバウンディングボックスを保存するリスト

    for index, image_path in enumerate(image_paths):
        # 進捗表示: 1000 枚ごとに出力
        if index % 1000 == 0:
            print('---%d/%d---' % (index, len(image_paths)))
        
        # 画像をRGB形式で読み込む
        img = dlib.load_rgb_image(image_path)

        # 画像の元サイズを取得
        old_height, old_width, _ = img.shape
        # 画像をリサイズするための比率を計算し、新しい幅と高さを決定する
        if old_width > old_height:
            resize_ratio = default_max_size / old_width
            new_width, new_height = default_max_size, int(old_height * resize_ratio)
        else:
            resize_ratio = default_max_size / old_height
            new_width, new_height =  int(old_width * resize_ratio), default_max_size
        
        # dlib を使って画像をリサイズ
        img = dlib.resize_image(img, cols=new_width, rows=new_height)

        # 顔検出を実行 (アップサンプリング回数 1)
        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(image_path))
            continue

        # 各顔のランドマーク（顔の主要な点）を保存するためのリスト
        faces = dlib.full_object_detections()

        for detection in dets:
            # 検出された顔の矩形領域を取得
            rect = detection.rect
            # 顔領域の5点ランドマークを取得し、リストに追加
            faces.append(sp(img, rect))
            # リサイズ前の座標に逆変換して元のスケールのバウンディングボックスを取得
            rect_tpl, rect_in_origin = reverse_resized_rect(rect, resize_ratio)
            rects.append(rect_in_origin)
        
        # 顔の切り出し（クロッピング）と位置合わせを実施
        images = dlib.get_face_chips(img, faces, size=size, padding=padding)
        for idx, image in enumerate(images):
            # ファイル名から拡張子を抽出し、保存用のファイル名を生成する
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = os.path.join(SAVE_DETECTED_AT, path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            # 切り出した顔画像を指定ディレクトリに保存
            dlib.save_image(image, face_name) 

    # 元画像上での全顔のバウンディングボックスのリストを返す
    return rects

def predidct_age_gender_race(save_prediction_at, bboxes, imgs_path='cropped_faces/'):
    # 指定ディレクトリ内のすべての顔画像ファイルパスをリストにまとめる
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 事前学習済みのResNet34モデルを読み込み、最終層を属性分類用に変更する
    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    state_dict = torch.load('fair_face_models/res34_fair_align_multi_4_20190809.pt', map_location=device)
    model_fair_7.load_state_dict(state_dict)
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    # 4クラス版のモデルも同様に読み込みと準備をする
    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    state_dict = torch.load('fair_face_models/res34_fair_align_multi_4_20190809.pt', map_location=device)
    model_fair_4.load_state_dict(state_dict)
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    # 画像前処理のパイプライン（リサイズ、テンソル変換、正規化）を定義
    trans = transforms.Compose([
        transforms.ToPILImage(),                    # PIL形式に変換
        transforms.Resize((224, 224)),                # 224x224にリサイズ
        transforms.ToTensor(),                        # テンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

    # 結果格納用リストの初期化
    face_names = []
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    # 全顔画像に対して属性推論を実施
    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append(img_name)
        # 顔画像を読み込み、前処理を実施
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # バッチ次元を追加して4次元テンソルに変換
        image = image.to(device)

        # fair 7クラス版モデルによる属性予測
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        # 各属性の出力をスライスで分割
        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        # ソフトマックス関数を適用して各出力の確率を計算
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        # 最大の確率を持つクラスを予測値とする
        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        # 結果をリストに追加
        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)
        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # 同様の処理を 4クラス版モデルで実施
        outputs = model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)
        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)
        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    # 結果をDataFrameにまとめ、CSVとして保存する準備
    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           race_preds_fair_4,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair, race_scores_fair_4,
                           gender_scores_fair,
                           age_scores_fair,
                           bboxes]).T
    result.columns = ['face_name_align',
                      'race_preds_fair',
                      'race_preds_fair_4',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      'race_scores_fair_4',
                      'gender_scores_fair',
                      'age_scores_fair',
                      "bbox"]

    # クラス番号を具体的なカテゴリ名に変換
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    # 必要な列だけ選択し、CSVファイルとして保存
    result[['face_name_align',
            'race', 'race4',
            'gender', 'age',
            'race_scores_fair', 'race_scores_fair_4',
            'gender_scores_fair', 'age_scores_fair',
            "bbox"]].to_csv(save_prediction_at, index=False)
    print("saved results at ", save_prediction_at)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', dest='input_csv', action='store',
                        help='csv file of image path where col name for image path is "img_path')
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    args = parser.parse_args()
    SAVE_DETECTED_AT = "detected_faces"
    ensure_dir(SAVE_DETECTED_AT)
    imgs = pd.read_csv(args.input_csv)['img_path']
    bboxes = detect_face(imgs, SAVE_DETECTED_AT)
    print("detected faces are saved at ", SAVE_DETECTED_AT)
    predidct_age_gender_race("test_outputs.csv", bboxes, SAVE_DETECTED_AT)
