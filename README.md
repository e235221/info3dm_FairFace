# FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age
注. 山脇が勝手に翻訳しました。

https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf

Karkkainen, K., & Joo, J. (2021).FairFace：バイアス測定と緩和のための人種、性別、年齢のバランスのとれた顔属性データセット。Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1548-1558).

### 論文で我々のデータセットやモデルを使用する場合は、引用してください：
```
 @inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021},
  pages={1548--1558}
}
```

### Examples of FairFace Prediction
![](https://github.com/dchen236/FairFace/blob/master/examples/female.png)
![](https://github.com/dchen236/FairFace/blob/master/examples/male.png)

### Instructions to use FairFace

- このリポジトリをダウンロードまたはクローンする
- 依存関係をインストールする
   1.Pytorch のインストールは [Pytorch's official documentation](https://pytorch.org/get-started/locally/) に従ってください。
   2.pipがインストールされている場合はdlibもインストールしてください。ターミナルで以下のコマンドを入力してください。
   ```
   pip install dlib
   ```
- Download our models
   Download our pretrained models from [here](https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu?usp=sharing) and save it in the same folder as where predict.py is located. Two models are included, race_4 model predicts race as White, Black, Asian and Indian and race_7 model predicts races as White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern.
- Unzip the downloaded FairFace model as well as dlib face detection models in dlib_models.
- Prepare the images
   - prepare a csv and provide the paths of testing images where the colname name of testing images is "img_path" (see our [template csv file](https://github.com/dchen236/FairFace/blob/master/test_imgs.csv).
### スクリプト predict.py を実行します。
predict.pyスクリプトを実行し、csvパスを指定します。
```
python3 predict.py --csv "NAME_OF_CSV"
```
このリポジトリをダウンロードした後、`python3 predict.py --csv test_imgs.csv` を実行すると、detected_faces (dlibが1つの画像から複数の顔を検出した場合、ここに保存されます) と test_outputs.csv に結果が表示されます。
#### 結果
結果は "test_outputs.csv"(predict.pyと同じフォルダにあります。サンプル[こちら](https://github.com/dchen236/FairFace/blob/master/test_outputs.csv))に保存されます。

### UPDATES：

### スクリプト predict_bbox.py を実行します。
 predict.pyと同じコマンドで、出力csvには検出された顔のバウンディングボックスである "bbox "カラムが追加されます。
```
python3 predict_bbox.py --csv "名前_OF_CSV"
```
 

##### 出力ファイルのドキュメント
タイプするインデックス
- race_scores_fair (モデルの信頼スコア) [白人、黒人、ラテン系ヒスパニック、東アジア、東南アジア、インド、中東].
- race_scores_fair_4 (モデルの信頼スコア) [白人、黒人、アジア系、インド系].
- 性別_scores_fair（モデル信頼スコア）［男性、女性］
- age_scores_fair（モデル信頼スコア）［0-2、3-9、10-19、20-29、30-39、40-49、50-59、60-69、70+］。


### データ
画像（訓練セット＋検証セット）：[パディング=0.25](https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view), [パディング=1.25](https://drive.google.com/file/d/1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL/view)

dlibのget_face_chip()を使用して、主実験ではpadding=0.25（余白が少ない）、商用APIのバイアス測定実験ではpadding=1.25で、顔の切り抜きと位置合わせを行った。
ラベル[訓練](https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view), [検証](https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view)

ライセンスCC BY 4.0

### 注意事項
モデルとスクリプトは8GbのGPUを搭載したデバイスでテストされ、テストフォルダ内の5つの画像を予測するのに2秒かからなかった。
