Q.なぜ、前処理前のコード前処理後のコードで実装したのか？

A.ChatGPT,Gemini,perplecity,copliot,cluadeに聞いた結果perprecity,GPTが前処理前、gemini,cludeが前処理後、copliotが両方のメリットデメリットを解説した上でユーザーに選択させる方式をとり、意見が分かれたので両方実装することにより、精度や評価といった部分で差異を確認し良い方を使用する。

※注意
実行コマンドで、python3と入力するとき、
どこかしらの単語が大文字だとエラーが起こるので気をつけて下さい
(例)
(base) nagase@nagaseisshounoMacBook-Air info3dm_FairFace % python3 BeforeLeftRightFlip.py
(base) nagase@nagaseisshounoMacBook-Air info3dm_FairFace % Python3 BeforeLeftRightFlip.py
Traceback (most recent call last):
  File "/Users/nagase/info3dm_FairFace/BeforeLeftRightFlip.py", line 2, in <module>
    from PIL import Image, ImageOps
ModuleNotFoundError: No module named 'PIL'

構築環境
Python 3.10.9
PIL Version: 9.4.0

左右反転画像
前処理前
python3 BeforeLeftRightFlip.pyでBeforeLeftRightFlipファイルに保存される

前処理後
python3 AfterLeftRightFlip.py で
AfterLeftRightFlipファイルに保存される。

アップサンプリングコード

前処理前のコードはBeforePretreatment.pyで実行可能で階層構造は
/Users/nagase/info3dm_FairFace
であり、FairFace階層にBeforePretreatment.pyを追加して下さい。
その後に、
 python3 BeforePretreatment.py --csv "test_outputs.csv" --input_dir "test" --output_dir "BeforePretreatment" --target_samples 100
このコマンドを実行し、BeforePretreatmentにアップサンプリングされた画像が保存される。

前処理後のコードは、AfterPretreatment.pyで実行可能で階層構造はFairFaceの階層にAfterPretreatment.pyを追加して下さい。
その後に、
python3 AfterPretreatment.py --input_dir detected_faces --output_dir AfterPretreatment --csv test_outputs.csv  
  このコマンドを実行し、AfterPretreatmentにアップサンプリングされた画像が保存される