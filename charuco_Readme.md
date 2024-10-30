# charucoを用いた特徴点マッチング

1. charuco自体のサンプルコードは`/workspace/EasyMocap/detect_charuco.py`を参照して

2. charucoを用いたチェッカーボード検出(`https://github.com/zju3dv/EasyMocap/blob/master/apps/calibration/Readme.md`が元コード)
    - `/workspace/EasyMocap/apps/calibration/detect_chessboard.py --out $/workspace/data/kandao/KD_20240731_193042_MP4/convert_center_cam/dataset_for_4k4d/output/calibration --pattern 5,7 --grid 0.1 --use_charuco`
        - `--pattern 5,7`は作成したcharucoによって変更白黒の行と列数を表している
        - `--use_charuco`を用いることでcharucoにおける検出が可能
        - `/workspace/data/kandao/KD_20240731_193042_MP4/convert_center_cam/dataset_for_4k4d/chessboard`が作成される
3. `python3 apps/calibration/calib_intri.py ${data} --step 5`と`python3 apps/calibration/calib_extri.py ${extri} --intri ${intri}/output/intri.yml`を用いて内部パラメータと外部パラメータを作成する