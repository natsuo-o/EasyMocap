'''
  @ Date: 2021-03-02 16:13:03
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-03 17:35:16
  @ FilePath: /EasyMocapPublic/apps/calibration/calib_extri.py
'''
from easymocap.mytools.camera_utils import write_intri
import os
from glob import glob
from os.path import join
import numpy as np
import cv2
from easymocap.mytools import read_intri, write_extri, read_json
from easymocap.mytools.debug_utils import mywarn
import LLFF.llff.poses.colmap_read_model as read_model


def init_intri(path, image):
    camnames = sorted(os.listdir(join(path, image)))
    cameras = {}
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, image, cam, '*.jpg')))
        assert len(imagenames) > 0
        imgname = imagenames[0]
        img = cv2.imread(imgname)
        height, width = img.shape[0], img.shape[1]
        focal = 1.2*max(height, width) # as colmap
        K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        dist = np.zeros((1, 5))
        cameras[cam] = {
            'K': K,
            'dist': dist
        }
    return cameras

def solvePnP(k3d, k2d, K, dist, flag, tryextri=False):
    k2d = np.ascontiguousarray(k2d[:, :2])
    # try different initial values:
    if tryextri:
        def closure(rvec, tvec):
            ret, rvec, tvec = cv2.solvePnP(k3d, k2d, K, dist, rvec, tvec, True, flags=flag)
            points2d_repro, xxx = cv2.projectPoints(k3d, rvec, tvec, K, dist)
            kpts_repro = points2d_repro.squeeze()
            err = np.linalg.norm(points2d_repro.squeeze() - k2d, axis=1).mean()
            return err, rvec, tvec, kpts_repro
        # create a series of extrinsic parameters looking at the origin
        height_guess = 2.1
        radius_guess = 7.
        infos = []
        for theta in np.linspace(0, 2*np.pi, 180):
            st = np.sin(theta)
            ct = np.cos(theta)
            center = np.array([radius_guess*ct, radius_guess*st, height_guess]).reshape(3, 1)
            R = np.array([
                [-st, ct,  0],
                [0,    0, -1],
                [-ct, -st, 0]
            ])
            tvec = - R @ center
            rvec = cv2.Rodrigues(R)[0]
            err, rvec, tvec, kpts_repro = closure(rvec, tvec)
            infos.append({
                'err': err,
                'repro': kpts_repro,
                'rvec': rvec,
                'tvec': tvec
            })
        infos.sort(key=lambda x:x['err'])
        err, rvec, tvec, kpts_repro = infos[0]['err'], infos[0]['rvec'], infos[0]['tvec'], infos[0]['repro']
    else:
        ret, rvec, tvec = cv2.solvePnP(k3d, k2d, K, dist, flags=flag)
        points2d_repro, xxx = cv2.projectPoints(k3d, rvec, tvec, K, dist)
        kpts_repro = points2d_repro.squeeze()
        err = np.linalg.norm(points2d_repro.squeeze() - k2d, axis=1).mean()
    # print(err)
    return err, rvec, tvec, kpts_repro


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def create_extri_yml(path):

    # 
    imagesfile = os.path.join(path, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    # imgdata = read_model.read_cameras_binary()

    camnames = sorted(os.listdir(join(path, "dataset_for_4k4d/images_libx265")))

    if len(camnames) != len(imdata):
        assert "the len of data extri and param is not same"
    
    # extriに外部パラメータの情報格納
    extri = {}
    for cam_num, im in imdata.items():
        cam = f'cam{cam_num:02}'
        extri[cam] = {}
        qvec = im.qvec
        R = qvec2rotmat(qvec)
        # R[:,2] -= 
        extri[cam]['R'] = R
        rvec = cv2.Rodrigues(R)[0]
        extri[cam]['Rvec'] = rvec
        T = im.tvec[:,np.newaxis]
        #T[1] += 1
        #T[-1] += 10
        extri[cam]['T'] = T
    output_path = join(path, 'dataset_for_4k4d/optimized/extri.yml')
    # extri = {"cam01":{'Rvec':[3,1], 'R':[3,3], 'T':[3,1]}}
    write_extri(output_path, extri)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    # path /workspace/data/kandao/KD_20240731_192530_MP4/convert_center_cam/
    parser.add_argument('--ext', type=str, default='.jpg')

    args = parser.parse_args()
    create_extri_yml(args.path)