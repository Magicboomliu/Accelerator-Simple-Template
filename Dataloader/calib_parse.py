import numpy as np


def parse_calib(mode, calib_path=None):
    if mode == 'raw':
        calib_cam2cam_path = calib_path

        with open(calib_cam2cam_path, encoding='utf-8') as f:
            text = f.readlines()
            P2 = np.array(text[-9].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P3 = np.array(text[-1].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            
            R_rect = np.zeros((4, 4))
            R_rect_tmp = np.array(text[8].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            R_rect[:3, :3] = R_rect_tmp
            R_rect[3, 3] = 1


    calib = {
        'P2': P2,
        'P3': P3
    }
    return calib