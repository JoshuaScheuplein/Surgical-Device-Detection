import logging

import cv2
import numpy as np

from scipy.ndimage import gaussian_filter


def preprocessing(_img, subtract_lowpass=False, clahe=False, clahe_clip_limit=3., clahe_window=(8, 8),
                                 return_uint8=False):
    
    # asserting assumptions
    assert _img.ndim == 2
    _img = _img.astype(np.float64) # cast to higher bit-depth to avoid data loss during rescaling

    # preprocess only inner region (1/4 image margin) (avoids pixel-errors at boarder)
    margin = _img.shape[0] // 8
    ROI = _img[margin:-margin, margin:-margin]
    cval = np.median(ROI) + 3 * np.std(ROI) # set contrast window upper limit to median + 3*std
    if cval == 0:
        logging.warning("cval is zero ...")
        return np.zeros_like(_img, dtype=np.uint8)
    _img = np.minimum(_img, cval) / cval # clip and normalize to 1

    # apply neglog transform
    _img = -np.log(np.maximum(_img, np.finfo(dtype=_img.dtype).eps))

    # apply histogram equalization
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_window)
        _img = clahe.apply(_img)

    # calculate lowpass component and subtract to mitigate intensity gradients
    if subtract_lowpass:
        physical_kernel_size = 20 # mm
        lowpass = gaussian_filter(_img, sigma=(physical_kernel_size/0.305) * (_img.shape[0]/976))
        _img -= lowpass

    # convert to np.float32
    _img = _img.astype(np.float32)

    # scale contrast to [0, 255] and cast to uint8
    ROI = _img[margin:-margin, margin:-margin]
    _img = (_img - ROI.min()) / (ROI.max() - ROI.min())
    if return_uint8:
        _img = (np.clip(_img * 255, 0, 255)).astype(np.uint8)
    else:
        # clip everything outside the ROI to [0, 1]
        _img = np.clip(_img, 0, 1)

    return _img
