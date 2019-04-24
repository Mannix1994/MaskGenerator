# Introduction
Align face image and generate mask for face. Using dlib 
to detect bounding box of face and detect 68 landmarks 
of face. then using these landmarks to warp, crop and 
generate mask of face image.

# Install

* Clone or download this project an copy `mask` to your 
project.

* Install python libraries:
    ```bash
    pip install -r requirements.txt
    ```
    or python3
    ```bash
    pip3 install -r requirements.txt
    ```

* Download [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) 
and uncompress it.

# Demo

```python
# coding=utf8
from __future__ import print_function
from mask import MaskGenerator
import cv2

if __name__ == '__main__':
    import glob

    LANDMARK_PATH = 'shape_predictor_68_face_landmarks.dat'

    images = glob.glob('Images/*.*')
    mask_gen = MaskGenerator(LANDMARK_PATH, detector_version=1)
    for _path in images:
        _im = cv2.imread(_path)
        ret = mask_gen.align(_im, warp=True, crop=True, resize=True, align_multi=True, draw_landmarks=True,
                             size=(128, 128), scale=1.5, crop_function_version=0)
        for i, (tag, mask, face, landmark) in enumerate(ret):
            print('Detected face: %s, index： %d' % (tag, i))
            cv2.imshow('mask', mask)
            cv2.imshow('image', face)
            if landmark is not None:
                cv2.imshow('landmark', landmark)
            if cv2.waitKey(0) == 27:
                exit()

```

* `__init__` take two parameter:
    * landmarks_path: the path of `shape_predictor_68_face_landmarks.dat`.
    * detector_version: define which detector used to to detect 
    bounding box of face. detector_version = 1 for using 
    `dlib.cnn_face_detection_model_v1` 
    ; detector_version = 1 for using 
    `dlib.get_frontal_face_detector`. `dlib.cnn_face_detection_model_v1`
    can detect more bounding boxes than `dlib.get_frontal_face_detector`.
    
* `align` take nine parameter:

    * image: a BGR format face image
    * size: target size, if resize is True, 'face' and 'mask'
    will be resize to `size`.
    * scale: the padding of face image.
    * warp: whether warp face and mask or not
    * crop: whether crop face and mask or not
    * resize: whether resize face and mask or not
    * crop_function_version: if crop_function_version is `0`, 
    `align` will try to detect all face in image. if 
    crop_function_version is `1`, `align` just try to detect
    one face image. `0` is faster than `1`.
    * align_multi: whether try to detect all face in image.
    * draw_landmarks: whether draw face landmarks and return 
    face landmark image.

# License

[WTFPL](http://www.wtfpl.net/)