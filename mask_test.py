from mask import MaskGenerator
import cv2

if __name__ == '__main__':
    import glob

    LANDMARK_PATH = '/home/creator/Projects/DL/SfSNet-Pytorch/data/shape_predictor_68_face_landmarks.dat'

    images = glob.glob('Images/*.*')
    mask_gen = MaskGenerator(LANDMARK_PATH)
    for _path in images:
        _im = cv2.imread(_path)
        ret = mask_gen.align(_im, warp=True, crop=True, resize=True, align_multi=True, draw_landmarks=False,
                             size=(128, 128), scale=1.5, crop_function_version=1)
        for i, (tag, mask, face, landmark) in enumerate(ret):
            print('Detected face: %s, index： %d' % (tag, i))
            cv2.imshow('mask', mask)
            cv2.imshow('image', face)
            if landmark is not None:
                cv2.imshow('landmark', landmark)
            if cv2.waitKey(0) == 27:
                exit()
