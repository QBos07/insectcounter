from typing import List, cast

import cv2 as cv
import ffmpegcv
import numpy as np
from numpy._typing import NDArray


def do_calibration(cap: ffmpegcv.ffmpeg_noblock.ReadLiveLast, size: tuple[int, int], number_of_images = 50):
    # Arrays to store object points and image points from all the images.
    object_points: List[NDArray] = []  # 3d point in real world space
    image_points: List[NDArray] = []  # 2d points in image plane.
    while len(image_points) < number_of_images:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

        ret, corners = cv.findChessboardCorners(frame, size, None)

        if ret:
            object_points.append(objp)

            corners2 = cv.cornerSubPix(frame, corners, (11, 11), (-1, -1),
                                       (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            image_points.append(corners2)

            # Draw and display the corners
            frame = frame.copy()
            cv.drawChessboardCorners(frame, size, corners2, ret)
            print(len(image_points))

        cv.imshow('img', frame)

        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
    cap.close()
    return cv.calibrateCamera(object_points, image_points, cast(tuple[int, int], cap.size), cast(cv.Mat, None),
                              cast(cv.Mat, None))
