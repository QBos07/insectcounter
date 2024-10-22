import time
from collections import deque
import os.path

import ffmpegcv
import cv2 as cv
import yaml

from calibration import do_calibration
from calibrationsavedata import CalibrationSaveData

url: str = "rtsp://guestuser:guestuser@192.168.178.139:554"
cam_name: str = "cam 1"
calibration_data_path: str = "calibrationdata.yml"
cap = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, url, gpu = 0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

if os.path.exists(calibration_data_path):
    print(f"using calibration data from {calibration_data_path} for {cam_name}")
    csd: CalibrationSaveData = yaml.load(open(calibration_data_path), yaml.Loader)
    matrix = csd.matrix
    dist = csd.dist
    calibration_size = csd.size
else:
    print(f"Calibration file \"{calibration_data_path}\" for {cam_name} not found!")
    number_of_images: int = int(input("Number of images used for calibration: "))
    chessboard_size_one: int = int(input("Number of tiles on the first axis: "))
    chessboard_size_two: int = int(input("Number of tiles on the second axis: "))
    _, matrix, dist, _, _ = do_calibration(cap, (chessboard_size_one, chessboard_size_two), number_of_images)
    calibration_size = cap.size
    csd = CalibrationSaveData(calibration_size, matrix, dist)
    yaml.dump(csd, stream = open(calibration_data_path, "w"))

maps = None
roi = None

frame_start = 0
decode_end = 0
remap_end = 0
fps_end = 0
show_end = 0
input_end = 0
fps_roll = deque([], 50)

while True:
    frame_start_new = time.time_ns()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    decode_end_new = time.time_ns()

    if maps is None:
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(matrix, dist, calibration_size, 1, cap.size)
        maps = cv.initUndistortRectifyMap(matrix, dist, None, new_camera_matrix, cap.size, cv.CV_16SC2)

    map_x, map_y = maps
    x, y, w, h = roi
    remapped = cv.remap(frame, map_x, map_y, cv.INTER_LINEAR)[y:y + h, x:x + w]
    remap_end_new = time.time_ns()

    decode_time = decode_end - frame_start
    remap_time = remap_end - decode_end
    fps_time = fps_end - remap_end
    show_time = show_end - fps_end
    input_time = input_end - show_end
    frame_time = input_end - frame_start
    if frame_time != 0:
        current_fps = 1 / (frame_time / 1_000_000_000)
        fps_roll.append(current_fps)
        fps = int(sum(fps_roll) / len(fps_roll) * 10) / 10
    else:
        fps = "undefined"
    cv.putText(remapped, f"Frame time: {int(frame_time / 1000) / 1000}ms", (10, 30), cv.FONT_HERSHEY_SIMPLEX, .75,
               (150, 0, 150))
    cv.putText(remapped, f"FPS: {fps}", (10, 100), cv.FONT_HERSHEY_DUPLEX, 2.5, (150, 0, 150))
    cv.putText(remapped, f"Decode time: {int(decode_time / 1000) / 1000}ms", (10, 120), cv.FONT_HERSHEY_SIMPLEX, .75,
               (150, 0, 150))
    cv.putText(remapped, f"Remap time: {int(remap_time / 1000) / 1000}ms", (10, 140), cv.FONT_HERSHEY_SIMPLEX, .75,
               (150, 0, 150))
    cv.putText(remapped, f"FPS time: {int(fps_time / 1000) / 1000}ms", (10, 160), cv.FONT_HERSHEY_SIMPLEX, .75,
               (150, 0, 150))
    cv.putText(remapped, f"Show time: {int(show_time / 1000) / 1000}ms", (10, 180), cv.FONT_HERSHEY_SIMPLEX, .75,
               (150, 0, 150))
    cv.putText(remapped, f"Input time: {int(input_time / 1000) / 1000}ms", (10, 200), cv.FONT_HERSHEY_SIMPLEX, .75,
               (150, 0, 150))
    fps_end_new = time.time_ns()

    cv.imshow('img', remapped)
    show_end_new = time.time_ns()

    if cv.pollKey() == ord('q'):
        break

    input_end_new = time.time_ns()

    frame_start = frame_start_new
    decode_end = decode_end_new
    remap_end = remap_end_new
    fps_end = fps_end_new
    show_end = show_end_new
    input_end = input_end_new

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
