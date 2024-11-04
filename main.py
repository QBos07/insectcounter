import time
from collections import deque
import os.path
from typing import cast
from dataclasses import dataclass, field

import ffmpegcv
import cv2 as cv
import yaml

from calibration import do_calibration
from calibrationsavedata import CalibrationSaveData
from configfiledata import ConfigFileData
from configfiledata import CameraConfigData
from to_data import to_data

config_file_path: str = "config.yml"
config: ConfigFileData = to_data(ConfigFileData, yaml.load(open("config.yml"), yaml.Loader))


@dataclass
class TimingData:
    frame_start: int = 0
    decode_end: int = 0
    remap_end: int = 0
    fps_end: int = 0
    show_end: int = 0


@dataclass
class CameraState:
    config: CameraConfigData
    csd: CalibrationSaveData
    cap: ffmpegcv.ffmpeg_noblock.ReadLiveLast
    maps: tuple[cv.Mat, cv.Mat]
    roi: tuple[int, int, int, int]
    timing: TimingData = field(default_factory=TimingData)


states: list[CameraState] = []

for cam_config in config.cams:

    if os.path.exists(cam_config.calib_data):
        print(f"using calibration data from {cam_config.calib_data} for {cam_config.name}")
        csd: CalibrationSaveData = yaml.load(open(cam_config.calib_data), yaml.Loader)
    else:
        print(f"Calibration file \"{cam_config.calib_data}\" for {cam_config.name} not found!")
        number_of_images: int = int(input("Number of images used for calibration: "))
        chessboard_size_one: int = int(input("Number of tiles on the first axis: "))
        chessboard_size_two: int = int(input("Number of tiles on the second axis: "))
        calib_cap = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, cam_config.url, pix_fmt = "gray")
        if not calib_cap.isOpened():
            print("Cannot open camera")
            exit()
        _, csd_matrix, csd_dist, _, _ = do_calibration(calib_cap, (chessboard_size_one, chessboard_size_two),
                                                       number_of_images)
        csd_calibration_size = calib_cap.size
        csd = CalibrationSaveData(csd_calibration_size, csd_matrix, csd_dist)
        yaml.dump(csd, stream = open(cam_config.calib_data, "w"))

    cap = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, cam_config.url, pix_fmt = config.pix_fmt)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(csd.matrix, csd.dist, csd.size, 1, cap.size)
    maps = cv.initUndistortRectifyMap(csd.matrix, csd.dist, cast(cv.Mat, None), new_camera_matrix,
                                      cast(tuple[int, int], cap.size), cv.CV_16SC2)

    states.append(CameraState(cam_config, csd, cap, maps, cast(tuple[int, int, int, int], roi)))

fps_roll = deque([], 50)

while True:
    frames_start = time.time_ns()
    for cam in states:
        new_times = TimingData()
        new_times.frame_start = time.time_ns()
        # Capture frame-by-frame
        ret, frame = cam.cap.read()
        # frame = cv.UMat(frame)  # GPU accel over opencl T-API
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        new_times.decode_end = time.time_ns()

        map_x, map_y = cam.maps
        x, y, w, h = cam.roi
        remapped = cv.remap(frame, map_x, map_y, cv.INTER_LINEAR)[y:y + h, x:x + w]
        new_times.remap_end = time.time_ns()

        decode_time = cam.timing.decode_end - cam.timing.frame_start
        remap_time = cam.timing.remap_end - cam.timing.decode_end
        fps_time = cam.timing.fps_end - cam.timing.remap_end
        show_time = cam.timing.show_end - cam.timing.fps_end
        frame_time = cam.timing.show_end - cam.timing.frame_start
        if len(fps_roll) != 0:
            fps = int(sum(fps_roll) / len(fps_roll) * 10) / 10
        else:
            fps = "undefined"
        cv.putText(remapped, f"Frame time: {int(frame_time / 1000) / 1000}ms", (10, 30), cv.FONT_HERSHEY_SIMPLEX, .75,
                   (150, 0, 150))
        cv.putText(remapped, f"FPS: {fps}", (10, 100), cv.FONT_HERSHEY_DUPLEX, 2.5, (150, 0, 150))
        cv.putText(remapped, f"Decode time: {int(decode_time / 1000) / 1000}ms", (10, 120), cv.FONT_HERSHEY_SIMPLEX,
                   .75,
                   (150, 0, 150))
        cv.putText(remapped, f"Remap time: {int(remap_time / 1000) / 1000}ms", (10, 140), cv.FONT_HERSHEY_SIMPLEX, .75,
                   (150, 0, 150))
        cv.putText(remapped, f"FPS time: {int(fps_time / 1000) / 1000}ms", (10, 160), cv.FONT_HERSHEY_SIMPLEX, .75,
                   (150, 0, 150))
        cv.putText(remapped, f"Show time: {int(show_time / 1000) / 1000}ms", (10, 180), cv.FONT_HERSHEY_SIMPLEX, .75,
                   (150, 0, 150))
        new_times.fps_end = time.time_ns()

        cv.imshow(cam.config.name, remapped)
        new_times.show_end = time.time_ns()

        cam.timing = new_times

    else:  # I wish python had labeled loops
        if cv.pollKey() == ord('q'):
            break
        frames_end = time.time_ns()
        frames_time = frames_end - frames_start
        current_fps = 1 / (frames_time / 1_000_000_000)
        fps_roll.append(current_fps)
        continue
    break

# When everything is done, release the capture
for cam in states:
    cam.cap.release()
cv.destroyAllWindows()
