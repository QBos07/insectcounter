import atexit
import time
from collections import deque
import os.path
from typing import cast, Sequence
from dataclasses import dataclass, field

import ffmpegcv
import cv2 as cv
import numpy as np
import yaml

from calibration import do_calibration
from calibrationsavedata import CalibrationSaveData
from configfiledata import ConfigFileData
from configfiledata import CameraConfigData
from to_data import to_data
from check_config import check_config

atexit.register(cv.destroyAllWindows)

config_file_path: str = "config.yml"
config: ConfigFileData = to_data(ConfigFileData, yaml.load(open("config.yml"), yaml.Loader))
check_config(config)


@dataclass
class TimingData:
    decode_duration: int = 0
    remap_duration: int = 0
    resize_duration: int = 0
    orb_duration: int = 0


@dataclass
class LinkTimingData:
    bf_matcher_duration: int = 0


@dataclass
class CameraState:
    config: CameraConfigData
    csd: CalibrationSaveData
    cap: ffmpegcv.ffmpeg_noblock.ReadLiveLast
    maps: tuple[cv.Mat, cv.Mat]
    roi: tuple[int, int, int, int]
    frames: dict[tuple[int, int] | None, cv.Mat] = field(default_factory = dict)  # None is raw output
    timing: TimingData = field(default_factory = TimingData)
    new_times: TimingData = field(default_factory = TimingData)
    orb: dict[tuple[int, int], tuple[Sequence[cv.KeyPoint], cv.Mat]] = field(default_factory = dict)


@dataclass
class LinkState:
    link: tuple[tuple[str, CameraState], tuple[str, CameraState]]
    size: tuple[int, int]
    bf_matches: list[cv.DMatch] = field(default_factory = list)
    timing: LinkTimingData = field(default_factory = LinkTimingData)
    new_times: LinkTimingData = field(default_factory = LinkTimingData)


states: dict[str, CameraState] = dict()

for name, cam_config in config.cams.items():
    # type inference shat itself here
    name: str
    cam_config: CameraConfigData

    if name not in set([name for link in config.links for name in link]):
        continue

    if os.path.exists(cam_config.calib_data):
        print(f"using calibration data from {cam_config.calib_data} for {name}")
        csd: CalibrationSaveData = yaml.load(open(cam_config.calib_data), yaml.Loader)
    else:
        print(f"Calibration file \"{cam_config.calib_data}\" for {name} not found!")
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
    atexit.register(cap.close)

    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(csd.matrix, csd.dist, csd.size, 1, cap.size)
    maps = cv.initUndistortRectifyMap(csd.matrix, csd.dist, cast(cv.Mat, None), new_camera_matrix,
                                      cast(tuple[int, int], cap.size), cv.CV_16SC2)

    states[name] = CameraState(cam_config, csd, cap, maps, cast(tuple[int, int, int, int], roi))

# cast because typechecking shits itself without loop-unrolling because tuples in python are unsized
links: list[tuple[tuple[str, CameraState], tuple[str, CameraState]]] = [
    cast(tuple[tuple[str, CameraState], tuple[str, CameraState]], tuple(
        [
            (
                name,
                states[name]
            )
            for name in link
        ]
    ))
    for link in config.links
]

needed_sizes: dict[str, set[tuple[int, int]]] = dict()
link_sizes: dict[tuple[str, str], tuple[int, int]] = dict()

for cam in states:
    needed_sizes[cam] = set()

for link in links:
    sizes = [cam[1].roi[2:4] for cam in link]
    xs = [size[0] for size in sizes]
    ys = [size[1] for size in sizes]
    x = sum(xs) // len(xs)
    y = sum(ys) // len(ys)
    size = (x, y)
    for cam in [cam[0] for cam in link]:
        needed_sizes[cam].add(size)
    link_sizes[(link[0][0], link[1][0])] = size

link_states: list[LinkState] = [LinkState(link, link_sizes[(link[0][0], link[1][0])]) for link in links]

fps_roll = deque([], 50)


def crop_to_aspect_ratio(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    src_height, src_width = image.shape[:2]
    target_aspect_ratio = target_width / target_height
    src_aspect_ratio = src_width / src_height

    if src_aspect_ratio > target_aspect_ratio:  # Image is wider
        new_width = int(src_height * target_aspect_ratio)
        x_start = (src_width - new_width) // 2
        cropped = image[:, x_start:x_start + new_width]
    else:  # Image is taller or equal
        new_height = int(src_width / target_aspect_ratio)
        y_start = (src_height - new_height) // 2
        cropped = image[y_start:y_start + new_height, :]

    return cropped


orb = cv.ORB_create()

while True:
    frames_start = time.time_ns()
    for name, cam in states.items():
        decode_start: int = time.time_ns()
        ret, frame = cam.cap.read()
        # frame = cv.UMat(frame)
        cam.frames[None] = frame
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cam.new_times.decode_duration = time.time_ns() - decode_start

    for _, cam in states.items():
        remap_start = time.time_ns()
        map_x, map_y = cam.maps
        x, y, w, h = cam.roi
        frame = cast(
            cv.Mat,
            cv.remap(cam.frames[None], map_x, map_y, cv.INTER_LINEAR)[y:y + h, x:x + w]
        )
        # frame = cv.UMat(frame)
        cam.frames[cam.roi[2:4]] = frame
        cam.new_times.remap_duration = time.time_ns() - remap_start

    for name, cam in states.items():
        resize_start = time.time_ns()
        for size in needed_sizes[name]:
            if size == cam.cap.size:
                continue
            cropped_image = crop_to_aspect_ratio(cam.frames[cam.roi[2:4]], *size)
            resized_image = cv.resize(cropped_image, size, interpolation = cv.INTER_LINEAR)
            # resized_image = cv.UMat(resized_image)
            cam.frames[size] = resized_image
        cam.new_times.resize_duration = time.time_ns() - resize_start

    for name, cam in states.items():
        orb_start = time.time_ns()
        for size in needed_sizes[name]:
            res = orb.detectAndCompute(cam.frames[size], None)
            cam.orb[size] = res
        cam.new_times.orb_duration = time.time_ns() - orb_start

    for link in link_states:
        bf_matcher_start = time.time_ns()
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
        matches = bf.match(link.link[0][1].orb[link.size][1], link.link[0][1].orb[link.size][1])

        # Filter matches using Lowe's ratio test
        # good_matches: list[cv.DMatch] = []
        # for m, n in matches:
        #    if m.distance < 0.75 * n.distance:
        #        good_matches.append(m)

        link.bf_matches = matches
        link.new_times.bf_matcher_duration = time.time_ns() - bf_matcher_start

    for link in link_states:
        img_matches = np.empty((link.size[1], link.size[0] * 2, 3), dtype = np.uint8)
        cv.drawMatches(link.link[0][1].frames[link.size], link.link[0][1].orb[link.size][0],
                       link.link[1][1].frames[link.size], link.link[1][1].orb[link.size][0],
                       link.bf_matches, img_matches, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow(f"{link.link[0][0]} + {link.link[1][0]}: Matches", img_matches)

    for name, cam in states.items():
        cam.timing = cam.new_times
        print(name, cam.timing, flush = False)

    print("-----------------", flush = False)

    for link in link_states:
        link.timing = link.new_times
        print([(link[0][0], link[1][0]) for link in links], link.timing, flush = False)

    print("-----------------", flush = False)

    frames_end = time.time_ns()
    frames_time = frames_end - frames_start
    current_fps = 1 / (frames_time / 1_000_000_000)
    fps_roll.append(current_fps)

    if len(fps_roll) != 0:
        fps = int(sum(fps_roll) / len(fps_roll) * 10) / 10
    else:
        fps = "undefined"

    print(fps)

    print(flush = False)
    print("=================", flush = False)
    print()

    if cv.pollKey() == ord("q"):
        break

    continue

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
for _, cam in states.items():
    cam.cap.release()
cv.destroyAllWindows()
