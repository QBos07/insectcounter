from dataclasses import dataclass


@dataclass
class CameraConfigData:
    url: str
    name: str
    calib_data: str


@dataclass
class ConfigFileData:
    cams: list[CameraConfigData]
    pix_fmt: str = "bgr24"
