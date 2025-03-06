from dataclasses import dataclass


@dataclass
class CameraConfigData:
    url: str
    calib_data: str


@dataclass
class ConfigFileData:
    cams: dict[str, CameraConfigData]
    links: list[tuple[str, str]]
    pix_fmt: str = "bgr24"
