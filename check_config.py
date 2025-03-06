from collections.abc import Iterable

from configfiledata import ConfigFileData, CameraConfigData

__all__ = ["check_config"]


def check_config(config: ConfigFileData):
    check_unique_links(config.links)
    check_name_in_links_existing(config.links, config.cams.keys())
    warn_unused_cams(config.links, config.cams.keys())


def check_unique_links(links: list[tuple[str, str]]):
    seen_links: set[tuple[str, str]] = set()
    known_duplicates: set[tuple[str, str]] = set()
    for link in links:
        if link in known_duplicates:
            continue
        if link in seen_links:
            known_duplicates.add(link)
            print(f"duplicate link {link} found!")
        seen_links.add(link)
    if len(known_duplicates) > 0:
        raise ValueError("links are not unique")


def check_name_in_links_existing(links: list[tuple[str, str]], names: Iterable[str]):
    has_unknown: bool = False
    for link in links:
        for name in link:
            if name not in names:
                has_unknown = True
                print(f"unknown name '{name}' in link {link}")
    if has_unknown:
        raise KeyError("unknown names in links")


def warn_unused_cams(links: list[tuple[str, str]], names: Iterable[str]):
    used_cams: set[str] = set()
    for link in links:
        for name in link:
            used_cams.add(name)
    for cam in names:
        if cam not in used_cams:
            print(f"cam {cam} is unused")
