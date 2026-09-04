#!/usr/bin/env python3
"""Extract selected ZIP members with HTTP Range requests.

The OPV2V ``additional-001.zip`` archive is large, while the GT-free MTR
adapter only needs the BEV lane rasters.  The archive's central directory is
cached locally; this utility resolves a short-lived Box download URL and
fetches only contiguous byte spans containing the requested members.
"""

from __future__ import annotations

import argparse
import binascii
import re
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import requests


CD_STRUCT = struct.Struct("<4s6H3I5H2I")
LOCAL_STRUCT = struct.Struct("<4s5H3I2H")


def parse_central_directory(path: Path) -> List[Dict[str, object]]:
    blob = path.read_bytes()
    records: List[Dict[str, object]] = []
    pos = 0
    while pos < len(blob):
        if pos + CD_STRUCT.size > len(blob):
            raise RuntimeError(f"truncated central-directory record at {pos}")
        values = CD_STRUCT.unpack_from(blob, pos)
        (
            signature,
            _version_made,
            _version_needed,
            flags,
            method,
            _mtime,
            _mdate,
            crc32,
            compressed_size,
            uncompressed_size,
            name_len,
            extra_len,
            comment_len,
            _disk,
            _internal_attr,
            _external_attr,
            local_offset,
        ) = values
        if signature != b"PK\x01\x02":
            raise RuntimeError(f"invalid central-directory signature at {pos}")
        name_start = pos + CD_STRUCT.size
        name_bytes = blob[name_start : name_start + name_len]
        name = name_bytes.decode("utf-8")
        records.append(
            {
                "name": name,
                "flags": flags,
                "method": method,
                "crc32": crc32,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "name_len": name_len,
                "extra_len": extra_len,
                "local_offset": local_offset,
            }
        )
        pos = name_start + name_len + extra_len + comment_len
    if pos != len(blob):
        raise RuntimeError(f"central-directory parse stopped at {pos}/{len(blob)}")
    return records


def resolve_box_url(session: requests.Session, share_url: str, shared_name: str, file_id: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    page = session.get(share_url, headers=headers, timeout=60)
    page.raise_for_status()
    match = re.search(r'"sharedName"\s*:\s*"([A-Za-z0-9]+)"', page.text)
    if match and match.group(1) != shared_name:
        shared_name = match.group(1)
    endpoint = (
        "https://ucla.app.box.com/index.php?rm=box_download_shared_file"
        f"&shared_name={shared_name}&file_id=f_{file_id}"
    )
    response = session.get(
        endpoint,
        headers={**headers, "Referer": share_url},
        allow_redirects=False,
        timeout=60,
    )
    if response.status_code not in (301, 302, 303, 307, 308) or not response.headers.get("location"):
        raise RuntimeError(f"Box did not return a download redirect: HTTP {response.status_code}")
    return response.headers["location"]


def fetch_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
        "Range": f"bytes={start}-{end}",
    }
    response = session.get(url, headers=headers, timeout=(30, 600))
    if response.status_code != 206:
        raise RuntimeError(f"range {start}-{end} returned HTTP {response.status_code}")
    content_range = response.headers.get("Content-Range", "")
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"range {start}-{end} returned {len(response.content)} bytes, expected {expected} ({content_range})"
        )
    return response.content


def extract_entry(blob: bytes, span_start: int, entry: Mapping[str, object]) -> bytes:
    local_offset = int(entry["local_offset"])
    rel = local_offset - span_start
    if rel < 0 or rel + LOCAL_STRUCT.size > len(blob):
        raise RuntimeError(f"local header outside fetched span: {entry['name']}")
    values = LOCAL_STRUCT.unpack_from(blob, rel)
    signature, _version, flags, method, _mtime, _mdate, _crc, _cs, _us, name_len, extra_len = values
    if signature != b"PK\x03\x04":
        raise RuntimeError(f"invalid local header for {entry['name']}")
    data_start = rel + LOCAL_STRUCT.size + name_len + extra_len
    data_end = data_start + int(entry["compressed_size"])
    if data_end > len(blob):
        raise RuntimeError(f"compressed member truncated in span: {entry['name']}")
    compressed = blob[data_start:data_end]
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported ZIP compression method {method} for {entry['name']}")
    if len(payload) != int(entry["uncompressed_size"]):
        raise RuntimeError(f"size mismatch for {entry['name']}: {len(payload)} != {entry['uncompressed_size']}")
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    if crc != int(entry["crc32"]):
        raise RuntimeError(f"CRC mismatch for {entry['name']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--central-directory", type=Path, default=root / "data/tmp/additional.cd")
    parser.add_argument("--output-root", type=Path, default=root / "data/raw/OPV2V/additional/test")
    parser.add_argument("--share-url", default="https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/folder/279976559690")
    parser.add_argument("--shared-name", default="vxetzti0z0dv97yeh1jgxdprxxqucs2a")
    parser.add_argument("--file-id", default="1621920078208")
    parser.add_argument("--scene", action="append", required=True)
    args = parser.parse_args()

    records = parse_central_directory(args.central_directory)
    selected = [
        record
        for record in records
        if str(record["name"]).startswith("additional/test/")
        and str(record["name"]).endswith("_bev_lane.png")
        and any(str(record["name"]).startswith(f"additional/test/{scene}/") for scene in args.scene)
    ]
    if not selected:
        raise SystemExit("no requested lane members in central directory")

    by_scene: Dict[str, List[Mapping[str, object]]] = {scene: [] for scene in args.scene}
    for record in selected:
        name = str(record["name"])
        for scene in args.scene:
            if name.startswith(f"additional/test/{scene}/"):
                by_scene[scene].append(record)
                break

    session = requests.Session()
    url = resolve_box_url(session, args.share_url, args.shared_name, args.file_id)
    print(f"selected_members={len(selected)}", flush=True)
    print(f"archive_url_resolved=yes", flush=True)

    extracted = 0
    for scene in args.scene:
        entries = sorted(by_scene[scene], key=lambda item: int(item["local_offset"]))
        if not entries:
            print(f"scene={scene} members=0", flush=True)
            continue
        # A small safety margin covers a local header's extra field, which is
        # not required to equal the central-directory extra field length.
        start = min(int(item["local_offset"]) for item in entries)
        end = max(
            int(item["local_offset"])
            + 30
            + int(item["name_len"])
            + int(item["extra_len"])
            + int(item["compressed_size"])
            + 65535
            for item in entries
        )
        blob = fetch_range(session, url, start, end - 1)
        scene_extracted = 0
        for entry in entries:
            payload = extract_entry(blob, start, entry)
            relative = Path(str(entry["name"])).relative_to("additional/test")
            destination = args.output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
                handle.write(payload)
                temporary = Path(handle.name)
            temporary.replace(destination)
            scene_extracted += 1
            extracted += 1
        print(
            f"scene={scene} members={len(entries)} extracted={scene_extracted} span_mb={(end - start) / 1e6:.2f}",
            flush=True,
        )
    print(f"extracted_total={extracted}", flush=True)


if __name__ == "__main__":
    main()
