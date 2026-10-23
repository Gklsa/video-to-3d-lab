"""Fetch selected PhysTwin ZIP members without downloading full archives.

Only uses the Python standard library. ZIP member CRCs and extracted SHA-256
digests are checked and recorded; pickle and model files are never loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import struct
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
import zlib

REPO = "Jianghanxiao/PhysTwin"
DEFAULT_REVISION = "05665bcd0edad8f069c7def9cbb58d2eb75d68d0"
SAMPLE = "double_stretch_sloth"
REQUESTS = {
    "data.zip": ["calibrate.pkl", "metadata.json", "final_data.pkl", "data/bg.png"],
    "experiments.zip": ["train/best_199.pth"],
    "experiments_optimization.zip": ["optimal_params.pkl"],
    "gaussian_output.zip": ["iteration_10000/point_cloud.ply"],
}


def request_bytes(url, headers=None, maximum=20 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "PhysTwin-selective-sample/1.0", **(headers or {})})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read(maximum + 1)
                if len(data) > maximum:
                    raise RuntimeError("Server response exceeded the permitted byte count")
                return r.status, dict(r.headers), data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + 2 * attempt)


class RemoteArchive(io.RawIOBase):
    def __init__(self, name, revision, info):
        self.name = name
        self.revision = revision
        self.info = info
        self.size = info["size"]
        self.pos = 0
        self.downloaded = 0
        self.etags = set()
        self.cache = []
        self.url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}"

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if not 0 <= self.pos <= self.size:
            raise ValueError("Invalid remote ZIP offset")
        return self.pos

    def range(self, start, count):
        if count == 0:
            return b""
        if count > 16 * 1024 * 1024 or start < 0 or start + count > self.size:
            raise RuntimeError("Unsafe remote range size")
        for cached_start, cached_data in self.cache:
            if cached_start <= start and start + count <= cached_start + len(cached_data):
                return cached_data[start - cached_start:start - cached_start + count]
        end = start + count - 1
        # A fresh query avoids a cached response for another byte range.
        url = self.url + "?download=true&range_request=" + uuid.uuid4().hex
        status, headers, data = request_bytes(url, {"Range": f"bytes={start}-{end}"}, maximum=count)
        content_range = next((v for k, v in headers.items() if k.lower() == "content-range"), "")
        if status != 206 or content_range != f"bytes {start}-{end}/{self.size}" or len(data) != count:
            raise RuntimeError(f"Range request not honored: HTTP {status}, {content_range}, {len(data)} bytes")
        self.downloaded += count
        self.etags.update(v for k, v in headers.items() if k.lower() == "etag")
        if count <= 4 * 1024 * 1024:
            self.cache.append((start, data))
        return data

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.pos
        count = min(count, self.size - self.pos)
        data = self.range(self.pos, count)
        self.pos += count
        return data


def extract_member(remote, entry, destination):
    if destination.exists():
        existing = destination.read_bytes()
        if len(existing) == entry.file_size and zlib.crc32(existing) & 0xffffffff == entry.CRC:
            return {"member": entry.filename, "compressed_bytes": entry.compress_size,
                    "bytes": len(existing), "zip_crc32": f"{entry.CRC:08x}",
                    "crc_verified": True, "sha256": hashlib.sha256(existing).hexdigest(),
                    "local_path": str(destination)}
    header = remote.range(entry.header_offset, 30)
    sig, _, flags, compression, _, _, _, _, _, name_len, extra_len = struct.unpack("<4s5H3I2H", header)
    if sig != b"PK\x03\x04" or flags & 1:
        raise RuntimeError("Invalid or encrypted local ZIP member")
    if compression not in (0, 8):
        raise RuntimeError(f"Unsupported ZIP compression {compression}")
    start = entry.header_offset + 30 + name_len + extra_len
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".partial")
    decompressor = zlib.decompressobj(-15) if compression == 8 else None
    crc = 0
    digest = hashlib.sha256()
    total = 0
    with temp.open("wb") as f:
        for offset in range(0, entry.compress_size, 8 * 1024 * 1024):
            block = remote.range(start + offset, min(8 * 1024 * 1024, entry.compress_size - offset))
            output = decompressor.decompress(block) if decompressor else block
            f.write(output)
            crc = zlib.crc32(output, crc)
            digest.update(output)
            total += len(output)
        if decompressor:
            output = decompressor.flush()
            f.write(output)
            crc = zlib.crc32(output, crc)
            digest.update(output)
            total += len(output)
            if not decompressor.eof:
                raise RuntimeError("Truncated deflate stream")
    if crc & 0xffffffff != entry.CRC or total != entry.file_size:
        raise RuntimeError(f"ZIP CRC or uncompressed size mismatch for {entry.filename}")
    os.replace(temp, destination)
    return {
        "member": entry.filename, "compressed_bytes": entry.compress_size,
        "bytes": total, "zip_crc32": f"{entry.CRC:08x}", "crc_verified": True,
        "sha256": digest.hexdigest(), "local_path": str(destination),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory receiving data/, experiments/, and other sample folders")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="PhysTwin dataset revision")
    parser.add_argument("--inspect", action="store_true", help="List candidate member names without downloading them")
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    return args


def main():
    args = parse_args()
    output = args.output
    revision = args.revision
    output.mkdir(parents=True, exist_ok=True)
    encoded_revision = urllib.parse.quote(revision, safe="")
    _, _, raw = request_bytes(f"https://huggingface.co/api/datasets/{REPO}/tree/{encoded_revision}?recursive=false&expand=false")
    tree = {item["path"]: item for item in json.loads(raw)}
    manifest_path = output / "download_manifest.json"
    manifest = {"dataset": REPO, "revision": revision, "sample": SAMPLE, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "archives": {}, "files": []}
    for archive_name, suffixes in REQUESTS.items():
        remote = RemoteArchive(archive_name, revision, tree[archive_name])
        with zipfile.ZipFile(remote) as zf:
            entries = [entry for entry in zf.infolist() if SAMPLE in entry.filename and not entry.is_dir()]
            print(f"{archive_name}: {len(entries)} sample members", flush=True)
            selected = []
            for suffix in suffixes:
                pool = zf.infolist() if suffix == "data/bg.png" else entries
                matches = [entry for entry in pool if entry.filename == suffix or entry.filename.endswith("/" + suffix)]
                if len(matches) != 1:
                    print("Candidates:\n" + "\n".join(entry.filename for entry in entries[:20]), flush=True)
                    raise RuntimeError(f"Expected one member ending in {suffix}; found {len(matches)}")
                selected.extend(matches)
            for entry in selected:
                print(f"  {entry.filename} ({entry.compress_size:,} compressed; {entry.file_size:,} extracted)", flush=True)
                if args.inspect:
                    continue
                relative = PurePosixPath(entry.filename)
                if ".." in relative.parts or relative.is_absolute():
                    raise RuntimeError("Unsafe member path")
                destination = output.joinpath(*relative.parts)
                record = extract_member(remote, entry, destination)
                record["local_path"] = relative.as_posix()
                record["archive"] = archive_name
                manifest["files"].append(record)
                print(f"    CRC verified; SHA256 {record['sha256']}", flush=True)
        manifest["archives"][archive_name] = {"url": remote.url, "archive_bytes": remote.size, "downloaded_bytes": remote.downloaded, "lfs_sha256": tree[archive_name].get("lfs", {}).get("oid"), "etags_observed": sorted(remote.etags)}
        if not args.inspect:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.inspect:
        print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
