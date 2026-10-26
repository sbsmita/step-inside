"""Scene file storage: local disk in development, S3 when SCENE_BUCKET is set."""
import os
import threading
from pathlib import Path

import boto3


class LocalStore:
    def __init__(self, root: Path):
        self.root = root

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic like an S3 put: job status is polled while a worker rewrites it.
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get(self, key: str) -> bytes | None:
        path = self.root / key
        return path.read_bytes() if path.is_file() else None

    def list_dirs(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.is_dir():
            return []
        return [p.name for p in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if p.is_dir()]


class S3Store:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def list_dirs(self, prefix: str) -> list[str]:
        pages = self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/")
        return [cp["Prefix"][len(prefix):].rstrip("/") for page in pages for cp in page.get("CommonPrefixes", [])]


def make_store(root: Path):
    bucket = os.environ.get("SCENE_BUCKET")
    return S3Store(bucket) if bucket else LocalStore(root)
