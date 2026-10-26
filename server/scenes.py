"""Saving and listing scenes. Files go to the store (S3 in AWS); the scene index goes to DynamoDB when configured."""
import json
import os
import re
import time
import uuid

import boto3

FILES = {"scene.glb": "model/gltf-binary", "meta.json": "application/json", "input.jpg": "image/jpeg",
         "depth.jpg": "image/jpeg", "layers.jpg": "image/jpeg", "filled.jpg": "image/jpeg",
         "narration.mp3": "audio/mpeg", "page.jpg": "image/jpeg", "panorama.jpg": "image/jpeg"}
BEAT_FILE = re.compile(r"^beat_\d{2}\.mp3$")


def content_type(name: str) -> str | None:
    if name in FILES:
        return FILES[name]
    return "audio/mpeg" if BEAT_FILE.match(name) else None


class SceneIndex:
    def __init__(self, store):
        self.store = store
        table = os.environ.get("SCENES_TABLE")
        self.table = boto3.resource("dynamodb").Table(table) if table else None

    def save(self, meta: dict, files: dict[str, bytes], title: str, kind: str) -> str:
        scene_id = uuid.uuid4().hex[:12]
        meta = {"id": scene_id, "title": title.strip()[:80] or "Untitled scene", "kind": kind,
                "created": int(time.time()), **meta}
        prefix = f"scenes/{scene_id}/"
        for name, data in files.items():
            self.store.put(prefix + name, data, content_type(name))
        self.put_meta(meta)
        return scene_id

    def put_meta(self, meta: dict) -> None:
        self.store.put(f"scenes/{meta['id']}/meta.json", json.dumps(meta).encode(), FILES["meta.json"])
        if self.table is not None:
            self.table.put_item(Item={"id": meta["id"], "title": meta["title"], "type": meta["type"],
                                      "kind": meta["kind"], "created": meta["created"]})

    def get(self, scene_id: str) -> dict | None:
        raw = self.store.get(f"scenes/{scene_id}/meta.json")
        return json.loads(raw) if raw else None

    def list(self, limit: int) -> list[dict]:
        if self.table is not None:
            items = self.table.scan(ProjectionExpression="id, title, #t, kind, created",
                                    ExpressionAttributeNames={"#t": "type"}).get("Items", [])
            items = [i for i in items if i["kind"] != "story_page"]  # pages are reached through their story
            items.sort(key=lambda i: int(i.get("created", 0)), reverse=True)
            return [{"id": i["id"], "title": i["title"], "type": i["type"], "kind": i["kind"]} for i in items[:limit]]
        out = []
        for scene_id in self.store.list_dirs("scenes/"):
            m = self.get(scene_id)
            if m and m["kind"] != "story_page":
                out.append({"id": m["id"], "title": m["title"], "type": m["type"], "kind": m["kind"]})
            if len(out) >= limit:
                break
        return out
