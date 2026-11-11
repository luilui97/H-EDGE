"""Cached model-service calls shared by planning, verification and answer generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


MODEL = "llama3.1:70b"
BASE = "http://127.0.0.1:11438/api/"


class LocalLLM:
    def __init__(self, cache: Path, *, seed: int = 17, timeout: int = 600,
                 model: str = MODEL, base_url: str = BASE,
                 expected_digest: str | None = None):
        self.cache = cache
        self.seed = seed
        self.timeout = timeout
        self.calls: list[dict] = []
        self.model = model
        self.base_url = base_url.rstrip("/") + "/"
        req = urllib.request.Request(self.base_url + "tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            tags = json.load(response)
        models = {m["name"]: m["digest"] for m in tags["models"]}
        self.model_digest = models.get(model)
        if not self.model_digest:
            raise RuntimeError("requested language model is not available")
        if expected_digest is not None and self.model_digest != expected_digest:
            raise RuntimeError("local Reader model digest mismatch")

    def ask(self, kind: str, system: str, content: dict, *, tokens: int = 700,
            schema: dict | None = None) -> dict:
        payload = {"model": self.model, "stream": False, "think": False,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": json.dumps(content, ensure_ascii=False)}],
                   "format": schema or "json",
                   "options": {"temperature": 0, "seed": self.seed, "num_ctx": 8192,
                               "num_predict": tokens}}
        key = hashlib.sha256(json.dumps({"model_digest": self.model_digest, "payload": payload},
                                        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.cache / kind / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["payload"] != payload or record["model_digest"] != self.model_digest:
                raise ValueError("cached LLM identity mismatch")
            raw, fresh = record["response"], False
        else:
            body = json.dumps({**payload, "keep_alive": -1}, ensure_ascii=False).encode()
            req = urllib.request.Request(self.base_url + "chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = json.load(response)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps({"payload": payload, "model_digest": self.model_digest,
                                        "response": raw}, ensure_ascii=False), encoding="utf-8")
            temp.replace(path)
            fresh = True
        receipt = {"kind": kind, "key": key, "path": str(path), "fresh": fresh,
                   "prompt_tokens": raw.get("prompt_eval_count", 0),
                   "generated_tokens": raw.get("eval_count", 0),
                   "model_seconds": raw.get("total_duration", 0) / 1e9}
        self.calls.append(receipt)
        if raw.get("done") is not True or raw.get("done_reason") == "length":
            raise ValueError(f"{kind} generation incomplete")
        try:
            value = json.loads(raw["message"]["content"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{kind} invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{kind} JSON object required")
        return value
