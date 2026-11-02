import tempfile
import unittest
from pathlib import Path

import numpy as np

from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.service import embed_images
from clip_compass.embeddings.types import ClipModelSpec, EmbeddingBatch


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0
        self.encoded_items = 0
        self._spec = ClipModelSpec("test", "fake", "v1", "bytes-v1")

    @property
    def spec(self) -> ClipModelSpec:
        return self._spec

    def encode_images(self, image_paths):
        self.calls += 1
        self.encoded_items += len(image_paths)
        return np.asarray(
            [[len(path.read_bytes()), path.read_bytes()[0], 2.0] for path in image_paths],
            dtype=np.float32,
        )


class EmbeddingTests(unittest.TestCase):
    def test_cache_hits_and_content_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.jpg"
            second = root / "second.jpg"
            first.write_bytes(b"abcd")
            second.write_bytes(b"xy")
            encoder = FakeEncoder()
            cache = EmbeddingCache(root / "cache")

            initial = embed_images(
                [first, second], encoder=encoder, cache=cache, item_ids=["a", "b"], batch_size=8
            )
            repeated = embed_images(
                [first, second], encoder=encoder, cache=cache, item_ids=["a", "b"], batch_size=8
            )
            second.write_bytes(b"xyz")
            changed = embed_images(
                [first, second], encoder=encoder, cache=cache, item_ids=["a", "b"], batch_size=8
            )

            self.assertEqual(encoder.calls, 2)
            self.assertEqual(encoder.encoded_items, 3)
            np.testing.assert_array_equal(initial.cache_hits, [False, False])
            np.testing.assert_array_equal(repeated.cache_hits, [True, True])
            np.testing.assert_array_equal(changed.cache_hits, [True, False])
            np.testing.assert_allclose(np.linalg.norm(initial.normalized, axis=1), 1.0)

    def test_embedding_artifact_round_trip(self) -> None:
        spec = ClipModelSpec("test", "fake", "v1", "bytes-v1")
        raw = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        batch = EmbeddingBatch(
            item_ids=("a", "b"),
            raw=raw,
            normalized=raw / np.linalg.norm(raw, axis=1, keepdims=True),
            cache_hits=np.asarray([False, True]),
            model_spec=spec,
            input_sha256=("1" * 64, "2" * 64),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "embeddings.npz"
            batch.save(path)
            loaded = EmbeddingBatch.load(path)

        self.assertEqual(loaded.item_ids, batch.item_ids)
        self.assertEqual(loaded.model_spec, batch.model_spec)
        np.testing.assert_array_equal(loaded.raw, batch.raw)
        np.testing.assert_array_equal(loaded.normalized, batch.normalized)


if __name__ == "__main__":
    unittest.main()
