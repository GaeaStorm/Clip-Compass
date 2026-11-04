import tempfile
import unittest
from pathlib import Path

import numpy as np

from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.service import embed_images, embed_texts
from clip_compass.embeddings.similarity import ClipSimilarityMatrix, image_text_similarity
from clip_compass.embeddings.types import ClipModelSpec, EmbeddingBatch


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0
        self.encoded_items = 0
        self._spec = ClipModelSpec("test", "fake", "v1", "bytes-v1")
        self._text_spec = ClipModelSpec("test", "fake", "v1", "text-v1")

    @property
    def spec(self) -> ClipModelSpec:
        return self._spec

    @property
    def text_spec(self) -> ClipModelSpec:
        return self._text_spec

    def encode_images(self, image_paths):
        self.calls += 1
        self.encoded_items += len(image_paths)
        return np.asarray(
            [[len(path.read_bytes()), path.read_bytes()[0], 2.0] for path in image_paths],
            dtype=np.float32,
        )

    def encode_texts(self, texts):
        self.calls += 1
        self.encoded_items += len(texts)
        return np.asarray(
            [[len(text), text.encode("utf-8")[0], 2.0] for text in texts],
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

    def test_text_cache_deduplicates_exact_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            encoder = FakeEncoder()
            cache = EmbeddingCache(root / "cache")
            initial = embed_texts(
                ["red car", "red car", "blue bus"],
                encoder=encoder,
                cache=cache,
                item_ids=["a", "b", "c"],
                batch_size=8,
            )
            repeated = embed_texts(
                ["red car", "red car", "blue bus"],
                encoder=encoder,
                cache=cache,
                item_ids=["a", "b", "c"],
                batch_size=8,
            )

        self.assertEqual(encoder.calls, 1)
        self.assertEqual(encoder.encoded_items, 2)
        self.assertEqual(initial.modality, "text")
        np.testing.assert_array_equal(initial.raw[0], initial.raw[1])
        np.testing.assert_array_equal(repeated.cache_hits, [True, True, True])

    def test_image_text_similarity_validates_model_and_round_trips(self) -> None:
        image_raw = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        text_raw = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        image_spec = ClipModelSpec("test", "shared", "v1", "image-v1")
        text_spec = ClipModelSpec("test", "shared", "v1", "text-v1")
        images = EmbeddingBatch(
            item_ids=("image-a", "image-b"),
            raw=image_raw,
            normalized=image_raw,
            cache_hits=np.zeros(2, dtype=np.bool_),
            model_spec=image_spec,
            input_sha256=("1" * 64, "2" * 64),
            modality="image",
        )
        texts = EmbeddingBatch(
            item_ids=("text-a", "text-b"),
            raw=text_raw,
            normalized=text_raw / np.linalg.norm(text_raw, axis=1, keepdims=True),
            cache_hits=np.zeros(2, dtype=np.bool_),
            model_spec=text_spec,
            input_sha256=("3" * 64, "4" * 64),
            modality="text",
            input_texts=("a straight direction", "a diagonal direction"),
        )
        result = image_text_similarity(images, texts)
        self.assertAlmostEqual(float(result.scores[0, 0]), 1.0)
        self.assertAlmostEqual(float(result.scores[1, 0]), 0.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "similarities.npz"
            result.save(path)
            loaded = ClipSimilarityMatrix.load(path)
        np.testing.assert_array_equal(loaded.scores, result.scores)
        self.assertEqual(loaded.model_identity, result.model_identity)


if __name__ == "__main__":
    unittest.main()
