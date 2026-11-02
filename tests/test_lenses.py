import unittest

import numpy as np

from clip_compass.embeddings.types import ClipModelSpec, EmbeddingBatch
from clip_compass.mapper.lenses import (
    compute_lens,
    eccentricity,
    local_density,
    minmax_scale,
    pca_first_component,
)


def embedding_batch(raw: np.ndarray) -> EmbeddingBatch:
    raw = np.asarray(raw, dtype=np.float32)
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return EmbeddingBatch(
        item_ids=tuple(str(index) for index in range(raw.shape[0])),
        raw=raw,
        normalized=normalized,
        cache_hits=np.zeros(raw.shape[0], dtype=np.bool_),
        model_spec=ClipModelSpec("test", "fake", "v1", "fake"),
        input_sha256=tuple(str(index) * 64 for index in range(raw.shape[0])),
    )


class LensTests(unittest.TestCase):
    def test_l2_lens_uses_raw_features(self) -> None:
        batch = embedding_batch(np.asarray([[3.0, 4.0], [0.0, 2.0]]))
        result = compute_lens("l2_raw", batch, scale=False)
        self.assertEqual(result.source, "raw")
        np.testing.assert_allclose(result.values, [5.0, 2.0])

    def test_pca_sign_is_deterministic_and_centered(self) -> None:
        values = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        first = pca_first_component(values)
        second = pca_first_component(values)
        np.testing.assert_allclose(first, second)
        self.assertAlmostEqual(float(first.mean()), 0.0)
        self.assertGreater(first[-1], first[0])

    def test_eccentricity_and_density(self) -> None:
        points = np.asarray([[0.0], [1.0], [3.0]])
        np.testing.assert_allclose(eccentricity(points), [4.0 / 3.0, 1.0, 5.0 / 3.0])
        np.testing.assert_allclose(local_density(points, neighbors=1), [1.0, 1.0, 2.0])

    def test_constant_minmax_scale_is_zero(self) -> None:
        np.testing.assert_array_equal(minmax_scale(np.asarray([5.0, 5.0])), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
