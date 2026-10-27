import unittest

import numpy as np
import torch

from reframe.association import extract_evidence, aggregate_evidence


class ContributionTests(unittest.TestCase):
    def test_extent_and_occlusion_use_visible_weight_not_centers(self):
        # Analytic two-pixel compositing, NOT a Gaussian renderer/model result.
        # Front splat alpha=.8 at pixel 0; back alpha=.5,.6. Back visible=.1,.6.
        weights = torch.tensor([[.8, 0.], [.1, .6]])
        def render(colors):
            return (colors.T @ weights).reshape(3, 1, 2)
        evidence = extract_evidence(render, 2, torch.tensor([[1., 0.]]))
        torch.testing.assert_close(evidence, torch.tensor([[.8, 0., .8], [.1, .6, .7]]))

    def test_all_ones_and_zeros_and_hidden_splat(self):
        weights = torch.tensor([[.7, .8], [0., 0.]])
        def render(colors):
            return (colors.T @ weights).reshape(3, 1, 2)
        for value in [0., 1.]:
            e = extract_evidence(render, 2, torch.full((1, 2), value))
            torch.testing.assert_close(e[1], torch.zeros(3))
            self.assertAlmostEqual(e[0, 0].item(), 1.5*value)
            self.assertAlmostEqual(e[0, 1].item(), 1.5*(1-value))

    def test_rejects_nonbinary_or_invalid_masks(self):
        for mask in [torch.tensor([[float('nan')]]), torch.tensor([[.5]]), torch.ones(2)]:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                extract_evidence(lambda c: c.T.reshape(3, 1, 1), 1, mask)


class AggregationTests(unittest.TestCase):
    def test_unseen_is_unknown_and_conflicting_is_ambiguous(self):
        e = np.array([[[9,1,10], [0,0,0], [10,0,10], [0,10,10]],
                      [[9,1,10], [0,0,0], [0,10,10], [0,10,10]]], dtype=float)
        r = aggregate_evidence(e)
        self.assertEqual(r['state'].tolist(), ['foreground','unknown','ambiguous','background'])
        np.testing.assert_allclose(r['score'][[0,2,3]], [.9,.5,0.])
        self.assertTrue(np.isnan(r['score'][1]))
        self.assertEqual(r['visible_frames'].tolist(), [2,0,2,2])

    def test_one_large_closeup_cannot_override_three_contradicting_views(self):
        e = np.array([[[10000,0,10000]], [[0,100,100]], [[0,100,100]], [[0,100,100]]])
        r = aggregate_evidence(e, max_frame_weight=10)
        self.assertAlmostEqual(r['score'][0], .25)
        self.assertEqual(r['state'][0], 'ambiguous')

    def test_negligible_support_does_not_become_confident_foreground(self):
        e = np.array([[[.00001,0,.00001]], [[.00001,0,.00001]]])
        self.assertEqual(aggregate_evidence(e)['state'][0], 'unknown')

    def test_bad_evidence_and_thresholds_rejected(self):
        for e in [np.zeros((0,2,3)), np.array([[[1,0,0]]]), np.array([[[np.nan,0,1]]]),
                  np.array([[[-1,2,1]]])]:
            with self.subTest(e=e), self.assertRaises(ValueError): aggregate_evidence(e)
        with self.assertRaises(ValueError):
            aggregate_evidence(np.ones((2,1,3)), foreground_threshold=.2, background_threshold=.8)


if __name__ == '__main__':
    unittest.main()
