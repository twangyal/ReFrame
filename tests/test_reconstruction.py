import unittest
import torch
from reframe.reconstruction import require_cuda, nearby_camera


class ReconstructionTests(unittest.TestCase):
    def test_nearby_camera_is_translation_in_camera_right_direction(self):
        # Original camera at world origin, looking down +z. Shift camera right.
        T=nearby_camera(torch.eye(4),2.)
        torch.testing.assert_close(T,torch.tensor([[1.,0,0,-2],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        self.assertEqual(torch.linalg.det(T[:3,:3]),1)

    def test_cpu_runtime_fails_explicitly_before_importing_cuda_extensions(self):
        if torch.cuda.is_available(): self.skipTest('CUDA is actually available')
        with self.assertRaisesRegex(RuntimeError,'CUDA GPU'):
            require_cuda()

if __name__ == '__main__': unittest.main()
