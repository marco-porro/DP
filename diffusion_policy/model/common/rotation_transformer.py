from typing import Union
import torch
import numpy as np
import functools
from scipy.spatial.transform import Rotation


class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
                 from_rep='axis_angle',
                 to_rep='rotation_6d',
                 from_convention=None,
                 to_convention=None):
        """
        Converts between rotation representations using matrix as intermediate.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None, "Euler 'from' needs convention"
        if to_rep == 'euler_angles':
            assert to_convention is not None, "Euler 'to' needs convention"

        self.from_rep = from_rep
        self.to_rep = to_rep
        self.from_convention = from_convention
        self.to_convention = to_convention

        forward_funcs = []
        inverse_funcs = []

        if from_rep != 'matrix':
            forward_funcs.append(
                functools.partial(self._to_matrix, rep=from_rep, convention=from_convention)
            )
            inverse_funcs.append(
                functools.partial(self._from_matrix, rep=from_rep, convention=from_convention)
            )

        if to_rep != 'matrix':
            forward_funcs.append(
                functools.partial(self._from_matrix, rep=to_rep, convention=to_convention)
            )
            inverse_funcs.append(
                functools.partial(self._to_matrix, rep=to_rep, convention=to_convention)
            )

        inverse_funcs = inverse_funcs[::-1]
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    # ---------- utility base conversioni ----------
    @staticmethod
    def _axis_angle_to_matrix(x: torch.Tensor) -> torch.Tensor:
        """axis-angle (radians, shape [...,3]) -> rotation matrix [...,3,3]"""
        theta = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
        axis = x / theta
        c = torch.cos(theta)[..., None]
        s = torch.sin(theta)[..., None]
        eye = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[:-1] + (3, 3))
        cross = torch.zeros_like(eye)
        cross[..., 0, 1], cross[..., 0, 2], cross[..., 1, 0], cross[..., 1, 2], cross[..., 2, 0], cross[..., 2, 1] = \
            -axis[..., 2], axis[..., 1], axis[..., 2], -axis[..., 0], -axis[..., 1], axis[..., 0]
        outer = axis[..., :, None] * axis[..., None, :]
        return c * eye + (1 - c) * outer + s * cross

    @staticmethod
    def _matrix_to_axis_angle(m: torch.Tensor) -> torch.Tensor:
        """rotation matrix [...,3,3] -> axis-angle [...,3]"""
        r = m
        cos_theta = ((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]) - 1) / 2
        cos_theta = cos_theta.clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        axis = torch.stack([
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1]
        ], dim=-1) / (2 * sin_theta[..., None].clamp_min(1e-8))
        return axis * theta[..., None]

    @staticmethod
    def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
        """quaternion [...,4] -> rotation matrix [...,3,3]"""
        q = q / torch.norm(q, dim=-1, keepdim=True)
        w, x, y, z = q.unbind(-1)
        B = q.shape[:-1]
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
            2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
            2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
        ], dim=-1).reshape(B + (3, 3))
        return R

    @staticmethod
    def _matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
        """rotation matrix [...,3,3] -> quaternion [...,4]"""
        r = m
        B = r.shape[:-2]
        q = torch.zeros(B + (4,), dtype=r.dtype, device=r.device)
        t = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
        cond = t > 0
        if cond.any():
            t_pos = t[cond]
            q_pos = q[cond]
            r_pos = r[cond]
            s = torch.sqrt(t_pos + 1.0) * 2
            q_pos[..., 0] = 0.25 * s
            q_pos[..., 1] = (r_pos[..., 2, 1] - r_pos[..., 1, 2]) / s
            q_pos[..., 2] = (r_pos[..., 0, 2] - r_pos[..., 2, 0]) / s
            q_pos[..., 3] = (r_pos[..., 1, 0] - r_pos[..., 0, 1]) / s
            q[cond] = q_pos
        return q / torch.norm(q, dim=-1, keepdim=True)

    @staticmethod
    def _rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
        """rotation 6D (Zhou et al. 2019) -> rotation matrix"""
        a1 = x[..., 0:3]
        a2 = x[..., 3:6]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    @staticmethod
    def _matrix_to_rotation_6d(m: torch.Tensor) -> torch.Tensor:
        return m[..., :3, :2].reshape(m.shape[:-2] + (6,))

    # ---------- conversion generiche ----------
    def _to_matrix(self, x: torch.Tensor, rep, convention=None):
        if rep == 'axis_angle':
            return self._axis_angle_to_matrix(x)
        elif rep == 'quaternion':
            return self._quaternion_to_matrix(x)
        elif rep == 'rotation_6d':
            return self._rotation_6d_to_matrix(x)
        elif rep == 'euler_angles':
            return torch.from_numpy(Rotation.from_euler(convention, x.cpu().numpy()).as_matrix()).to(x)
        elif rep == 'matrix':
            return x
        else:
            raise ValueError(f"Unsupported representation: {rep}")

    def _from_matrix(self, x: torch.Tensor, rep, convention=None):
        if rep == 'axis_angle':
            return self._matrix_to_axis_angle(x)
        elif rep == 'quaternion':
            return self._matrix_to_quaternion(x)
        elif rep == 'rotation_6d':
            return self._matrix_to_rotation_6d(x)
        elif rep == 'euler_angles':
            return torch.from_numpy(Rotation.from_matrix(x.cpu().numpy()).as_euler(convention)).to(x)
        elif rep == 'matrix':
            return x
        else:
            raise ValueError(f"Unsupported representation: {rep}")

    # ---------- API pubblica ----------
    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        for func in funcs:
            x_ = func(x_)
        y = x_.detach().cpu().numpy() if isinstance(x, np.ndarray) else x_
        return y

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


# ------------------- TEST -------------------
def test():
    tf = RotationTransformer()
    rotvec = np.random.uniform(-2*np.pi, 2*np.pi, size=(1000, 3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-6

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1, atol=1e-5)
    print("✅ Test passed!")
