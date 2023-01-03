import os
import torch as t
import einops
import matplotlib.pyplot as plt
from ipywidgets import interact
import w1d1_test
import lovely_tensors as lt
lt.monkey_patch()

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")


def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros((num_pixels, 2, 3))
    rays[:, 1, 0] = 1.0
    t.linspace(-y_limit, +y_limit, num_pixels, out=rays[:,1,1])
    return rays

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    """
    ray: shape (n_points=2, n_dim=3) # O, D points
    segment: shape (n_points=2, n_dim=3) # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    D = ray[1]
    O = ray[0]
    L_1 = segment[0]
    L_2 = segment[1]

    A = t.stack([D, L_1 - L_2], dim=0)
    print(A)

    intersection = t.zeros_like(O)
    t.linalg.solve()
    return None
    


def render_lines_with_pyplot(lines: t.Tensor):
    """Plot any number of line segments in 3D.

    lines: shape (num_lines, num_points=2, num_dims=3).
    """
    (fig, ax) = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
    for line in lines:
        ax.plot(line[:, 1].numpy(), line[:, 0].numpy(), line[:, 2].numpy())
    ax.set(xlabel="Y", ylabel="X", zlabel="Z")
    return fig


rays1d = make_rays_1d(3, 10.0)
if MAIN and (not IS_CI):
    rays1d = make_rays_1d(3, 10.0)
    segments = t.tensor([[[1.0, -12.0, 0.0], [1, -6.0, 0.0]], [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], [[2, 12.0, 0.0], [2, 21.0, 0.0]]])
    
    intersect_ray_1d(rays1d[0], segments[0])

    render_lines_with_pyplot(rays1d)
    render_lines_with_pyplot(segments)
    plt.show()
