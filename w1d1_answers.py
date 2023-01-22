
import os
import torch as t
from einops import rearrange, repeat, reduce
import matplotlib.pyplot as plt
from ipywidgets import interact
import w1d1_test
#import lovely_tensors as lt
#lt.monkey_patch()

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

def make_rays_2d(num_pixels_y: int, num_pixels_z, y_limit: float, z_limit: float) -> t.Tensor:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.
    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    rays = t.zeros((num_pixels_y, num_pixels_z, 2, 3))
    rays[:, :, 1, 0] = 1.0
    ys = t.linspace(-y_limit, +y_limit, num_pixels_y)
    zs = t.linspace(-z_limit, +z_limit, num_pixels_z)
    ys = repeat(ys, "y -> y z", z=num_pixels_z)
    zs = repeat(zs, "z -> y z", y=num_pixels_y)
    rays[:, :, 1, 1] = ys
    rays[:, :, 1, 2] = zs
    rays = rearrange(rays, 'ny nz p d -> (ny nz) p d')
    return rays 

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    """
    ray: shape (n_points=2, n_dim=3) # O, D points
    segment: shape (n_points=2, n_dim=3) # L_1, L_2 points

    Returns: shape (n_dim=2) # intersection point
    """
    D = ray[1][:2]
    O = ray[0][:2]
    L_1 = segment[0][:2]
    L_2 = segment[1][:2]
    
    A = t.stack([D[:2], L_1[:2] - L_2[:2]], dim=0).t()
    B = L_1 - O
    intersection = t.zeros_like(O)
    try:
        t.linalg.solve(A, B, out=intersection)
    except:
        return False
    if intersection[0] < 0:
        return False
    if intersection[1] < 0 or intersection[1] > 1:
        return False
    return True

def intersect_rays_1d(rays: t.Tensor, segments: t.Tensor) -> t.Tensor:
    """
    rays: shape (NR, 2, 3) - NR is the number of rays
    segments: shape (NS, 2, 3) - NS is the number of segments

    Return: shape (NR, )
    """
    intersects = t.zeros((rays.shape[0],), dtype=t.bool)
    #TODO: vectorize this
    for i, ray in enumerate(rays):
        for _, segment in enumerate(segments):
            if intersect_ray_1d(ray, segment):
                intersects[i] = True
                break
            intersects[i] = False
    return intersects
                
def render_lines_with_pyplot(lines: t.Tensor):
    """Plot any number of line segments in 3D.

    lines: shape (num_lines, num_points=2, num_dims=3).
    """
    (fig, ax) = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
    for line in lines:
        ax.plot(line[:, 1].numpy(), line[:, 0].numpy(), line[:, 2].numpy())
    ax.set(xlabel="Y", ylabel="X", zlabel="Z")
    return fig

def triangle_line_intersects(A: t.Tensor, B: t.Tensor, C: t.Tensor, O: t.Tensor, D: t.Tensor) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the line and the triangle intersect.
    """
    Matrix = t.zeros((3,3))
    Matrix[:, 0] = -D
    Matrix[:, 1] = -A + B
    Matrix[:, 2] = -A + C

    Vector = O - A
    try:
        t.linalg.solve(Matrix, Vector, out=Vector)
    except:
        return False
    if Vector[0] < 0:
        return False
    if Vector[1] < 0 or Vector[2] < 0 or Vector[1] + Vector[2] > 1:
        return False
    return True

def raytrace_triangle(triangle: t.Tensor, rays: t.Tensor) -> t.Tensor:
    """For each ray, return True if the triangle intersects that ray.

    triangle: shape (n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)

    return: shape (n_pixels, )
    """
    n_pixels = rays.shape[0]
    Matrix = t.zeros((n_pixels, 3, 3))
    Ds = rays[:, 1, :]
    Matrix[:, :, 0] = -Ds
    Matrix[:, :, 1] = repeat(triangle[1, :] - triangle[0, :], "d -> p d", p=n_pixels)
    Matrix[:, :, 2] = repeat(triangle[2, :] - triangle[0, :], "d -> p d", p=n_pixels)

    Vector = - triangle[0, :]
    v = t.linalg.solve(Matrix, Vector)
    condition_1 = v[:, 0] > 0
    condition_2 = (v[:, 1] > 0) & (v[:, 2] > 0) & (v[:, 1] + v[:, 2] < 1)
    intersections = condition_1 & condition_2
    print(intersections)
    # taking the & of two boolean tensors returns a boolean tensor
    return intersections

def raytrace_mesh(triangles: t.Tensor, rays: t.Tensor) -> t.Tensor:
    """For each ray, return the distance to the closest intersecting triangle, or infinity.

    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)

    return: shape (n_pixels, )
    """
    n_pixels = rays.shape[0]
    n_triangles = triangles.shape[0]
    Matrix = t.zeros(( n_pixels, n_triangles, 3, 3))
    Os = rays[:, 0, :]
    Os = repeat(Os, "p d -> p t d", t=n_triangles)
    Ds = rays[:, 1, :]
    Ds = repeat(Ds, "p d -> p t d", t=n_triangles)
    As = triangles[:, 0, :]
    As = repeat(As, "t d -> p t d", p=n_pixels)
    Bs = triangles[:, 1, :]
    Bs = repeat(Bs, "t d -> p t d", p=n_pixels)
    Cs = triangles[:, 2, :]
    Cs = repeat(Cs, "t d -> p t d", p=n_pixels)

    Matrix[:, :, :, 0] = -Ds
    Matrix[:, :, :, 1] = -As + Bs
    Matrix[:, :, :, 2] = -As + Cs
    Vector = Os - As

    assert Matrix.shape == ( n_pixels, n_triangles, 3, 3)
    assert Vector.shape == ( n_pixels , n_triangles, 3)

    Matrix = rearrange(Matrix, "p t d b -> (p t) d b")
    Vector = rearrange(Vector, "p t d -> (p t) d")

    assert Matrix.shape == (n_pixels * n_triangles, 3, 3)
    assert Vector.shape == (n_pixels * n_triangles, 3)

    dets = t.det(Matrix)
    where_singular = dets < 1e-9
    Matrix[where_singular] = t.eye(3)

    v = t.linalg.solve(Matrix, Vector)
    s = v[:, 0]
    u = v[:, 1]
    v = v[:, 2]

    condition_1 = s > 0
    condition_2 = (u > 0) & (v > 0) & (u + v < 1) & ~where_singular
    intersections = condition_1 & condition_2
    assert intersections.shape == (n_pixels * n_triangles, )
    s[~intersections] = float("inf")
    s = rearrange(s, "(p t) -> p t", p=n_pixels, t=n_triangles)
    return s.min(dim=1)[0]

num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

with open("w1d1_pikachu.pt", "rb") as f:
    triangles = t.load(f)

num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1    


if MAIN:
    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    # make the origins (-3,0,0)
    rays[:, 0, :] = t.tensor([-3.0, 0.0, 0.0])
    dists = raytrace_mesh(triangles, rays)
    dists_square = rearrange(dists, "(y z) -> y z", y=num_pixels_y, z=num_pixels_z)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(dists_square)
    axes[1].imshow(dists_square < 9999)
    plt.show()
    
    A = t.tensor([2, 0.0, -1.0])
    B = t.tensor([2, -1.0, 0.0])
    C = t.tensor([2, 1.0, 1.0])
    num_pixels_y = num_pixels_z = 1000
    y_limit = z_limit = 0.5
    tri = t.stack([A, B, C], dim=0)
    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    intersects = raytrace_triangle(tri, rays)
    img = rearrange(intersects, "(y z) -> y z", y=num_pixels_y, z=num_pixels_z)
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()


    rays1d = make_rays_1d(9, 10.0)
    segments = t.tensor([[[1.0, -12.0, 0.0], [1, -6.0, 0.0]], [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], [[2, 12.0, 0.0], [2, 21.0, 0.0]]])
    intersect_ray_1d(rays1d[0], segments[0])
    two_rays = make_rays_2d(5, 5, 10.0, 10.0)

    # combine the rays and segments into a single tensor
    segs_and_rays = t.concatenate((rays1d, segments), dim = 0)
    render_lines_with_pyplot(two_rays)
    #plt.show()


    
    







