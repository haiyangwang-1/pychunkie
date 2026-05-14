"""Exterior Laplace Neumann solve on a square with true corners."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nonsmooth_laplace_rcip_common import square_graph

from chunkie import PointInfo, chunkerkerneval, chunkermat, kernel

depth = 2
rcip_subdivisions = 20
grid_size = 80

cg = square_graph(depth=depth, k=12)
boundary = cg.merged()
point_info = PointInfo.from_any(boundary)
nodes = point_info.r
normals = point_info.n
lap_s = kernel("lap", "s")
system_kernel = -2.0 * kernel("lap", "sp")
mat = chunkermat(
    cg,
    system_kernel,
    rcip_subdivisions=rcip_subdivisions,
    rcip_save_depth=rcip_subdivisions,
)
system = np.eye(boundary.npt, dtype=mat.dtype) + mat

x = nodes[0]
y = nodes[1]
r2 = x**2 + y**2
r4 = r2**2
normal_data = -2.0 * (((y**2 - x**2) / r4) * normals[0] + (-2.0 * x * y / r4) * normals[1])
sigma = np.linalg.solve(system, normal_data)

targets = np.array([[1.3, 2.0, -1.4, 0.2], [0.2, 0.5, -1.2, 1.5]])
target_truth = targets[0] / np.sum(targets**2, axis=0)
values = chunkerkerneval(
    cg,
    lap_s,
    sigma,
    targets,
    force_adaptive=True,
    use_panel_quadrature=True,
).reshape(-1)

xs = np.linspace(-2.0, 2.0, grid_size)
ys = np.linspace(-2.0, 2.0, grid_size)
xx, yy = np.meshgrid(xs, ys)
domain = (np.abs(xx) >= 1.005) | (np.abs(yy) >= 1.005)
plot_targets = np.vstack((xx[domain], yy[domain]))
plot_truth = np.full(xx.shape, np.nan)
plot_truth[domain] = plot_targets[0] / np.sum(plot_targets**2, axis=0)
plot_values = chunkerkerneval(
    cg,
    lap_s,
    sigma,
    plot_targets,
    force_adaptive=True,
    use_panel_quadrature=True,
).reshape(-1)

solution = np.full(xx.shape, np.nan)
solution[domain] = plot_values
log_error = np.log10(np.maximum(np.abs(solution - plot_truth), 1e-16))
outline = np.array([[-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0, -1.0]])

solution_cmap = plt.get_cmap("RdBu_r").copy()
solution_cmap.set_bad("#eeeeee")
fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=160)
mesh = ax.pcolormesh(
    xs, ys, np.ma.masked_invalid(solution), shading="auto", cmap=solution_cmap, vmin=-1, vmax=1
)
ax.plot(outline[0], outline[1], color="black", linewidth=0.7)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Exterior Neumann: layer potential")
fig.colorbar(mesh, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1], label="u")
fig.tight_layout()
fig.savefig(__file__.replace(".py", "_solution.png"))
plt.close(fig)

error_cmap = plt.get_cmap("magma").copy()
error_cmap.set_bad("#eeeeee")
fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=160)
mesh = ax.pcolormesh(
    xs, ys, np.ma.masked_invalid(log_error), shading="auto", cmap=error_cmap, vmin=-14, vmax=-3
)
ax.plot(outline[0], outline[1], color="black", linewidth=0.7)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Exterior Neumann: log10 abs error")
fig.colorbar(mesh, ax=ax, ticks=[-14, -11, -8, -5, -3], label="log10 |error|")
fig.tight_layout()
fig.savefig(__file__.replace(".py", "_error_log10.png"))
plt.close(fig)

net_charge = float(np.dot(boundary.quadrature_weights.T.reshape(-1), sigma))
print(
    f"RCIP square: depth {depth}, subdivisions {rcip_subdivisions}, {boundary.nch} chunks, {boundary.npt} nodes"
)
print(f"exterior Neumann boundary residual: {np.max(np.abs(system @ sigma - normal_data)):.3e}")
print(f"exterior Neumann target max error: {np.max(np.abs(values - target_truth)):.3e}")
print(f"exterior Neumann density net charge: {net_charge:.3e}")
print(f"solution PNG: {__file__.replace('.py', '_solution.png')}")
print(f"error PNG: {__file__.replace('.py', '_error_log10.png')}")
