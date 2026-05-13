"""Interior Laplace Dirichlet solve on a square with true corners."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chunkie import chunkerkerneval, chunkermat, kernel
from nonsmooth_laplace_rcip_common import square_graph


depth = 2
nsub = 20
grid_size = 80

cg = square_graph(depth=depth, k=12)
chnkr = cg.merged()
boundary = chnkr.r.reshape(2, chnkr.npt, order="F")
system_kernel = -2.0 * kernel("lap", "d")
mat = chunkermat(cg, system_kernel, {"nsub": nsub, "rcip_savedepth": nsub})
system = np.eye(chnkr.npt, dtype=mat.dtype) + mat
rhs = boundary[0]
sigma = np.linalg.solve(system, rhs)

targets = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])
values = chunkerkerneval(cg, system_kernel, sigma, targets, {"forceadap": True}).reshape(-1)

xs = np.linspace(-1.15, 1.15, grid_size)
ys = np.linspace(-1.15, 1.15, grid_size)
xx, yy = np.meshgrid(xs, ys)
domain = (np.abs(xx) <= 0.995) & (np.abs(yy) <= 0.995)
plot_targets = np.vstack((xx[domain], yy[domain]))
plot_truth = np.full(xx.shape, np.nan)
plot_truth[domain] = plot_targets[0]
plot_values = chunkerkerneval(cg, system_kernel, sigma, plot_targets, {"forceadap": True}).reshape(-1)

solution = np.full(xx.shape, np.nan)
solution[domain] = plot_values
log_error = np.log10(np.maximum(np.abs(solution - plot_truth), 1e-16))
outline = np.array([[-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0, -1.0]])

solution_cmap = plt.get_cmap("RdBu_r").copy()
solution_cmap.set_bad("#eeeeee")
fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=160)
mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(solution), shading="auto", cmap=solution_cmap, vmin=-1, vmax=1)
ax.plot(outline[0], outline[1], color="black", linewidth=0.7)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Interior Dirichlet: layer potential")
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
ax.set_title("Interior Dirichlet: log10 abs error")
fig.colorbar(mesh, ax=ax, ticks=[-14, -11, -8, -5, -3], label="log10 |error|")
fig.tight_layout()
fig.savefig(__file__.replace(".py", "_error_log10.png"))
plt.close(fig)

print(f"RCIP square: depth {depth}, nsub {nsub}, {chnkr.nch} chunks, {chnkr.npt} nodes")
print(f"interior Dirichlet boundary residual: {np.max(np.abs(system @ sigma - rhs)):.3e}")
print(f"interior Dirichlet target max error: {np.max(np.abs(values - targets[0])):.3e}")
print(f"solution PNG: {__file__.replace('.py', '_solution.png')}")
print(f"error PNG: {__file__.replace('.py', '_error_log10.png')}")
