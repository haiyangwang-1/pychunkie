"""Exterior Laplace Dirichlet solve on a square with true corners."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chunkie import chunkerkerneval, chunkerkernevalmat, chunkermat, chunkerpoly, kernel


depth = 40
grid_size = 80
verts = np.array([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]])

chnkr = chunkerpoly(
    verts,
    {"ifclosed": True, "dyadic": True, "depth": depth, "widths": 0.25},
    {"k": 12, "nchmax": 2000},
)
boundary = chnkr.r.reshape(2, chnkr.npt, order="F")
weights = chnkr.wts.reshape(-1, order="F")
lap_s = kernel("lap", "s")
s_mat = chunkermat(chnkr, lap_s)

system = np.block(
    [
        [s_mat, np.ones((chnkr.npt, 1))],
        [weights[None, :], np.zeros((1, 1))],
    ]
)
boundary_truth = boundary[0] / np.sum(boundary**2, axis=0)
rhs = np.concatenate((boundary_truth, [0.0]))
sol = np.linalg.solve(system, rhs)
sigma = sol[:-1]
const = float(sol[-1])

targets = np.array([[1.3, 2.0, -1.4, 0.2], [0.2, 0.5, -1.2, 1.5]])
target_truth = targets[0] / np.sum(targets**2, axis=0)
cormat = chunkerkernevalmat(chnkr, lap_s, targets, {"corrections": True, "fac": 1.0})
values = chunkerkerneval(
    chnkr, lap_s, sigma, targets, {"forcesmooth": True, "cormat": cormat}
).reshape(-1) + const

xs = np.linspace(-2.0, 2.0, grid_size)
ys = np.linspace(-2.0, 2.0, grid_size)
xx, yy = np.meshgrid(xs, ys)
domain = (np.abs(xx) >= 1.005) | (np.abs(yy) >= 1.005)
plot_targets = np.vstack((xx[domain], yy[domain]))
plot_truth = np.full(xx.shape, np.nan)
plot_truth[domain] = plot_targets[0] / np.sum(plot_targets**2, axis=0)
cormat = chunkerkernevalmat(chnkr, lap_s, plot_targets, {"corrections": True, "fac": 1.0})
plot_values = chunkerkerneval(
    chnkr, lap_s, sigma, plot_targets, {"forcesmooth": True, "cormat": cormat}
).reshape(-1) + const

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
ax.set_title("Exterior Dirichlet: layer potential")
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
ax.set_title("Exterior Dirichlet: log10 abs error")
fig.colorbar(mesh, ax=ax, ticks=[-14, -11, -8, -5, -3], label="log10 |error|")
fig.tight_layout()
fig.savefig(__file__.replace(".py", "_error_log10.png"))
plt.close(fig)

print(f"dyadic square: depth {depth}, {chnkr.nch} chunks, {chnkr.npt} nodes")
print(f"exterior Dirichlet boundary residual: {np.max(np.abs(s_mat @ sigma + const - boundary_truth)):.3e}")
print(f"exterior Dirichlet target max error: {np.max(np.abs(values - target_truth)):.3e}")
print(f"solution PNG: {__file__.replace('.py', '_solution.png')}")
print(f"error PNG: {__file__.replace('.py', '_error_log10.png')}")
