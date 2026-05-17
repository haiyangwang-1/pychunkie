"""Interior Laplace Neumann solve on a square with true corners."""

from _nonsmooth_laplace_common import main_case

if __name__ == "__main__":
    main_case("interior", "neumann")
