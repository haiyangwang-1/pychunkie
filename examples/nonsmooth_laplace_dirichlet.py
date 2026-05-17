"""Run both square-corner Laplace Dirichlet examples."""

from _nonsmooth_laplace_common import main_case


def main() -> None:
    main_case("interior", "dirichlet")
    main_case("exterior", "dirichlet")


if __name__ == "__main__":
    main()
