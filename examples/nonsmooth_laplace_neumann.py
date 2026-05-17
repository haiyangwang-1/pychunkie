"""Run both square-corner Laplace Neumann examples."""

from _nonsmooth_laplace_common import main_case


def main() -> None:
    main_case("interior", "neumann")
    main_case("exterior", "neumann")


if __name__ == "__main__":
    main()
