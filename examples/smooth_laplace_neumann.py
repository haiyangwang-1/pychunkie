"""Run both smooth-circle Laplace Neumann examples."""

from _smooth_laplace_common import main_case


def main() -> None:
    main_case("interior", "neumann")
    main_case("exterior", "neumann")


if __name__ == "__main__":
    main()
