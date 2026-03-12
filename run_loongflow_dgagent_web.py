from __future__ import annotations

from importlib import import_module


def main() -> None:
    module = import_module("loongflow_DGagent.webapp")
    module.main()


if __name__ == "__main__":
    main()
