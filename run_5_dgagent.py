from __future__ import annotations

import importlib


def main() -> None:
    module = importlib.import_module("5_DGagent.main")
    module.main()


if __name__ == "__main__":
    main()
