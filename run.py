from pathlib import Path

from PyRocket import run_simulation


if __name__ == "__main__":
    run_simulation(config_path=Path(__file__).resolve().parent / "config.toml")
