"""Run AeroClim's refresh and validation pipeline."""

from __future__ import annotations

import subprocess
import sys


COMMANDS = [
    ["tools/refresh_climate_indices.py"],
    ["tools/refresh_ocean_sst.py"],
    ["tools/repair_atmospheric_gaps.py"],
    ["tools/audit_data_health.py"],
]


def main() -> None:
    for arguments in COMMANDS:
        command = [sys.executable, *arguments]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
