"""Run every demo in turn and write the figures/ gallery."""

from pathlib import Path
import runpy
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    scripts = sorted(HERE.glob("demo_*.py"))
    for s in scripts:
        print(f"\n>>> running {s.name}")
        t0 = time.time()
        runpy.run_path(str(s), run_name="__main__")
        print(f"    finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
