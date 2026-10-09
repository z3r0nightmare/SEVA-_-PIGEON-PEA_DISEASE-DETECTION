"""
High-DPI Research Figure Generator Script.
Executes the visualization suite to generate all 300+ DPI research graphics in `figures/`.

Usage:
    python scripts/generate_all_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.generate_research_plots import generate_all_figures


def main():
    generate_all_figures()


if __name__ == "__main__":
    main()
