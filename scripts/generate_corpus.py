#!/usr/bin/env python3
"""
OrgTrace — CLI Corpus Generator
Generates the synthetic communication corpus and ground truth manifest.
Used at Docker build time and for local development.

Usage:
    python scripts/generate_corpus.py --seed 42
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Generate the OrgTrace synthetic corpus"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Import and run the generator
    import generator as gen

    # Override seed if specified
    import random
    random.seed(args.seed)
    gen.SEED = args.seed

    corpus, ground_truth = gen.generate_corpus()
    gen.save(corpus, ground_truth)

    print(f"\n✅ Corpus generated with seed={args.seed}")
    print(f"   Messages: {len(corpus)}")
    print(f"   Output:   {gen.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
