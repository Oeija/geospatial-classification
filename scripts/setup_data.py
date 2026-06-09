"""Standalone script to ensure the dataset is present (download + extract if needed)."""

import logging

from geospatial_classification.data.setup import ensure_extracted

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ensure_extracted()
