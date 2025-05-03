import os

from aurum import networkbuildercoordinator
from pathlib import Path
from aurum.aurum_cli import AurumCLI
from aurum.knowledgerepr.fieldnetwork import FieldNetwork
from aurum.knowledgerepr.networkbuilder import build_pkfk_relation
from aurum.elasticstore import StoreHandler
import pandas as pd


def build_aurum_and_extract_joins(csv_dir: str, output_dir: str, schema_path: str):
    """
    Build Aurum model from CSV files and extract join paths for Metam

    Args:
        csv_dir: Path to directory containing CSV files
        output_dir: Dir to save join paths CSV
        schema_path: Path to the ddprofiler schema file
    """
    # Convert to absolute path
    csv_dir = os.path.abspath(csv_dir)
    output_dir = os.path.abspath(output_dir)
    schema_path = os.path.abspath(schema_path)

    # Initialize Aurum CLI
    aurum = AurumCLI()

    # Add CSV source and profile
    print("Adding and profiling data source...")
    aurum.add_csv_data_source("csv_source", csv_dir)
    aurum.profile("csv_source", schema_path=str(Path(schema_path).absolute()))
    print("finished profiling".center(70, "-"))

    # Build network model
    print("Building network model...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    networkbuildercoordinator.main(output_path=output_dir)
    print("finished building network model".center(70, "-"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_dir",
        default="data/nyc_csvs",
        help="Path to directory containing CSV files",
    )
    parser.add_argument(
        "--output", default="saved_aurum_graphs", help="Path to save join paths CSV"
    )
    parser.add_argument(
        "--schema",
        default="configs/profile_schema.yml",
        help="Path to the profile schema file",
    )
    args = parser.parse_args()

    build_aurum_and_extract_joins(args.csv_dir, args.output, args.schema)
