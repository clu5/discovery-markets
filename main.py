import os
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
    aurum = AurumCLI(schema_path=schema_path)
    
    # Add CSV source and profile
    print("Adding and profiling data source...")
    aurum.add_csv_data_source("csv_source", csv_dir)
    aurum.profile("csv_source")
    
    # Build network model
    print("Building network model...")
    network = FieldNetwork()
    store = StoreHandler()
    
    # Get fields from store
    fields_gen = store.get_all_fields()
    network.init_meta_schema(fields_gen)
    
    # Build PKFK relationships
    print("Building PKFK relationships...")
    build_pkfk_relation(network)
    
    # Extract join paths from PKFK relationships
    print("Extracting join paths...")
    join_paths = []
    pkfk_graph = network._get_underlying_repr_graph()
    for src, tgt in pkfk_graph.edges():
        if 'PKFK' in pkfk_graph[src][tgt]:
            src_info = network.get_info_for([src])[0]
            tgt_info = network.get_info_for([tgt])[0]
            join_paths.append({
                'tbl1': src_info[2], # source table
                'col1': src_info[3], # source column 
                'tbl2': tgt_info[2], # target table
                'col2': tgt_info[3]  # target column
            })
    
    # Save join paths in format for Metam
    save_name = f'{Path(csv_dir).stem}_join_paths.csv'
    output_dir = Path(output_dir)
    if output_dir.exists() is False:
        output_dir.mkdir()
    print(f"Found {len(join_paths)} join paths")
    df = pd.DataFrame(join_paths)
    df.to_csv(output_dir / save_name, index=False)
    print(f"Saved join paths to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default='data/nyc_csvs', help="Path to directory containing CSV files")
    parser.add_argument("--output", default='saved_aurum_graphs', help="Path to save join paths CSV") 
    parser.add_argument("--schema", default='configs/profile_schema.yml', help="Path to the profile schema file")
    args = parser.parse_args()
    
    build_aurum_and_extract_joins(args.csv_dir, args.output, args.schema)