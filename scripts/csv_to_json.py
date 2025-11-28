#!/usr/bin/env python3
"""
Convert a row from X_test.csv to JSON format for API testing
Usage: python scripts/csv_to_json.py [row_number]
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def csv_row_to_json(csv_path: str, row_number: int = 0, output_file: str = "sample_flow.json"):
    """Convert a CSV row to JSON format for API testing"""

    # Read the CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, nrows=row_number + 1)

    if row_number >= len(df):
        print(f"Error: Row {row_number} doesn't exist (only {len(df)} rows)")
        return

    # Get the specified row
    row = df.iloc[row_number]

    # Convert to dictionary
    flow_data = row.to_dict()

    # Save to JSON
    output_path = project_root / output_file
    with open(output_path, "w") as f:
        json.dump(flow_data, f, indent=2)

    print(f"\n✅ Created {output_file}")
    print(f"Features: {len(flow_data)}")
    print(f"\nSample features:")
    for i, (key, value) in enumerate(list(flow_data.items())[:5]):
        print(f"  {key}: {value}")
    print("  ...")
    print(f"\nTest with:")
    print(f"  curl -X POST http://localhost:8000/predict \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d @{output_file}")


if __name__ == "__main__":
    csv_path = project_root / "data" / "processed" / "X_test.csv"

    row_num = 0
    if len(sys.argv) > 1:
        row_num = int(sys.argv[1])

    output_name = "sample_flow.json"
    if len(sys.argv) > 2:
        output_name = sys.argv[2]

    csv_row_to_json(str(csv_path), row_num, output_name)
