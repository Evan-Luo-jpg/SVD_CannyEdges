import objaverse.xl as oxl
import pandas as pd
import json
import os

OUTPUT_PATH = "scripts/my_5_objects.json"
NUM_OBJECTS = 5

def sample_from_annotations(n=5, out_path=OUTPUT_PATH):
    # Step 1: Load the full metadata dataframe
    annotations = oxl.get_alignment_annotations(download_dir="~/.objaverse")

    # Step 2: Sample N objects
    sampled = annotations.sample(n)

    # Step 3: Format the JSON output
    result = []
    for _, row in sampled.iterrows():
        result.append({
            "sha256": row["sha256"],
            "fileIdentifier": row["fileIdentifier"],
            "source": row["source"],
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {n} sampled objects to {out_path}")

if __name__ == "__main__":
    sample_from_annotations()
