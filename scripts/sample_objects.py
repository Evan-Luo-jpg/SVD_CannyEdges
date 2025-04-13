import objaverse
import random
import json
import os

# Load all available UIDs
uids = objaverse.load_uids()

# Sample 5
sampled_uids = random.sample(uids, 5)

# Download and get metadata
objects = objaverse._download_object(sampled_uids)

# Format into expected JSON
formatted = []
for uid in sampled_uids:
    metadata = objaverse.get_metadata(uid)  # Assuming get_metadata() retrieves metadata for a UID
    formatted.append({
        "fileIdentifier": objects["fileIdentifier"],
        "sha256": metadata["sha256"],
        "repo": metadata["repo"],
        "ref": metadata["ref"],
        "path": metadata["path"]
    })

# Save to file
os.makedirs("scripts", exist_ok=True)
with open("scripts/my_objects.json", "w") as f:
    json.dump(formatted, f, indent=2)

print("Saved to scripts/my_objects.json")