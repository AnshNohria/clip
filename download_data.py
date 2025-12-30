import os
from huggingface_hub import snapshot_download

# Define the target directory
local_dir = r"D:\datasets\RemoteCLIP"

# Create the directory if it doesn't exist
if not os.path.exists(local_dir):
    print(f"Creating directory: {local_dir}")
    os.makedirs(local_dir)
else:
    print(f"Directory already exists: {local_dir}")

print(f"Downloading dataset to {local_dir}...")

try:
    snapshot_download(
        repo_id="gzqy1026/RemoteCLIP",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete.")
except Exception as e:
    print(f"An error occurred during download: {e}")
