#!/usr/bin/env python3
"""
Upload a folder of subdirs to Hugging Face Hub (one repo, one folder per subdir).
Shows progress [i/N], continues on failure, optional tqdm.
Usage:
  REPO_ID=... HF_TOKEN=... python3 scripts/upload_to_hf.py [FOLDER_ROOT]
  FOLDER_ROOT defaults to PROJECT_ROOT/outputs/ckpts; use PROJECT_ROOT/models for models.
"""
import os
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    folder_root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(project_root, "outputs", "ckpts"))
    repo_id = os.environ.get("REPO_ID")
    token = os.environ.get("HF_TOKEN", "")
    if not repo_id:
        print("REPO_ID not set. Example: export REPO_ID=your-username/trt-models-test", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    if not os.path.isdir(folder_root):
        print(f"Not a directory: {folder_root}", file=sys.stderr)
        sys.exit(1)

    subdirs = [d for d in os.listdir(folder_root) if os.path.isdir(os.path.join(folder_root, d))]
    subdirs.sort()
    if not subdirs:
        print(f"No subdirs in {folder_root}")
        return

    api = HfApi()
    n = len(subdirs)
    failed = []

    iterator = tqdm(subdirs, desc="Upload", unit="folder") if use_tqdm else subdirs
    for i, name in enumerate(iterator):
        if not use_tqdm:
            print(f"[{i+1}/{n}] Uploading {name} -> {repo_id} ...", flush=True)
        folder_path = os.path.join(folder_root, name)
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                path_in_repo=name,
                token=token or None,
            )
            if not use_tqdm:
                print(f"  OK", flush=True)
        except Exception as e:
            if use_tqdm:
                tqdm.write(f"  FAILED {name}: {e}")
            else:
                print(f"  FAILED: {e}", flush=True)
            failed.append((name, str(e)))

    if failed:
        print(f"\nFailed {len(failed)}/{n}:", file=sys.stderr)
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
    else:
        print(f"\nDone. All {n} folders uploaded.")
    print(f"Download: huggingface-cli download {repo_id} --local-dir ./out")


if __name__ == "__main__":
    main()
