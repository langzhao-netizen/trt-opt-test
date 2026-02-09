# shared/

Small shared config/data tracked in the repo. Large assets are gitignored.

- **`models/`** – Gitignored (~69GB). Put local HF model copies here for offline PTQ, or use HF model IDs in scripts.
- **`data/`** – Small shared data (e.g. calibration lists, configs). Tracked.
- **`utils/`** – Small shared utilities. Tracked.

After clone, run `./scripts/setup.sh` for one-click env; use `shared/models/` only if you need local model copies.
