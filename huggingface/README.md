# Hugging Face publishing files

- `space/` is the local source for the leaderboard Space. The deployment script generates `cover.jpeg` and uploads its files to the **root** of the Space repository.
- `dataset_card.md` is the editable source for the dataset repository's `README.md`.
- `hf_downloads.json` is a **fallback seed**, not the current download history. The scheduled workflow first restores the latest history from the dataset repository's `data/hf_downloads.json`, then adds and publishes the new snapshot. The Space reads that published file first and uses this seed only if it is unavailable.
