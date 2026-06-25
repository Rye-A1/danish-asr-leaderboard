# Contributing

Thanks for helping improve the Danish ASR Leaderboard.

There are two ways to get a model onto the board, depending on whether you've
run the evaluation yourself.

## Requesting a model — *we run it* (open an issue)

If you can't (or would rather not) run the eval yourself — e.g. it's a model you
saw and want benchmarked — open an
[issue](https://github.com/Rye-A1/danish-asr-leaderboard/issues) with the model
id, backend, and where to find it. We'll run it through the harness and add it.

## Submitting a score — *you ran it* (open a PR)

1. Install the core library and the backend you need (see the
   [README](README.md#installation)).
2. Run the evaluation on all core test sets:
   ```bash
   danish-asr-eval --model <model> --backend <backend>
   ```
   This writes two things, exactly as the PR expects them:
   - `results/<model-slug>.json` — the scores and metadata
   - `outputs/<model-slug>/` — the raw per-dataset transcriptions
     (`<dataset>.jsonl` + `meta.json`)
3. Commit **both** and open a pull request. On merge, CI publishes them to the
   Hugging Face dataset and redeploys the leaderboard automatically — no manual
   `push_results.py` / `push_outputs.py` step needed.

Please include in the PR description: the exact command you ran, the hardware
(for context on `speed_x`), and whether the model is `open` or `proprietary`.

For results to be comparable, do **not** modify the normalisation or metrics —
run the harness as-is.

> **Verification.** Whichever path a model arrives by, we re-run the evaluation
> ourselves before publishing, to confirm the scores reproduce on our hardware
> and catch any configuration differences. Submitting the raw `outputs/` lets us
> diff transcriptions directly.

## Adding a backend

1. Create `danish_asr_leaderboard/backends/<name>_backend.py`.
2. Subclass `Backend`, implement `transcribe_one` (and `transcribe_batch` if the
   model batches efficiently).
3. Decorate the loader with `@register("<name>")`.
4. Import the module in `danish_asr_leaderboard/backends/__init__.py` so it
   registers on import.
5. Add `requirements/<name>.txt` (start with `-r base.txt`).
6. Keep heavy third-party imports *inside* the loader / methods so importing the
   package stays cheap.

Verify it registers:

```bash
python3 -c "from danish_asr_leaderboard.backends import available_backends; print(available_backends())"
danish-asr-eval --help     # your backend should appear in --backend choices
```

## Code style

- Keep changes minimal and consistent with the surrounding code.
- No new runtime dependencies in the core package — backend frameworks belong in
  `requirements/<backend>.txt`.
- Run a smoke test (`--max-samples 10` on a small model) before submitting code
  that touches the harness.
