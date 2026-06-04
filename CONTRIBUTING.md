# Contributing

Thanks for helping improve the Danish ASR Leaderboard.

## Submitting a model result

1. Install the core library and the backend you need (see the
   [README](README.md#installation)).
2. Run the evaluation on all core test sets:
   ```bash
   danish-asr-eval --model <model> --backend <backend>
   ```
   This writes `results/<model-slug>.json`.
3. Submit the result in one of two ways:
   - **PR (preferred):** commit your `results/<model-slug>.json` and open a pull
     request. Maintainers run `scripts/push_results.py` to publish.
   - **Discussion:** if you can't run the eval yourself, open a
     [discussion](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard/discussions)
     on the Space with the model id and backend.

Please include in the PR description: the exact command you ran, the hardware
(for context on `speed_x`), and whether the model is `open` or `proprietary`.

For results to be comparable, do **not** modify the normalisation or metrics —
run the harness as-is.

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
