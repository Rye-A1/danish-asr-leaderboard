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
3. Add an entry for the model to `scripts/model_metadata.json`, keyed by the
   model name as it appears on the board. It holds the model's release date and
   its [openness annotations](#openness-annotations):
   ```json
   "<model>": {
     "released": {"date": "YYYY-MM-DD", "source": "huggingface"},
     "training_code": null,
     "training_data": "https://…",
     "model_card": "https://…",
     "paper": null
   }
   ```
   `released.date` is what the Over Time chart plots. For a Hugging Face repo it
   is `createdAt` from `https://huggingface.co/api/models/<model>`.
   For a hosted API use `"source": "published"` with the announcement link in a
   `"note"`, or `"source": "best guess"` when no date is published anywhere —
   say in the `"note"` what the guess is based on. A model with no date is
   simply left off that chart. For an API that changes after launch, keep its
   actual release date and add `"plot_by": "submitted"` to `released`; its
   latest result's `"submitted"` date then becomes the current score's plotted
   date.
   To retain an older evaluation of the same model, add a `"history"` array to
   its `results/<model-slug>.json`. Each entry needs `"submitted"`, `"mean_wer"`,
   `"mean_cer"`, and the per-dataset WER/CER scores from that evaluation. The
   top-level fields remain the latest leaderboard result. Older evaluations
   appear in the Over Time mean WER/CER plots, not as extra leaderboard rows.
4. Commit **all of it** and open a pull request. On merge, CI publishes the
   results to the Hugging Face dataset and redeploys the leaderboard
   automatically — no manual `push_results.py` / `push_outputs.py` step needed.

Please include in the PR description: the exact command you ran, the hardware
(for context on `speed_x`), and whether the model is `open` or `proprietary`.

For results to be comparable, do **not** modify the normalisation or metrics —
run the harness as-is.

### Openness annotations

The leaderboard's **Openness** column shows six criteria per model. Each one
that is met links to its evidence, so every claim can be checked in one click:

| Criterion | Source | Counts when… |
|---|---|---|
| Open weights | derived | the weights are downloadable from a Hugging Face repo (gated repos count unless access needs manual approval) |
| Open licence | derived | the repo's licence tag is in `OPEN_LICENSES` in `scripts/update_space.py`: it allows use, modification and redistribution, commercial use included, with no use restrictions (so not CC BY-NC, OpenRAIL or bespoke vendor licences) |
| Open training code | `training_code` | the code used to train this checkpoint is public (fine-tuning code for a fine-tune). |
| Open training data | `training_data` | all data used to train this checkpoint is public (the fine-tuning data for a fine-tune). |
| Model card | `model_card` | there is a filled-in model card, a release blog post, or a detailed documentation page dedicated to the model (e.g. a provider's per-model reference page for a hosted API). An auto-generated "more information needed" card, a generic API reference, or a page that only lists the model's name does not count |
| Paper | `paper` | there is a paper or a technical report. |

The four annotated fields can be found in `scripts/model_metadata.json` and includes a an evidence URL (which is its
source), or `null` when the criterion is not met:

```json
"CoRal-project/roest-v3-whisper-1.5b": {
  "released": {"date": "…", "source": "huggingface"},
  "training_code": "https://github.com/alexandrainst/coral",
  "training_data": "https://huggingface.co/datasets/CoRal-project/coral-v3",
  "model_card": "https://huggingface.co/CoRal-project/roest-v3-whisper-1.5b",
  "paper": null
}
```

The deploy lists any model on the board that is missing one of the four
annotated fields.

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
