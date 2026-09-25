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
3. Add the model's release date to `scripts/release_dates.json`, which is what
   the Over Time chart plots. For a Hugging Face repo it is `createdAt` from
   `https://huggingface.co/api/models/<model>`:
   ```json
   "<model>": {"released": "YYYY-MM-DD", "source": "huggingface"}
   ```
   For a hosted API use `"source": "published"` with the announcement link in a
   `"note"`, or `"source": "best guess"` when no date is published anywhere —
   say in the `"note"` what the guess is based on. A model with no entry is
   simply left off that chart.
4. Commit **all of it** and open a pull request. On merge, CI publishes the
   results to the Hugging Face dataset and redeploys the leaderboard
   automatically — no manual `push_results.py` / `push_outputs.py` step needed.

Please include in the PR description: the exact command you ran, the hardware
(for context on `speed_x`), and whether the model is `open` or `proprietary`.

For results to be comparable, do **not** modify the normalisation or metrics —
run the harness as-is.

> **Verification.** Whichever path a model arrives by, we re-run the evaluation
> ourselves before publishing, to confirm the scores reproduce on our hardware
> and catch any configuration differences. Submitting the raw `outputs/` lets us
> diff transcriptions directly.

## Model openness and feature profiles

The leaderboard's **Openness** and **Features** columns read
`scripts/model_profiles.json`. A profile is optional; unreviewed fields display
as unknown. Hover, focus, or tap a column value to see the status map and source
links. Add a source URL for each confirmed `yes` claim. The four openness
fields are:

| Field | What a `yes` means |
|---|---|
| `data` | The exact fine-tuning data, splits, and filtering are disclosed and accessible under published terms. For a base model, assess its full training corpus. |
| `code` | Scripts, preprocessing, and configuration for this checkpoint's training run are public. Inference code alone does not count. |
| `model_card` | The card explains base-model lineage, intended use, training data and method, evaluation, and limitations. |
| `license` | The checkpoint and its base permit commercial reuse and redistribution without model-specific field-of-use restrictions. |

An optional `report` links a paper or technical report about this checkpoint.
It is displayed in the details but does not add an openness tile or score.
Do not count a base model's paper for a fine-tune.

The separate feature fields are `punctuation_case` (cased and punctuated Danish
output, either by default or through an option), `timestamps` (word or segment
output for Danish), `diarization` (speaker labels), and `streaming`
(incremental output). A formatting `yes` does not require independent controls.
Mark `no` when the scored output contains no cased, punctuated text and no
formatting option is documented; use `unknown` if neither output nor documentation
is available. Saved raw hypotheses are evidence for what the released inference
path actually returns.
Link documentation or a reproducible example for the released checkpoint or
its official inference package. An external aligner or chunking demo is not
automatically a native model feature. `partial` means only part of the claim is
supported, `no` means documented absence or restrictive terms, and `unknown`
means the available sources do not decide it.

For streaming, a completed-file API that sends partial text is `partial`; a
documented path that accepts ongoing audio and emits interim results is `yes`.

Each reviewed model may set `reviewed_source` and `reviewed_on` in
`model_profiles.json`. Missing fields then appear as reviewed `unknown` with a
link to the source instead of implying that nobody checked them. This does not
add a positive tile or infer unsupported features from a base model.

Hub license tags are review leads, not automatic positive license claims: a
model card, attached terms, or base license can narrow them. A non-commercial
tag is marked `no`; custom commercial licenses with field-of-use restrictions
are `partial`. Dataset and arXiv tags are also leads, not proof that all data
is open or the paper covers this checkpoint. Manual decisions in
`model_profiles.json` override these hints. Example:

```json
{
  "example/asr": {
    "openness": {
      "code": {
        "state": "yes",
        "url": "https://github.com/example/asr",
        "detail": "Training recipe and scripts"
      }
    },
    "features": {
      "timestamps": {
        "state": "yes",
        "url": "https://example.org/asr/timestamps"
      }
    }
  }
}
```

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
