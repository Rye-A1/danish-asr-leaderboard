import os
import custom_backends  # noqa: F401 -- register the local backend
from danish_asr_leaderboard import cli
cli._notify = lambda _message: None
_write_meta = cli.write_meta

def _write_run_meta(outputs_dir, model_id, meta):
    meta.update({'model_revision': os.environ['EVAL_MODEL_REVISION'], 'backend': os.environ['EVAL_BACKEND'], 'decoding': os.environ['EVAL_DECODING'], 'dtype': os.environ['EVAL_ACTUAL_DTYPE'], 'batch_size': int(os.environ['EVAL_BATCH_SIZE']), 'speed_hardware': 'NVIDIA A100 80GB PCIe', 'target_language': 'da'})
    if meta['backend'] == 'phonon-2.5':
        meta['speed_hardware'] = 'AMD EPYC 7V13 CPU, 4 pinned vCPUs'
        meta['language_route'] = 'automatic'
    if meta['backend'] == 'fluid-speech-pro':
        meta['clean_mode'] = False
        meta['language_route'] = 'automatic'
    return _write_meta(outputs_dir, model_id, meta)

cli.write_meta = _write_run_meta
if __name__ == '__main__':
    cli.main()
