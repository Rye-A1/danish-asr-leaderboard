import os
import custom_backends  # noqa: F401 -- register the local backend
from danish_asr_leaderboard import cli
cli._notify=lambda _message: None
_write_meta=cli.write_meta
def write_meta(outputs_dir,model_id,meta):
 meta.update({'model_revision':os.environ['EVAL_MODEL_REVISION'],'backend':os.environ['EVAL_BACKEND'],'decoding':os.environ['EVAL_DECODING'],'dtype':os.environ['EVAL_ACTUAL_DTYPE'],'batch_size':int(os.environ['EVAL_BATCH_SIZE']),'speed_hardware':'NVIDIA A100 80GB PCIe','language_route':'automatic','runtime':'NeMo 3.0; original Ultra tensors, NVIDIA v3 tokenizer and preprocessor'})
 return _write_meta(outputs_dir,model_id,meta)
cli.write_meta=write_meta
if __name__=='__main__': cli.main()
