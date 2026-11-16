"""Run original Parakeet Ultra tensors in the leaderboard's NeMo TDT decoder."""
import os,re
from pathlib import Path
from danish_asr_leaderboard.backends.base import Backend,register
from danish_asr_leaderboard.backends.nemo_backend import _text_of

RULES=[
 (r'encoder\.subsampling\.layers\.(\d+)\.(weight|bias)',r'encoder.pre_encode.conv.\1.\2'),
 (r'encoder\.subsampling\.linear\.(weight|bias)',r'encoder.pre_encode.out.\1'),
 (r'encoder\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)',r'encoder.layers.\1.self_attn.linear_q.\2'),
 (r'encoder\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)',r'encoder.layers.\1.self_attn.linear_k.\2'),
 (r'encoder\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)',r'encoder.layers.\1.self_attn.linear_v.\2'),
 (r'encoder\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)',r'encoder.layers.\1.self_attn.linear_out.\2'),
 (r'encoder\.layers\.(\d+)\.self_attn\.relative_k_proj\.(weight|bias)',r'encoder.layers.\1.self_attn.linear_pos.\2'),
 (r'encoder\.layers\.(\d+)\.self_attn\.bias_u',r'encoder.layers.\1.self_attn.pos_bias_u'),
 (r'encoder\.layers\.(\d+)\.self_attn\.bias_v',r'encoder.layers.\1.self_attn.pos_bias_v'),
 (r'encoder\.layers\.(\d+)\.conv\.norm\.(.+)',r'encoder.layers.\1.conv.batch_norm.\2'),
 (r'encoder\.layers\.(\d+)\.(.+)',r'encoder.layers.\1.\2'),
 (r'decoder\.embedding\.weight',r'decoder.prediction.embed.weight'),
 (r'decoder\.lstm\.(.+)',r'decoder.prediction.dec_rnn.lstm.\1'),
 (r'decoder\.decoder_projector\.(weight|bias)',r'joint.pred.\1'),
 (r'encoder_projector\.(weight|bias)',r'joint.enc.\1'),
 (r'joint\.head\.(weight|bias)',r'joint.joint_net.2.\1'),
]

class UltraNemoBackend(Backend):
 def transcribe(self,audio_paths,*,batch_size):
  import torch
  out=[]
  for i in range(0,len(audio_paths),400):
   with torch.inference_mode(): result=self.model.transcribe([str(Path(p).resolve()) for p in audio_paths[i:i+400]],batch_size=batch_size,verbose=False)
   if isinstance(result,tuple): result=result[0]
   out.extend(_text_of(x) for x in result)
   print(f'  {len(out)}/{len(audio_paths)} transcribed',flush=True)
  if len(out)!=len(audio_paths): raise RuntimeError('unexpected output count')
  return out
 def transcribe_one(self,audio_path): return self.transcribe([audio_path],batch_size=1)[0]

@register('ultra-nemo')
def load_ultra(model_ref,options):
 import torch
 import nemo.collections.asr as nemo_asr
 from safetensors.torch import load_file
 base=os.environ['PARAKEET_V3_NEMO']
 model=nemo_asr.models.ASRModel.restore_from(base,map_location='cpu')
 sd=model.state_dict(); source=load_file(str(Path(model_ref)/'model.safetensors'),device='cpu'); mapped={}; ignored=[]
 for key,tensor in source.items():
  for pat,tpl in RULES:
   if re.fullmatch(pat,key): target=re.sub(pat,tpl,key); break
  else: ignored.append(key); continue
  if target not in sd or target in mapped: raise RuntimeError('invalid or duplicated tensor mapping')
  t=sd[target]
  if tensor.shape != t.shape:
   if tensor.numel()==t.numel() and tensor.dim()+1==t.dim() and t.shape[-1]==1: tensor=tensor.reshape(t.shape)
   else: raise RuntimeError('mismatched tensor shape')
  mapped[target]=tensor.to(t.dtype)
 missing=[k for k in sd if k not in mapped and not k.endswith('num_batches_tracked') and not k.startswith('preprocessor.')]
 if missing or not all(k.startswith('vad_head.') for k in ignored) or len(ignored)!=6: raise RuntimeError(f'incomplete mapping: {len(missing)} missing, {len(ignored)} ignored')
 sd.update(mapped); model.load_state_dict(sd,strict=True)
 check=model.state_dict()
 if not all(torch.equal(check[k],v) for k,v in mapped.items()): raise RuntimeError('converted weights differ')
 strategy=str(model.cfg.decoding.strategy)
 if strategy not in {'greedy','greedy_batch'} or int(model.cfg.decoding.greedy.max_symbols) != 10: raise RuntimeError('unexpected decoding strategy or per-frame limit')
 print(f'Verified {len(mapped)} mapped tensors; only 6 VAD tensors omitted; decoder {strategy}',flush=True)
 print('NeMo greedy configuration:',model.cfg.decoding.greedy,flush=True)
 os.environ['EVAL_DECODING']=f'{strategy}, NeMo TDT'
 os.environ['EVAL_ACTUAL_DTYPE']=str(next(model.parameters()).dtype).removeprefix('torch.')
 return UltraNemoBackend(model.to(options.device).eval(),options=options)
