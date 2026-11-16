"""Local, offline benchmark backends for pinned Fluid and Qwen checkpoints."""
import importlib.util
import os
from pathlib import Path
from danish_asr_leaderboard.backends.base import Backend, register
from danish_asr_leaderboard.backends.nemo_backend import _text_of


class FluidBackend(Backend):
    def transcribe(self, audio_paths, *, batch_size):
        import torch
        out = []
        for i in range(0, len(audio_paths), 400):
            chunk = audio_paths[i:i+400]
            with torch.inference_mode():
                result = self.model.transcribe(chunk, batch_size=batch_size, verbose=False)
            if isinstance(result, tuple):
                result = result[0]
            if len(result) != len(chunk):
                raise RuntimeError('Fluid returned an unexpected number of transcripts')
            out.extend(_text_of(x) for x in result)
            print(f'  {len(out)}/{len(audio_paths)} transcribed', flush=True)
        return out

    def transcribe_one(self, audio_path):
        return self.transcribe([audio_path], batch_size=1)[0]


@register('fluid-speech-pro')
def load_fluid(model_ref, options):
    spec = importlib.util.spec_from_file_location('fluid_official', Path(model_ref).parent / 'fluid_speech.py')
    if not spec or not spec.loader:
        raise RuntimeError('Fluid loader is missing')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load(model_ref, map_location=options.device, clean_mode=False).to(options.device).eval()
    strategy = str(model.cfg.decoding.strategy)
    if strategy not in {'greedy', 'greedy_batch'}:
        raise RuntimeError(f'Expected greedy decoding, got {strategy}')
    os.environ['EVAL_DECODING'] = strategy
    os.environ['EVAL_ACTUAL_DTYPE'] = str(next(model.parameters()).dtype).removeprefix('torch.')
    return FluidBackend(model, options=options)


class QwenBackend(Backend):
    def __init__(self, model, processor, options):
        super().__init__(model, options=options)
        self.processor = processor

    def transcribe(self, audio_paths, *, batch_size):
        import torch
        import soundfile as sf
        import numpy as np
        from scipy.signal import resample_poly
        from math import gcd
        out = []
        for i in range(0, len(audio_paths), batch_size):
            chunk = audio_paths[i:i+batch_size]
            audio = []
            for path in chunk:
                wave, sr = sf.read(path, dtype='float32')
                if wave.ndim == 2:
                    wave = wave.mean(axis=1)
                if sr != 16000:
                    divisor = gcd(sr, 16000)
                    wave = resample_poly(wave, 16000 // divisor, sr // divisor).astype(np.float32)
                audio.append(wave)
            inputs = self.processor.apply_transcription_request(audio=audio, language=['Danish'] * len(chunk)).to(self.model.device, self.model.dtype)
            with torch.inference_mode():
                ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generated = ids[:, inputs['input_ids'].shape[1]:]
            texts = self.processor.decode(generated, return_format='transcription_only')
            if len(texts) != len(chunk) or not all(isinstance(x, str) for x in texts):
                raise RuntimeError('Qwen returned malformed transcripts')
            out.extend(texts)
            if len(out) % 400 < batch_size or len(out) == len(audio_paths):
                print(f'  {len(out)}/{len(audio_paths)} transcribed', flush=True)
        return out

    def transcribe_one(self, audio_path):
        return self.transcribe([audio_path], batch_size=1)[0]


@register('qwen3-asr')
def load_qwen(model_ref, options):
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    processor = AutoProcessor.from_pretrained(model_ref, local_files_only=True)
    dtype = torch.bfloat16 if options.device == 'cuda' else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(model_ref, local_files_only=True, dtype=dtype).to(options.device).eval()
    os.environ['EVAL_DECODING'] = 'greedy'
    os.environ['EVAL_ACTUAL_DTYPE'] = str(model.dtype).removeprefix('torch.')
    return QwenBackend(model, processor, options)

class PhononBackend(Backend):
    def transcribe(self, audio_paths, *, batch_size):
        out = []
        for i, path in enumerate(audio_paths):
            text, _, _ = self.model.transcribe(path)
            if not isinstance(text, str):
                raise RuntimeError('Phonon returned a malformed transcript')
            out.append(text)
            if (i + 1) % 100 == 0 or i + 1 == len(audio_paths):
                print(f'  {i + 1}/{len(audio_paths)} transcribed', flush=True)
        return out

    def transcribe_one(self, audio_path):
        return self.transcribe([audio_path], batch_size=1)[0]


@register('phonon-2.5')
def load_phonon(model_ref, options):
    import fermion
    model = fermion.load_speech(model_ref, quiet=True)
    details = model.describe()
    if details.get('engine') != 'cpu':
        raise RuntimeError('Expected the documented CPU engine')
    os.environ['EVAL_DECODING'] = 'greedy; repetition_penalty=1.05 (default)'
    os.environ['EVAL_ACTUAL_DTYPE'] = 'five-value quantized (CPU)'
    print('Phonon runtime:', details['runtime'], flush=True)
    return PhononBackend(model, options=options)
