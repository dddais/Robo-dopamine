"""Exact checkpoint architecture; no random, missing, or unused progress weights.

The MLP and prog-token readout follow robometer/robometer models/rbm.py,
models/heads.py and evals/eval_server.py. Legacy similarity weights are loaded
for a strict state-dict audit but are not used for progress inference.
"""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, Qwen3VLModel
from safetensors.torch import load_file
from accelerate import init_empty_weights


class RobometerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3VLModel(config)
        d = config.text_config.hidden_size
        def head(n):
            return nn.Sequential(nn.Linear(d, d//2), nn.LayerNorm(d//2), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(d//2, n))
        self.progress_head = head(10)
        self.success_head = head(1)
        self.preference_head = head(1)
        self.similarity_head = head(1)
        self.frame_pool_attn = nn.Linear(d, 1, bias=False)

    @classmethod
    def load(cls, path, device='cuda:0'):
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        cfg._attn_implementation = 'sdpa'
        cfg.text_config._attn_implementation = 'sdpa'
        cfg.vision_config._attn_implementation = 'sdpa'
        with init_empty_weights(include_buffers=False):
            model = cls(cfg)
        state = {}
        for shard in sorted(Path(path).glob('model-*.safetensors')):
            state.update(load_file(str(shard)))
        audit = model.load_state_dict(state, strict=True, assign=True)
        model = model.to(device=device, dtype=torch.bfloat16).eval()
        model.loading_audit = {'missing_keys': list(audit.missing_keys), 'unexpected_keys': list(audit.unexpected_keys),
                               'loaded_tensors': len(state), 'strict': True}
        return model

    def forward(self, **inputs):
        return self.model(**inputs)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def read_progress(self, hidden, ids, prog_token):
        progress, success, positions = [], [], []
        for i in range(len(ids)):
            p = (ids[i] == prog_token).nonzero(as_tuple=True)[0]
            if not len(p): raise ValueError('Missing trained progress readout token')
            h = hidden[i, p]
            logits = self.progress_head(h).float()
            success_logits = self.success_head(h).float()
            if not torch.isfinite(logits).all() or not torch.isfinite(success_logits).all():
                raise ValueError('Non-finite Robometer head logits')
            values = (logits.softmax(-1)*torch.linspace(0, 1, 10, device=logits.device)).sum(-1)
            progress.append(values.detach().cpu().tolist())
            success.append(success_logits.sigmoid().squeeze(-1).detach().cpu().tolist())
            positions.append(p.detach().cpu().tolist())
        return progress, success, positions
