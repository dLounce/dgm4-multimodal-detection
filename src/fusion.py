import torch
import torch.nn as nn


class CoAttnBlock(nn.Module):
    def __init__(self, d, heads=8, ff=2048, p=0.1):
        super().__init__()
        self.ln_iq, self.ln_ikv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ln_tq, self.ln_tkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.i2t = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.t2i = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ln_if, self.ln_tf = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffi = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        self.fft = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))

    def forward(self, img, txt, txt_pad):                     # txt_pad: (B,L) True=PAD
        # image attends to text (mask out PAD text), then its own FFN
        img = img + self.i2t(self.ln_iq(img), self.ln_ikv(txt), self.ln_ikv(txt),
                             key_padding_mask=txt_pad, need_weights=False)[0]
        img = img + self.ffi(self.ln_if(img))
        # text attends to image (no mask — every patch token is valid)
        txt = txt + self.t2i(self.ln_tq(txt), self.ln_tkv(img), self.ln_tkv(img),
                             need_weights=False)[0]
        txt = txt + self.fft(self.ln_tf(txt))
        return img, txt


class Fusion(nn.Module):
    def __init__(self, di=1024, dt=768, d=768, layers=2):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(di, d), nn.LayerNorm(d))   # LN tames CLIP outlier activations
        self.txt_proj = nn.Sequential(nn.Linear(dt, d), nn.LayerNorm(d))
        self.blocks = nn.ModuleList([CoAttnBlock(d) for _ in range(layers)])
        self.trunk = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.2))
        self.bin_head, self.multi_head = nn.Linear(d, 1), nn.Linear(d, 4)

    def forward(self, img_tok, txt_tok, txt_pad):
        img, txt = self.img_proj(img_tok), self.txt_proj(txt_tok)
        for blk in self.blocks:
            img, txt = blk(img, txt, txt_pad)
        m = (~txt_pad).float().unsqueeze(-1)
        joint = torch.cat([img[:, 0], (txt * m).sum(1) / m.sum(1).clamp(min=1)], -1)  # img CLS + masked-mean text
        z = self.trunk(joint)
        return self.bin_head(z).squeeze(-1), self.multi_head(z), img, txt