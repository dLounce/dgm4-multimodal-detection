"""Natural-language explanation of a detector verdict, via LLaVA-1.5.

The detector and grounding heads produce a structured analysis (probability,
manipulation types, suspicious face box, suspicious caption words); LLaVA then
turns that, plus the image itself, into a short human-readable explanation.

The prompt is deliberately constrained: LLaVA is told to use only the supplied
analysis and what it can see, never to name or guess real people, and not to
invent evidence — the model grounds the explanation rather than free-associating
about a real news photo.

Construct an Explainer with your trained components and call analyze() for the
raw signals or explain() for the written verdict.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from data import ATOMIC, resolve_path

ATOMIC_NAMES = {
    "face_swap": "face swap",
    "face_attribute": "face attribute edit",
    "text_swap": "caption swapped",
    "text_attribute": "caption attribute changed",
}


def load_llava(llava_path, dtype=torch.float16):
    """Load LLaVA-1.5-7B for generating explanations."""
    llava = (
        LlavaForConditionalGeneration
        .from_pretrained(llava_path, local_files_only=True)
        .to("cuda", dtype)
        .eval()
    )
    proc = AutoProcessor.from_pretrained(llava_path, local_files_only=True)
    return llava, proc


class Explainer:
    def __init__(self, detector, clip_processor, box_head, token_head,
                 llava, llava_processor, device="cuda"):
        self.model = detector
        self.proc = clip_processor
        self.box_head = box_head
        self.token_head = token_head
        self.llava = llava
        self.llava_proc = llava_processor
        self.device = device
        # CLIP needs token ids even to embed an image; this dummy text lets us
        # encode bare face crops (the text content is ignored for image_embeds).
        self._dummy = clip_processor.tokenizer("a photo", return_tensors="pt")

    @torch.no_grad()
    def analyze(self, e):
        """Run the detector + grounding heads, returning the raw signals."""
        dev = self.device
        img = Image.open(resolve_path(e["image"])).convert("RGB")
        enc = self.proc(text=[e["text"]], images=[img],
                        return_tensors="pt", padding=True, truncation=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ob, om, fi, ft, ie, te = self.model(
                enc["pixel_values"].to(dev),
                enc["input_ids"].to(dev),
                enc["attention_mask"].to(dev),
            )
        p_fake = torch.sigmoid(ob.float()).item()
        pm = torch.sigmoid(om.float())[0].tolist()
        types = [ATOMIC_NAMES[a] for a, p in zip(ATOMIC, pm) if p > 0.5]

        sus = []
        if pm[2] > 0.5 or pm[3] > 0.5:                      # text manip -> flag words
            tp = torch.sigmoid(self.token_head(ft.float()).squeeze(-1)[0])
            off = self.proc.tokenizer(e["text"], truncation=True, max_length=77,
                                      return_offsets_mapping=True)["offset_mapping"]
            words = e["text"].split()
            spans, i = [], 0
            for w in words:
                s = e["text"].index(w, i)
                spans.append((s, s + len(w)))
                i = s + len(w)
            wids = set()
            for ti, (ts, te_) in enumerate(off):
                if ts == te_ or ti >= len(tp) or tp[ti] <= 0.5:
                    continue
                mid = (ts + te_) / 2
                w = next((k for k, (a_, b_) in enumerate(spans) if a_ <= mid < b_), None)
                if w is not None:
                    wids.add(w)
            sus = [words[k] for k in sorted(wids)]

        box = None
        if (pm[0] > 0.5 or pm[1] > 0.5) and e.get("mtcnn_boxes"):   # face manip -> pick box
            crops = [img.crop(tuple(b)) for b in e["mtcnn_boxes"]]
            px = self.proc(images=crops, return_tensors="pt")["pixel_values"].to(dev)
            n = len(crops)
            ids = self._dummy["input_ids"].expand(n, -1).to(dev)
            am = self._dummy["attention_mask"].expand(n, -1).to(dev)
            co = self.model.clip(pixel_values=px, input_ids=ids, attention_mask=am)
            best = int(torch.sigmoid(self.box_head(co.image_embeds.float()).squeeze(-1)).argmax())
            box = e["mtcnn_boxes"][best]

        return {"p_fake": p_fake, "types": types, "sus": sus, "box": box, "img": img}

    def explain(self, e):
        """Analyze, then have LLaVA write a 2-3 sentence verdict explanation."""
        dev = self.device
        a = self.analyze(e)
        verdict = "MANIPULATED" if a["p_fake"] > 0.5 else "AUTHENTIC"
        ev_words = ", ".join(a["sus"]) if a["sus"] else "NONE flagged"
        ev_types = ", ".join(a["types"]) if a["types"] else "none"
        box_str = ("yes, box " + str(a["box"])) if a["box"] else "none"

        prompt = (
            f'USER: <image>\nCaption: "{e["text"]}"\n\n'
            f'A forensic model analyzed this image-caption pair:\n'
            f'- Verdict: {verdict} ({a["p_fake"]:.0%} probability manipulated)\n'
            f'- Manipulation type(s): {ev_types}\n'
            f'- Localized suspicious face region: {box_str}\n'
            f'- Suspicious caption words: {ev_words}\n\n'
            f'Write a 2-3 sentence explanation of this verdict. Strict rules: use ONLY the '
            f'analysis above and what you can directly see; do NOT name or guess any specific '
            f'real people; if suspicious caption words say "NONE flagged", do not claim the '
            f'caption is suspicious; do not invent evidence not listed. ASSISTANT:'
        )
        inp = self.llava_proc(images=a["img"], text=prompt, return_tensors="pt")
        inp = {k: (v.to(dev, torch.float16) if v.dtype == torch.float32 else v.to(dev))
               for k, v in inp.items()}
        with torch.no_grad():
            out = self.llava.generate(**inp, max_new_tokens=110, do_sample=False)
        text = self.llava_proc.decode(out[0], skip_special_tokens=True)
        return a, text.split("ASSISTANT:")[-1].strip()