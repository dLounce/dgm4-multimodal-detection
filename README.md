## Setup

```bash
git clone https://github.com/USERNAME/dgm4-multimodal-detection.git
cd dgm4-multimodal-detection
pip install -r requirements.txt   # install torch matching your CUDA: https://pytorch.org
```

**Dataset.** DGM4 is built on VisualNews; download it from the [official repository](https://github.com/rshaojimmy/MultiModal-DeepFake) or the [Kaggle mirror](https://www.kaggle.com/datasets/shubhamkumar9812/dgm4-dataset), then point the code at it:

```bash
export DGM4_ROOT=/path/to/dgm4-dataset
```

**Model weights.** The code loads CLIP and LLaVA from local directories (`local_files_only=True`). Fetch them once, e.g.:

```python
from huggingface_hub import snapshot_download
snapshot_download("openai/clip-vit-large-patch14", local_dir="models/clip-vit-large-patch14")
snapshot_download("llava-hf/llava-1.5-7b-hf",       local_dir="models/llava-1.5-7b-hf")
```

## Usage

Build the detector and run a forward pass:

```python
import torch
from data import DGM4, make_collate
from detector import build_detector
from transformers import CLIPProcessor

proc = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14")
model = build_detector("models/clip-vit-large-patch14").cuda()

batch = make_collate(proc)([DGM4(split="test")[i] for i in range(8)])
with torch.autocast("cuda", dtype=torch.bfloat16):
    binary, types, *_ = model(batch["pixel_values"].cuda(),
                              batch["input_ids"].cuda(),
                              batch["attention_mask"].cuda())
```

Score predictions with the DGM4 metrics:

```python
from metrics import binary_metrics, multilabel_metrics
print(binary_metrics(y_true, y_score))
print(multilabel_metrics(y_true_multi, y_score_multi))
```

Generate an explanation:

```python
from explain import Explainer, load_llava
llava, llava_proc = load_llava("models/llava-1.5-7b-hf")
expl = Explainer(model, proc, box_head, token_head, llava, llava_proc)
analysis, text = expl.explain(test_entry)
print(text)
```

## Results

Evaluated on the DGM4 test split, following the protocol from Shao et al.

| Task | Metric | Score |
|------|--------|-------|
| Detection (real/fake) | AUC / EER / ACC | — / — / — |
| Manipulation type | mAP / CF1 / OF1 | — / — / — |
| Image grounding (box) | mean IoU / IoU@0.5 | — / — |
| Text grounding (tokens) | precision / recall / F1 | — / — / — |

## Acknowledgements

This work uses the **DGM4** dataset and builds on the problem formulation introduced by Shao, Wu and Liu. It also relies on OpenAI's CLIP, LLaVA-1.5, and the PEFT library.

```bibtex
@inproceedings{shao2023dgm4,
  title     = {Detecting and Grounding Multi-Modal Media Manipulation},
  author    = {Shao, Rui and Wu, Tianxing and Liu, Ziwei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}

@article{shao2024dgm4beyond,
  title   = {Detecting and Grounding Multi-Modal Media Manipulation and Beyond},
  author  = {Shao, Rui and Wu, Tianxing and Liu, Ziwei},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2024}
}
```

## License

Code released under the MIT License. The DGM4 dataset is subject to its own terms — see the [original repository](https://github.com/rshaojimmy/MultiModal-DeepFake) before use.