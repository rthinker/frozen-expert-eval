# Licensing and attribution

| Component | Terms and source |
|---|---|
| Code offered with this release | See the [release licence and copyright notice](../LICENSE). |
| Derived feature cache and repository artifacts | CC BY-NC-SA 4.0 for our contributions. The cache contains features, logits, labels, sample identifiers and split provenance. Repository artifacts include sample IDs, predictions, exclusion lists, duplicate counts and fold manifests. Upstream terms apply. See the [dataset notice](https://huggingface.co/datasets/anon-frozen-experts/df40-frozen-experts-anon/blob/main/LICENSE_NOTICE.txt). |
| DF40 and DeepfakeBench | CC BY-NC 4.0 as published by [DF40](https://github.com/YZY-stack/DF40) and [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). Each underlying dataset has its own terms. |
| CLIP detector backbones | Obtained through DF40. See the upstream [licence](https://github.com/openai/CLIP/blob/main/LICENSE) and [model card](https://github.com/openai/CLIP/blob/main/model-card.md). |
| Xception detector backbones | Obtained through [DF40](https://github.com/YZY-stack/DF40). `scripts/00_setup.py` also downloads the ImageNet Xception file that DF40's detector constructor expects, from Cadene's [pretrained-models](https://github.com/Cadene/pretrained-models.pytorch) (BSD-3-Clause). DF40's checkpoints then replace all of its weights. |
| Python dependencies | Each dependency is installed from its own distribution and follows its published terms. The [lock file](../requirements-lock.txt) and [project metadata](../pyproject.toml) identify the dependencies. |
