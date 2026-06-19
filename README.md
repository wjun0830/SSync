<h1 align="center">SSync</h1>
<h3 align="center">Selective Synergistic Learning for Video Object-Centric Learning</h3>

<p align="center">
  <a href="https://github.com/wjun0830">WonJun Moon</a><sup>1</sup> &nbsp;·&nbsp;
  Jae-Pil Heo<sup>2</sup>
  <br>
  <sup>1</sup>KAIST &nbsp;&nbsp; <sup>2</sup>Sungkyunkwan University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.15527v1"><img src="https://img.shields.io/badge/Paper-ECCV%202026-blue" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2606.15527v1"><img src="https://img.shields.io/badge/arXiv-2606.15527-b31b1b.svg" alt="arXiv"></a>
  <a href="."><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://huggingface.co/WJ0830/SSync"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
  <a href="https://wjun0830.github.io/SSync"><img src="https://img.shields.io/badge/Project-Page-brightgreen" alt="Project Page"></a>
</p>

> Official PyTorch implementation of **SSync (Selective Synergistic Learning)**, a selective mutual-distillation framework for video object-centric learning.

---

## 📖 Abstract

Typical video object-centric learning (VOCL) approaches employ slot-based frameworks that rely on reconstruction-driven encoder–decoder architectures, where learning is mediated by two spatial maps: attention maps from the encoder and object maps from the decoder. As these two distinct maps exhibit different properties, a recent dense alignment strategy attempted to reconcile this discrepancy by enforcing agreement across all spatio-temporal patches via contrastive learning. However, this indiscriminate alignment inadvertently propagates the inherent weaknesses of each module, such as noisy encoder predictions and blurred decoder boundaries. Moreover, computing dense similarities across all pairs incurs a computational cost quadratic in the total number of spatio-temporal patches, severely limiting scalability.

Motivated by this, we propose **Selective Synergistic Learning (SSync)**. Instead of exhaustive patch-to-patch alignment, SSync prevents error propagation by selectively distilling only the most reliable cues: leveraging the encoder strictly for boundary refinement and the decoder for interior denoising. This is realized via pseudo-labeling with linear complexity, eliminating the need for quadratic spatial comparisons. Also, to prevent the reinforcement of architectural biases like slot redundancy, we introduce a transitive pseudo-label merging that consolidates overlapping slots based on spatio-temporal activation consistency. Extensive studies demonstrate that SSync improves decomposition quality and serves as a versatile, plug-and-play module while also exhibiting exceptional robustness to slot configurations.

![SSync selective supervision (Figure 2)](https://wjun0830.github.io/SSync/static/images/fig2.png)

---

## ⚙️ Installation

```bash
pip install poetry
poetry lock
poetry install
poetry run pip install matplotlib coco notebook
poetry run pip install tensorboard tensorboardX
```

---

## 🗂️ Datasets

To download the datasets used in this work, see the instructions in [`data/README.md`](data/README.md).
For more details, we refer to the [SlotContrast](https://github.com/martius-lab/slotcontrast/tree/main) repository.

The datasets should be placed under a common root directory with the following structure:

```
├── SlotCurri/
└── dataset/
    ├── ytvis2021_resized/
    ├── movi_c/
    └── movi_e/
```

| Dataset | Download | Size |
|---|---|---|
| YouTube-VIS 2021 | [Google Drive](https://drive.google.com/file/d/1Iv-2zK6MnH0oDFTx9iBgQcPNh5PBzM7i/view?usp=sharing) | 26.43 GB |
| MOVi-C | [Google Drive](https://drive.google.com/file/d/1CvHkK0PhqHrC8MtMtFCXBEXZbyrC6-hH/view?usp=sharing) | 7.43 GB |
| MOVi-E | [Google Drive](https://drive.google.com/file/d/1qGXzMwEMbYRp7OH2GJlkS3lvZPv3grkt/view?usp=sharing) | 8.26 GB |

---

## 🚀 Training

Run one of the configurations in [`configs/SSync`](configs/SSync), for example:

```bash
poetry run python -m SSync.train --run-eval-after-training configs/SSync/coco.yaml
poetry run python -m SSync.train --run-eval-after-training configs/SSync/movi_c.yaml
poetry run python -m SSync.train --run-eval-after-training configs/SSync/movi_e.yaml
poetry run python -m SSync.train --run-eval-after-training configs/SSync/ytvis2021.yaml
```

---

## 🧪 Pretrained Checkpoints

| Dataset | Download |
|---|---|
| MOVi-C | [Google Drive](https://drive.google.com/file/d/1cfuGqAeCjF1GHhzRdtW2kRJZgnQ5BWD_/view?usp=sharing) |
| MOVi-E | [Google Drive](https://drive.google.com/file/d/1aiBHCI8ypU5o-UTGAFT-cE9foTm9B0Zj/view?usp=sharing) |
| YouTube-VIS 2021 | [Google Drive](https://drive.google.com/file/d/1FoILMi6yfUbXAcF5Ctt-9i8TABJThJib/view?usp=sharing) |
| COCO 2017 | [Google Drive](https://drive.google.com/file/d/1iIgavtLRZNWGMDycYKH8B9wz5Y-tH_cx/view?usp=sharing) |

---

## 🖼️ Qualitative Results

Each clip blends the original input with the predicted slot map (each object in a distinct color). For interactive controls, visit the [project page](https://wjun0830.github.io/SSync).

<table>
  <tr>
    <th align="center">MOVi-C</th>
    <th align="center">MOVi-E</th>
    <th align="center">YouTube-VIS 2021</th>
  </tr>
  <tr>
    <td align="center"><video src="https://wjun0830.github.io/SSync/static/results/movi_c/000_mixed.mp4" autoplay loop muted width="240"></video></td>
    <td align="center"><video src="https://wjun0830.github.io/SSync/static/results/movi_e/017_mixed.mp4" autoplay loop muted width="240"></video></td>
    <td align="center"><video src="https://wjun0830.github.io/SSync/static/results/ytvis2021/016_mixed.mp4" autoplay loop muted width="240"></video></td>
  </tr>
</table>

---

## 📌 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{moon2026ssync,
  title     = {Selective Synergistic Learning for Video Object-Centric Learning},
  author    = {Moon, WonJun and Heo, Jae-Pil},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

as well as SlotCurri and SRL:

```bibtex
@inproceedings{moon2026reconstruction,
  title     = {Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning},
  author    = {Moon, WonJun and Seong, Hyun Seok and Heo, Jae-Pil},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}

@inproceedings{seong2026synergistic,
  title     = {From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning},
  author    = {Seong, Hyun Seok and Moon, WonJun and Heo, Jae-Pil},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgement

Our implementation is built upon the official repositories of
[VideoSAUR](https://github.com/martius-lab/videosaur),
[SlotContrast](https://github.com/martius-lab/slotcontrast/tree/main),
[SRL](https://github.com/hynnsk/SRL/tree/main), and
[SlotCurri](https://github.com/wjun0830/SlotCurri.git).

---

## 📄 License

This codebase is released under the [MIT License](LICENSE).
Some parts of the codebase were adapted from other codebases; a comment was added to the code where this is the case, and those parts are governed by their respective licenses.
