# SC4001 Flowers 102 Report Design

## Context

Final project report for SC4001: Neural Networks and Deep Learning. Ten-page PDF (Arial 10 font equivalent — LaTeX default serif is acceptable), covering our work on Oxford Flowers 102 fine-grained classification.

**Team:** Jonas Tua, Jia Ying Thor, F Syed Abu Thahir
**Deadline:** 19 April 2026

**Assessment weights:** execution (30%), experiments (30%), report presentation (15%), novelty (15%), peer review (10%).

## Goal

Produce a LaTeX report in the style of the reference `NLarge: Data augmentation for NLP` submission: academic-technical tone, `\we` voice, numbered sections with inline results, figure grids, honest discussion. First draft produced by Claude using the `humanizer` skill, then edited by the team.

## Thesis

> For fine-grained classification with limited training data, parameter efficiency beats architectural modification. We demonstrate this by systematically exploring CNN modifications (which hurt performance), regularization techniques (which help modestly), and Visual Prompt Tuning (which trains <0.2% of parameters yet achieves the best result).

## Report Structure (hypothesis-driven, matches reference)

```
Title page: NTU logo, "Advanced Few-Shot Flower Recognition", team members + IDs, date
Abstract (~150 words)

1 Introduction
  1.1 Literature Review
  1.2 Objective
  1.3 Training and Evaluation
    1.3.1 Dataset
    1.3.2 Metrics (mean per-class accuracy as primary, following Nilsback & Zisserman)
    1.3.3 Experimental Setup

2 Baseline: ResNet18 Transfer Learning
  2.1 Architecture
  2.2 Results and Analysis

3 Advanced Convolution Techniques
  3.1 Dilated Convolution
  3.2 Deformable Convolution
  3.3 Depthwise-Separable Convolution
  3.4 Novel Hybrid: Dilated + Deformable
  3.5 Results and Analysis

4 Regularization Techniques
  4.1 Freezing Early Layers
  4.2 Label Smoothing
  4.3 MixUp Augmentation
  4.4 Stronger Data Augmentation
  4.5 Results and Analysis

5 Scaling the Backbone: ResNet50
  5.1 Results and Analysis

6 Visual Prompt Tuning
  6.1 VPT-Shallow
  6.2 VPT-Deep
  6.3 Results and Analysis

7 Few-Shot Learning Analysis
  7.1 Experimental Setup
  7.2 Results and Analysis

8 Discussion
9 Conclusion
References
```

## Page Budget

| Section | Pages |
|---|---|
| Abstract + §1 Introduction | 1.5 |
| §2 Baseline | 0.5 |
| §3 Advanced Convolutions | 1.5 |
| §4 Regularization | 1.5 |
| §5 ResNet50 | 0.5 |
| §6 VPT | 1.5 |
| §7 Few-shot | 1.0 |
| §8 Discussion | 1.0 |
| §9 Conclusion | 0.5 |
| **Total content** | **9.5** |
| References (not counted) | 1.0 |

## Experiments Still Needed

### E1: Re-evaluate VPT with proper metrics

- Load VPT-Shallow and VPT-Deep checkpoints
- Compute accuracy, mean per-class accuracy, top-5 accuracy, per-class breakdown
- Save to `results/logs/vpt_shallow_classification_report.txt` and `vpt_deep_classification_report.txt`

### E2: Few-shot VPT (k=1,2,5)

- Train VPT-Shallow and VPT-Deep on k-shot subsets for k=1,2,5
- 30 epochs each (same as CNN few-shot experiments)
- Record best validation accuracy (MPC accuracy on val set where possible)
- 6 runs total (~30 min on RTX 3070 Ti)

### E3: Few-shot for frozen baseline (k=1,2,5)

- Train `frozen` variant on k=1,2,5
- Needed because frozen is the best CNN — incomplete few-shot picture without it
- 3 runs (~10 min)

### E4: Prompt length ablation for VPT-Deep

- Train VPT-Deep with num_prompts ∈ {1, 5, 10, 50}
- Full training set, 50 epochs each
- Shows optimal prompt length; matches original VPT paper's ablation (Fig 6)
- 4 runs (~40 min)

## Key Figures and Tables

### Table 1: Method comparison (central table)

Columns: Method | Test Acc | MPC Acc | Top-5 Acc | Trainable Params | % of Backbone

All 13 methods + VPT-Shallow + VPT-Deep, sorted by MPC Acc. Bold the best.

### Table 2: Few-shot results

Columns: Method | k=1 | k=2 | k=5 | k=10 (full)
Rows: Baseline, Frozen, Deformable, VPT-Shallow, VPT-Deep

### Figure 1: Parameter efficiency scatter

X-axis: trainable parameters (log scale). Y-axis: mean per-class accuracy.
VPT-Deep and VPT-Shallow appear in the top-left (best + fewest params), visually proving the thesis.

### Figure 2: Few-shot degradation curves

X-axis: k (1, 2, 5, 10). Y-axis: accuracy. Line per method.

### Figure 3: Training curves comparison

Validation accuracy over epochs, one line per method. Use existing CSV logs.

### Figure 4: Confusion matrix for VPT-Deep

Already produced in VPT evaluation notebook. 102×102, normalized.

### Figure 5: Prompt length ablation

From E4. X-axis: num_prompts (log). Y-axis: test accuracy.

### Equations to include

- VPT-Shallow prompt injection: `[x_0, P, E_0] → L_1`
- VPT-Deep per-layer injection: `[x_i, _, E_i] = L_i([x_{i-1}, P_{i-1}, E_{i-1}])`
- MixUp: `x̃ = λx_i + (1-λ)x_j` where `λ ~ Beta(α, α)`
- Dilated conv receptive field: `RF = k + (k-1)(d-1)`

## Novelty Claims (for 15% weight)

1. **Systematic comparison** of three adaptation paradigms (architectural mods, regularization, parameter-efficient tuning) on a fine-grained few-shot task — this breadth is rare in the VPT literature
2. **Novel hybrid CNN** combining dilated convolutions in layer3 with deformable convolutions in layer4
3. **Honest negative results** — MixUp hurts with 10 shots/class, dilated conv alone hurts performance — discussed candidly in §3.5, §4.3, §8
4. **Few-shot parameter efficiency analysis** — showing that the gap between VPT and full fine-tuning widens as data shrinks (using k=1,2,5,10 with the same seed)

## Writing Style

- **Tone:** "We" voice, active, present tense for method descriptions, past tense for experiments
- **Structure per section:** brief motivation → formal description (with equations where relevant) → results table/figure → analysis paragraph
- **Honest language:** when something didn't work, say so and explain why (reference report does this with its LLM DA section)
- **Citations:** numbered, following the reference's style
- **Humanizer:** apply the `humanizer` skill to the draft to remove AI-sounding patterns (em dash overuse, rule of three, inflated symbolism)

## Citations (to include)

1. Nilsback & Zisserman, "Automated flower classification over a large number of classes," ICVGIP 2008 — dataset and primary metric
2. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016 — ResNet
3. Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021 — ViT
4. Jia et al., "Visual Prompt Tuning," ECCV 2022 — VPT
5. Zhang et al., "mixup: Beyond Empirical Risk Minimization," ICLR 2018 — MixUp
6. Dai et al., "Deformable Convolutional Networks," ICCV 2017 — Deformable conv
7. Yu & Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions," ICLR 2016 — Dilated conv
8. Howard et al., "MobileNets," 2017 — depthwise-separable
9. Szegedy et al., "Rethinking the Inception Architecture," CVPR 2016 — label smoothing
10. Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database," CVPR 2009 — pretraining data

## Deliverables

1. `report/main.tex` — LaTeX source, root file
2. `report/sections/*.tex` — one file per section for parallel editing
3. `report/bibliography.bib` — BibTeX references
4. `report/figures/` — all figures referenced in the report (copied from `results/logs/` or regenerated)
5. `report/README.md` — compilation instructions (Overleaf or local `latexmk`)

## Out of Scope

- Triplet loss ablation (not enough time; note as future work)
- Test-time augmentation (not enough time; note as future work)
- Self-supervised pre-training experiments (out of scope)
- Comparison with recent 2024/2025 methods (out of scope; acknowledge in discussion)

## Success Criteria

1. Report fits in 10 content pages
2. Every experiment has a table or figure
3. Discussion includes at least one negative finding (MixUp failure) and one counter-intuitive insight (modifying convs hurts)
4. Parameter efficiency scatter plot is the central visual argument
5. LaTeX compiles cleanly with `latexmk -pdf` (no errors)
6. Team members can edit individual section files without merge conflicts
