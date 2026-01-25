# Deep Research: Active Learning and Uncertainty Estimation for Human Preference Prediction in Reward Models

## Research Context & Background

I am conducting research on **evaluating epistemic uncertainty prediction strategies of human preference predictors in text-to-image generation through the lens of active learning**. The goal is to identify research gaps and positioning opportunities for a TMLR (Transactions on Machine Learning Research) paper submission.

### Project Setup
- **Base model**: CLIP ViT-B/32 fine-tuned with a linear classification head for pairwise preference prediction
- **Dataset**: Pick-a-Pic v2 (~580K preference pairs from real users comparing text-to-image outputs)
- **Uncertainty estimation**: MC Dropout (dropout rate 0.3, 30 forward passes) applied during inference for acquisition scoring
- **Active learning strategies tested**: Random Sampling, BALD (Bayesian Active Learning by Disagreement), Predictive Entropy, Least Confidence, Coreset K-Center
- **Protocol**: Start with 100 labeled pairs, acquire 100 new pairs per round, up to ~1,100 total training samples
- **Training**: AdamW optimizer, weight_decay=0.1, 10 epochs per AL round, cosine LR schedule

### Key Findings
1. **Uncertainty-based acquisition functions (BALD, Entropy, Least Confidence) fail to outperform random sampling**, suggesting that MC Dropout uncertainty estimates may not capture meaningful epistemic uncertainty in this setting
2. Best accuracy achieved: **~66.34%** with ~1,100 samples (random sampling)
3. Reference baseline: PickScore achieves **71.1%** trained on the full Pick-a-Pic dataset
4. The failure of uncertainty-guided selection raises questions about **whether current uncertainty estimation methods are well-suited for preference/reward models**
5. Coreset K-Center (a diversity-based, non-uncertainty method) performs comparably to random, suggesting the issue is not limited to uncertainty methods alone
6. The performance gap between our sample-efficient model and full-data PickScore is ~5 percentage points

---

## Broad Survey Questions

Please conduct a comprehensive literature survey covering the following five areas. For each area, identify the most relevant papers (prioritizing 2022-2025), summarize key findings, and note methodological details.

### 1. Epistemic Uncertainty Estimation in Reward/Preference Models

- How effective is **MC Dropout** for epistemic uncertainty estimation in fine-tuned vision-language models (CLIP, BLIP, etc.)?
- What are the **known limitations** of MC Dropout as an uncertainty estimator, particularly for:
  - Models with batch normalization or layer normalization
  - Pre-trained models with dropout only in certain layers
  - Low-dimensional output spaces (binary classification)
  - Pairwise comparison / preference prediction tasks
- How do alternative uncertainty methods compare for reward models: **deep ensembles**, **evidential deep learning**, **spectral-normalized neural GPs**, **last-layer Laplace approximation**?
- Is there work on **uncertainty calibration** specifically for preference models or pairwise comparison tasks?
- What does well-calibrated uncertainty look like in the context of **subjective human preferences** where ground truth is inherently noisy?
- How is **epistemic vs. aleatoric uncertainty** distinguished in preference settings, where label noise is high?

### 2. Active Learning as an Evaluation Framework for Uncertainty Quality

- How has active learning been used to **evaluate the quality of uncertainty estimates** (not just as a data-selection method)?
- What existing work applies active learning to **pairwise preference learning** or **reward model training**?
- How has AL been used in **RLHF (Reinforcement Learning from Human Feedback)** pipelines to reduce annotation costs?
- Are there AL methods specifically designed for **Bradley-Terry models** or other preference frameworks?
- What acquisition functions have been proposed for preference data beyond standard uncertainty sampling?
- How do AL methods perform when the underlying model uses **contrastive learning** (e.g., CLIP-based architectures)?
- What does **AL failure** (uncertainty-guided selection not beating random) tell us about the underlying uncertainty estimates?

### 3. Human Preference Prediction for Generative AI

- What is the landscape of **learned preference/reward models** for text-to-image generation? Cover:
  - PickScore (Pick-a-Pic)
  - ImageReward
  - HPS (Human Preference Score) v1 and v2
  - CLIP-based reward models
  - Any other preference predictors for diffusion models
- What **preference datasets** exist beyond Pick-a-Pic? (e.g., ImageRewardDB, HPS datasets, DiffusionDB ratings)
- How do these models handle the **subjectivity and noise** inherent in human preference judgments?
- What are the **data efficiency characteristics** of these models? How much data is needed for reliable preference prediction?
- Are there approaches that achieve strong preference prediction with **limited labeled data**?
- Do any of these models provide or evaluate **uncertainty estimates** alongside their predictions?

### 4. Data Efficiency and Selection in Reward Model Training

- What methods exist for **data pruning** or **data selection** in reward model or preference model training?
- How do **curriculum learning** approaches apply to preference data?
- What are the **scaling laws** for preference model performance as a function of dataset size?
- Are there **coreset construction methods** tailored for pairwise comparison data?
- How do **semi-supervised** or **self-training** approaches reduce labeling needs for preference prediction?
- What role does **data quality** vs **data quantity** play in preference model accuracy?
- Can uncertainty estimates guide **data selection** even when they fail to guide active acquisition?

### 5. Uncertainty-Based Active Learning Failure Modes

- What are **documented cases** where uncertainty-guided sampling fails to outperform random selection?
- What **theoretical explanations** exist for uncertainty-based AL failure? Consider:
  - Poor uncertainty calibration leading to uninformative acquisition
  - Sampling bias and distribution shift induced by uncertainty sampling
  - Cold-start problems with unreliable early uncertainty estimates
  - Batch mode AL pathologies (redundancy in uncertainty-selected batches)
  - Mismatch between epistemic uncertainty and sample informativeness
- Is there work on **diagnosing whether uncertainty estimates are actionable** for a given task?
- Is there work on **when active learning helps vs. hurts** as a function of:
  - Dataset characteristics (noise level, class balance, feature dimensionality)
  - Model capacity and architecture
  - Uncertainty estimation method quality
  - Task difficulty and label noise
- Are there **adaptive strategies** that detect poor uncertainty quality and adjust acquisition accordingly?

---

## Gap Identification Questions

Based on your survey, please address the following:

1. **What are the open problems** in uncertainty estimation for preference/reward models?
2. **Where does existing work fall short?** Are there assumptions about uncertainty quality that don't hold for preference models in practice?
3. **What novel contributions could be made** given our finding that uncertainty-guided acquisition fails? Consider:
   - Diagnostic frameworks for evaluating uncertainty quality in reward models
   - Alternative uncertainty estimation methods better suited to preference prediction
   - Theoretical analysis of why standard uncertainty methods fail in noisy preference settings
   - Connections between aleatoric noise in preferences and epistemic uncertainty estimation failure
   - New metrics for uncertainty quality beyond active learning performance
4. **Are there unexplored method combinations?** For example:
   - Combining epistemic uncertainty with preference-specific noise models
   - Disentangling aleatoric and epistemic uncertainty in preference data
   - Using LLM-generated pseudo-preferences to validate uncertainty estimates
   - Uncertainty-aware reward model architectures
   - Transfer of uncertainty calibration across preference domains
5. **What would constitute a TMLR-level contribution** in this space? What is the bar for novelty and significance?

---

## Output Format Requirements

Please structure your response as follows:

### A. Literature Summary Table
For each relevant paper, provide:
- **Title** | **Authors** | **Year** | **Venue**
- **Key finding** (1-2 sentences)
- **Relevance to our work** (1 sentence)

### B. Identified Research Gaps
Organize gaps by theme:
- Gap description
- Why it matters
- How our work could address it
- Feasibility assessment (low/medium/high effort; low/medium/high novelty)

### C. Suggested Research Directions
For each direction:
- **Direction title**
- **Description** (2-3 sentences)
- **Required experiments** (bullet list)
- **Expected contribution type** (empirical, theoretical, methodological)
- **Feasibility for TMLR** (with justification)

### D. Positioning Recommendations
- Top 3 angles for framing the paper around uncertainty estimation in reward models
- Which gaps are most tractable given our existing experimental setup (MC Dropout, AL evaluation, CLIP-based preference model)
- What additional experiments would strengthen the contribution (e.g., alternative uncertainty methods, calibration analysis, ablations)
- Suggested paper title options

---

## Additional Notes

- Prioritize papers from **top ML venues**: NeurIPS, ICML, ICLR, TMLR, CVPR, ECCV, AAAI, and relevant workshops
- Include **preprints from 2024-2025** that may not yet be published
- Consider both the **computer vision** and **NLP/RLHF** communities, as uncertainty estimation for preference/reward models spans both
- Flag any **concurrent work** that overlaps significantly with our approach (uncertainty evaluation in reward models, AL for preference learning)
- If specific papers are hard to find, note the gap—absence of work in an area is itself a valuable signal
