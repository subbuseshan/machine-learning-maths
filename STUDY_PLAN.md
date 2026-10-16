# Senior ML Engineer: Study Plan (Oct 2026 → Apply April 2027)

**Timeline:** 24 weeks, Mon 19 Oct 2026 → Sun 4 Apr 2027. Applications start the week of 5 Apr 2027.
**Budget assumed:** ~12–15 hrs/week (≈ 2 hrs on weekdays + 4–5 hrs on weekends). Scale phase lengths if yours differs.
**Companion:** [TODO.md](TODO.md) holds the maths gaps (Phase 0).

---

## 0. What "Senior MLE" means in 2026 postings

Patterns from current postings and interview guides (sources at the bottom):

| Signal | What they ask for | What it means for your prep |
|---|---|---|
| **Production ownership** | "Deployed models serving live traffic", training → deployment → monitoring → feedback loops | Theory isn't enough. You need **2 end-to-end deployed projects** you can defend in depth |
| **LLMs + RAG** | Now in most postings: LLMs, RAG, embeddings, vector DBs, agentic frameworks, LLM evaluation | A full phase below; GenAI system design is now a standard round |
| **Classical ML still matters** | "LLMs *and* traditional ML models", PyTorch / scikit-learn | Tabular, ranking and fraud problems still run on GBDTs, so expect breadth questions |
| **MLOps / infra** | Docker, Kubernetes, GPU infra, Kubeflow / Airflow / MLflow, CI/CD, model versioning, drift detection | Know the concepts and have used one tool per category. Feature stores are rarely required (nice-to-have) |
| **Cloud** | AWS / GCP / Azure | Go deep on **one** (AWS SageMaker or GCP Vertex AI) |
| **Seniority** | Lead technical design, set standards, mentor, explain ML to non-specialists | System design + behavioural rounds decide your level |

**Typical interview loop (5 axes):** ① coding (LeetCode-medium + applied/ML coding) ② ML breadth & depth ③ **ML / GenAI system design** (often the heaviest round at senior level) ④ production ML deep-dive on your past projects ⑤ behavioural / leadership.

---

## 1. Schedule at a glance

| Phase | Weeks | Dates | Focus | Deliverable |
|---|---|---|---|---|
| 0 | 1–2 | 19 Oct – 1 Nov | Maths gaps | TODO.md items 1, 2, 4, 6 done |
| 1 | 3–7 | 2 Nov – 6 Dec | Classical ML | From-scratch implementations + **Project A** model trained |
| 2 | 8–12 | 7 Dec – 10 Jan | Deep learning (weeks 10–11 = holiday, light) | Transformer from scratch; CNN fine-tune |
| 3 | 13–17 | 11 Jan – 14 Feb | LLMs, embeddings, RAG, agents | **Project B** (RAG system + eval harness) |
| 4 | 18–21 | 15 Feb – 14 Mar | MLOps, serving, monitoring, cloud | Project A & B deployed with CI/CD + monitoring |
| 5 | 22–24 | 15 Mar – 4 Apr | Interview prep: system design, mocks, résumé | Résumé, 8 STAR stories, 10 design cases rehearsed |

**Every week, all phases:**
- [ ] 3 LeetCode mediums (arrays/hashing, two pointers, heaps, BFS/DFS, intervals, binary search). Target ~120 by April.
- [ ] From week 13: 1 system design case per week (list in §7).
- [ ] 30 min/week: write up what you learned, which doubles as interview talking points.

---

## 2. Phase 0: Maths gaps (Weeks 1–2)

Work through [TODO.md](TODO.md) items **1 (information theory), 2 (matrix calculus/backprop), 4 (optimizers), 6 (numerical stability)**. Fold items 3, 5 and 7 into the later phases where they come up (SVD → PCA in Phase 1, norms → embeddings in Phase 3).

Ready when you can derive the gradient of softmax + cross-entropy on a whiteboard and explain why it simplifies to `ŷ − y`.

---

## 3. Phase 1: Classical ML (Weeks 3–7)

### 3.1 Fundamentals: know cold
- [ ] Bias–variance trade-off; under- vs overfitting; learning curves
- [ ] Train / val / test splits; k-fold, stratified, **time-series** and **group** CV
- [ ] **Data leakage**: target leakage, temporal leakage, preprocessing before the split
- [ ] Regularization: L1 (sparsity) vs L2 (shrinkage), elastic net, early stopping
- [ ] Feature engineering: scaling, encoding (one-hot, target, hashing), missing values, interactions, binning
- [ ] Class imbalance: resampling, class weights, focal loss, threshold tuning, and why accuracy is misleading
- [ ] Hyperparameter search: grid vs random vs Bayesian (Optuna)
- [ ] Probability calibration (Platt, isotonic) and why it matters for downstream decisions
- [ ] Curse of dimensionality; PCA via SVD (TODO item 3)

### 3.2 Algorithms

For each one, be able to explain: **intuition → objective/loss → how it's trained → key hyperparameters → assumptions & failure modes → complexity → when you'd pick it**.

| Algorithm | Must be able to explain | Common interview probes |
|---|---|---|
| **Linear regression** | OLS closed form vs GD; assumptions (linearity, homoscedasticity, independence); Ridge / Lasso | Multicollinearity? Why does Lasso give sparsity (geometry of the L1 ball)? |
| **Logistic regression** | Sigmoid, log-loss from MLE, decision boundary, odds interpretation | Why not MSE for classification? Multiclass via softmax / OvR |
| **k-NN** | Distance metrics, choice of k, scaling | Why it breaks in high dimensions; inference cost → ANN (links to vector DBs) |
| **Naive Bayes** | Conditional independence, Laplace smoothing | Why it works for text despite a wrong assumption (you built this in C3!) |
| **SVM** | Max margin, hinge loss, soft margin C, kernel trick (RBF), support vectors | Primal vs dual; why kernels avoid explicit feature maps; scaling to large n |
| **Decision trees** | Gini vs entropy / information gain, variance reduction, pruning | Why are trees high-variance? Handling categorical features |
| **Random forest** | Bagging + feature subsampling, OOB error, feature importance | Why decorrelating trees reduces variance; impurity- vs permutation-importance bias |
| **Gradient boosting / XGBoost / LightGBM / CatBoost** | Fitting residuals = gradient in function space; shrinkage; XGBoost's 2nd-order (Hessian) objective + regularization; LightGBM histograms & leaf-wise growth; CatBoost ordered target stats | **Bagging vs boosting**; key hyperparams (learning_rate, n_estimators, max_depth, subsample, colsample); why GBDTs still beat DL on tabular data |
| **k-means** | Lloyd's algorithm, k-means++, elbow / silhouette | Non-convex clusters, sensitivity to scale and initialization |
| **DBSCAN / GMM** | Density-based clustering; EM for GMMs (soft k-means) | When to use each vs k-means |
| **PCA / t-SNE / UMAP** | Variance-maximizing projection via SVD; t-SNE/UMAP for visualization only | Why t-SNE distances aren't meaningful |
| **Anomaly detection** | Isolation forest, one-class SVM, autoencoder reconstruction error | Fraud / monitoring use cases |
| **Recommenders & ranking** | Collaborative filtering, matrix factorization (ALS), two-tower retrieval → ranking, learning-to-rank (pointwise / pairwise / listwise, LambdaMART) | **Very common in senior system design**: cold start, candidate generation vs ranking |
| **Time series** | Stationarity, ARIMA basics, lag/rolling features + GBDT, backtesting | Leakage with random splits |

### 3.3 Cost functions (loss)

| Loss | Used for | Know |
|---|---|---|
| MSE / RMSE | Regression | Gaussian-noise MLE; sensitive to outliers |
| MAE | Regression | Laplace MLE; robust; predicts the median |
| Huber / smooth-L1 | Robust regression, object detection | Quadratic near 0, linear in the tails |
| Quantile (pinball) | Prediction intervals, forecasting | Asymmetric penalty |
| Binary cross-entropy / log-loss | Binary classification | Bernoulli MLE; pairs with sigmoid |
| Categorical cross-entropy | Multiclass, **LLM next-token prediction** | Softmax pairing; label smoothing; perplexity = exp(CE) |
| Hinge | SVM | Margin-based, not probabilistic |
| Focal loss | Heavy imbalance, detection | Down-weights easy examples |
| KL divergence | Distillation, VAEs, RLHF penalty | Asymmetric; CE = H(p) + KL(p‖q) |
| Contrastive / InfoNCE / triplet | **Embeddings**, CLIP, retrieval | In-batch negatives, temperature, hard-negative mining |
| Pairwise ranking (BPR, RankNet), LambdaRank | Search / recsys | Optimizes ordering rather than scores |
| Dice / IoU loss | Segmentation | Overlap-based |
| DPO loss | LLM preference alignment | Preference-pair log-ratio, no reward model |

### 3.4 Evaluation metrics
- [ ] Classification: precision, recall, F1, ROC-AUC vs **PR-AUC** (use under imbalance), confusion matrix, log-loss, calibration curves
- [ ] Regression: RMSE, MAE, MAPE (and its pitfalls), R²
- [ ] Ranking / retrieval: precision@k, **recall@k, MRR, nDCG, MAP**
- [ ] Business / online: A/B tests (C3 W4!), statistical power, guardrail metrics, novelty effects; **offline vs online metric mismatch**

### 3.5 Phase 1 deliverables
- [ ] NumPy from scratch: linear & logistic regression (GD), k-means, decision-tree split finder, k-NN. ML-coding rounds ask for exactly these.
- [ ] **Project A, part 1**: a tabular problem (e.g. churn, fraud or demand forecasting). Proper CV, an XGBoost/LightGBM baseline vs logistic regression, SHAP explanations, MLflow experiment tracking. Put it in a **new repo**.
- 📚 *Hands-On Machine Learning* (Géron, 3rd ed.) Part I; StatQuest videos for trees and boosting; the original XGBoost paper (Chen & Guestrin 2016) sections 2–3.

---

## 4. Phase 2: Deep learning (Weeks 8–12, with weeks 10–11 light over the holidays)

### 4.1 Core
- [ ] MLP, activations (ReLU, GELU, SiLU; why not sigmoid in hidden layers), universal approximation
- [ ] Backprop on computation graphs, autograd (micrograd from TODO item 2)
- [ ] Initialization (Xavier / He), vanishing & exploding gradients, gradient clipping
- [ ] Normalization: **BatchNorm vs LayerNorm vs RMSNorm** (and why transformers use LayerNorm/RMSNorm)
- [ ] Regularization: dropout, weight decay (**AdamW's decoupling**), data augmentation, early stopping
- [ ] Residual connections and why they enable depth

### 4.2 Optimizers: compare on the same loss surface
| Optimizer | Key idea | Know |
|---|---|---|
| (Mini-batch) SGD | Noisy gradient estimate | Batch size ↔ generalization ↔ LR (linear scaling rule) |
| Momentum / Nesterov | Velocity smooths oscillations | Look-ahead gradient |
| AdaGrad → RMSProp | Per-parameter adaptive LR | AdaGrad's decaying LR problem |
| **Adam / AdamW** | Momentum + RMSProp + bias correction; decoupled weight decay | Default for transformers; β₁, β₂, ε; memory cost = 2× params |
| LR schedules | Warmup, cosine decay, step, one-cycle | Why warmup stabilizes early transformer training |
| Newer (awareness) | Lion, Adafactor (memory-light), Muon | Know they exist |
| 2nd order | Newton's method (C2 W3!), L-BFGS | Why the Hessian is impractical for big nets |

### 4.3 Architectures
- [ ] **CNNs**: convolution, padding/stride, receptive field, pooling, parameter sharing; LeNet → VGG → **ResNet** → EfficientNet; transfer learning / fine-tuning; detection (YOLO idea) & segmentation (U-Net) at awareness level
- [ ] **RNN / LSTM / GRU**: gating, why they lost to transformers (sequential compute, long-range dependencies)
- [ ] **Transformers** (most important):
  - [ ] Scaled dot-product attention (why √d), multi-head, causal masking
  - [ ] Positional encodings: sinusoidal → learned → **RoPE**
  - [ ] Encoder-only (BERT) vs decoder-only (GPT) vs encoder-decoder (T5)
  - [ ] Back-of-envelope maths: params ≈ 12·L·d², training FLOPs ≈ 6·N·D, **KV-cache memory** = 2·L·d·seq·batch·bytes
  - [ ] Efficiency: FlashAttention, MQA/GQA, Mixture-of-Experts (awareness)
- [ ] Autoencoders / VAEs; diffusion at intuition level; ViT and CLIP (multimodal embeddings)

### 4.4 Training at scale
- [ ] Mixed precision (fp16 / bf16, loss scaling): ties to TODO item 6
- [ ] Gradient accumulation, gradient checkpointing
- [ ] Data parallel (DDP) vs **FSDP / ZeRO** vs tensor/pipeline parallelism: what each shards
- [ ] Debugging training: overfit one batch first, loss-curve diagnosis, NaNs

### 4.5 Phase 2 deliverables
- [ ] Karpathy *Zero to Hero* → **build nanoGPT / GPT from scratch** (best single investment for transformer interviews)
- [ ] Fine-tune a pretrained ResNet / ViT on a small image dataset in PyTorch (a proper training loop, not just `.fit`)
- 📚 Karpathy *Zero to Hero*; d2l.ai chapters 5–11; *Build a Large Language Model (From Scratch)* (Raschka); "Attention Is All You Need"

---

## 5. Phase 3: LLMs, embeddings, RAG, agents (Weeks 13–17)

### 5.1 LLM fundamentals
- [ ] Tokenization: BPE / SentencePiece, why token counts drive cost and latency
- [ ] Pretraining → **SFT** → preference alignment (**RLHF** with PPO vs **DPO**); reasoning models / test-time compute at awareness level
- [ ] Decoding: greedy, beam, temperature, top-k, top-p
- [ ] Context windows, long-context trade-offs, "lost in the middle"
- [ ] **Fine-tuning vs RAG vs prompting**: the decision framework (a classic senior question)
- [ ] PEFT: **LoRA / QLoRA** (links to SVD/low-rank, TODO item 3)
- [ ] Quantization: int8 / int4, GPTQ, AWQ, GGUF; accuracy vs memory trade-offs
- [ ] Distillation into smaller models

### 5.2 Embeddings & vector search
- [ ] Word2vec → sentence embeddings (SBERT) → modern embedding models; contrastive training
- [ ] **Bi-encoder vs cross-encoder** (retrieve vs rerank)
- [ ] Similarity: cosine vs dot vs L2, normalization (TODO item 7)
- [ ] **ANN indexes**: flat, **HNSW**, IVF, **PQ** (product quantization); recall vs latency vs memory
- [ ] Vector DBs: FAISS (library), **pgvector**, Qdrant, Weaviate, Milvus, Pinecone. Know the trade-offs (managed vs self-hosted, metadata filtering, hybrid support, scale)

### 5.3 RAG (the most common GenAI design question)
- [ ] Ingestion: parsing, **chunking strategy** (size, overlap, structure-aware), metadata
- [ ] Retrieval: dense vs **BM25** vs **hybrid** (reciprocal rank fusion); query rewriting, HyDE, multi-query
- [ ] **Reranking** with cross-encoders (adds tens of ms, big relevance gains)
- [ ] Generation: prompt assembly, citations, handling "no answer"
- [ ] Advanced: GraphRAG, agentic RAG, caching (exact + semantic), freshness / incremental re-indexing, access control
- [ ] **Evaluation is the senior differentiator**: retrieval (recall@k, MRR, nDCG) + generation (faithfulness / groundedness, answer relevance), golden datasets, LLM-as-judge and its biases, regression testing. Tools: RAGAS, DeepEval

### 5.4 Agents & frameworks
- [ ] Tool / function calling, structured outputs (JSON schema), ReAct loop, planning, memory
- [ ] Multi-agent patterns; **MCP** (Model Context Protocol)
- [ ] Frameworks: **LangChain / LangGraph** and LlamaIndex. Learn one, but be able to build RAG **without** a framework. Interviewers probe whether you understand the primitives.
- [ ] Guardrails & safety: prompt injection, PII, jailbreaks, output validation
- [ ] LLM observability: LangSmith / Langfuse; tracing, token cost and latency tracking

### 5.5 Phase 3 deliverables
- [ ] **Project B**: a production-style RAG system over a real corpus (e.g. your own course notes, or arXiv papers): hybrid retrieval + reranker + citations + **an eval harness with a golden set and metrics dashboard**. Write a report with numbers ("hybrid + rerank lifted recall@5 from X to Y").
- [ ] Stretch: QLoRA fine-tune of a small open model (e.g. a ~1–8B model) for a narrow task, compared against prompting/RAG
- 📚 *AI Engineering* (Chip Huyen, 2025); Hugging Face LLM & NLP courses; LangChain/LangGraph docs; Jay Alammar's Illustrated Transformer / Illustrated GPT posts

---

## 6. Phase 4: MLOps, serving, monitoring (Weeks 18–21)

### 6.1 ML lifecycle & tooling (one tool per row)
| Category | Concepts | Tools (pick ⭐) |
|---|---|---|
| Experiment tracking & registry | Reproducibility, lineage, model versioning, stage promotion | ⭐ MLflow, W&B |
| Data versioning & validation | Schema checks, data contracts | DVC, Great Expectations / Pandera |
| Pipelines / orchestration | DAGs, retraining triggers, idempotency | ⭐ Airflow or Prefect; Kubeflow / SageMaker Pipelines (awareness) |
| Feature stores | **Online/offline consistency, training-serving skew, point-in-time joins** | Feast (concepts are enough; rarely required) |
| Distributed data/compute | Partitioning, shuffles | Spark, ⭐ Ray |
| CI/CD for ML | Unit tests on features, model quality gates, automated retrain | ⭐ GitHub Actions |
| Containers & orchestration | Images, resource limits, autoscaling, GPU nodes | ⭐ Docker, ⭐ Kubernetes basics (Deployments, Services, HPA) |
| Cloud | Managed training/endpoints, IAM, cost | ⭐ AWS SageMaker **or** GCP Vertex AI |
| IaC | Reproducible infra | Terraform (awareness) |

### 6.2 Model serving
- [ ] Patterns: **batch vs online (real-time) vs streaming**; sync vs async; edge
- [ ] REST/gRPC API with **FastAPI**; model servers: Triton, TorchServe, BentoML, KServe
- [ ] Optimization: ONNX, TensorRT, quantization, pruning, distillation, dynamic batching, caching
- [ ] **LLM serving**: **vLLM** (PagedAttention, continuous batching), TGI, SGLang; KV cache, speculative decoding, prefix caching; tokens/sec, TTFT, cost per 1K tokens
- [ ] Rollout: **shadow → canary → A/B → full**, blue-green, rollback; multi-armed bandits
- [ ] Latency budgets, p50/p95/p99, autoscaling, GPU vs CPU cost trade-offs

### 6.3 Monitoring
- [ ] **Data drift** (covariate shift): PSI, KS test, KL/JS divergence
- [ ] **Concept drift** (P(y|x) changes) & label delay; prior/label shift
- [ ] **Training-serving skew**
- [ ] Model performance with delayed or proxy labels; slice-based monitoring; fairness
- [ ] System metrics: latency, throughput, errors, GPU utilization (Prometheus + Grafana)
- [ ] LLM-specific: hallucination rate, groundedness, toxicity, cost and token drift, user feedback loops
- [ ] Tools: Evidently, WhyLabs / Arize (awareness)
- [ ] Retraining strategy: scheduled vs triggered; continual learning

### 6.4 Phase 4 deliverables
- [ ] **Project A, part 2**: package the model as a FastAPI service → Docker → deploy (cloud or local k8s via kind/minikube) → CI/CD pipeline → Evidently drift report + Grafana dashboard → scheduled retrain in Airflow/Prefect
- [ ] **Project B, part 2**: deploy the RAG service, add tracing (Langfuse), semantic caching, a cost/latency dashboard, and an eval run in CI
- 📚 *Designing Machine Learning Systems* (Chip Huyen), the most-cited book for this; Made With ML (Goku Mohandas); Full Stack Deep Learning lectures

---

## 7. Phase 5: Interview prep (Weeks 22–24)

### 7.1 ML system design
A framework to rehearse until it's automatic:
**clarify requirements & scale → frame as an ML problem (inputs/outputs/labels) → metrics (offline + online + guardrail) → data & labeling → features → model (baseline first, then iterate) → training pipeline → serving & latency budget → monitoring & feedback loop → trade-offs, failure modes, cost.** Use concrete numbers (QPS, latency, data volume, retraining cadence).

Rehearse each case (practise from week 13 on, ~1 per week):
- [ ] News feed / content recommendation (two-tower retrieval + ranker)
- [ ] Search ranking / query autocomplete
- [ ] Ads click-through-rate (CTR) prediction
- [ ] Fraud / anomaly detection (imbalance, label delay)
- [ ] Spam / harmful-content moderation (multimodal, precision-recall trade-off)
- [ ] ETA / demand forecasting
- [ ] **RAG customer-support chatbot** (10M docs, ~1K QPS)
- [ ] **LLM evaluation pipeline / guardrails platform**
- [ ] **Vector search service at billion scale** (sharding, HNSW vs IVF-PQ, filtering)
- [ ] **Coding / research agent** (tool use, long-running jobs, sandboxing, cost control)

📚 *Machine Learning System Design Interview* (Ali Aminian & Alex Xu); *Generative AI System Design Interview* (same authors)

### 7.2 Coding rounds
- [ ] LeetCode: finish ~120 mediums (NeetCode 150 list) and re-solve the misses
- [ ] ML coding drills (timed, 30–45 min, NumPy only): logistic regression with GD, k-means, **multi-head attention forward pass**, softmax + cross-entropy, k-NN, a decision-tree split, AUC computation, a simple tokenizer / BPE merge step, a mini-batch data loader
- [ ] Applied coding: build a small API / pipeline in 60 minutes (pandas, SQL window functions)

### 7.3 ML breadth rapid-fire (flashcards)
Bias–variance, bagging vs boosting, L1 vs L2, ROC vs PR, handling imbalance, leakage, why Adam, BatchNorm vs LayerNorm, vanishing gradients, why attention scales by √d, KV cache, LoRA, RAG vs fine-tuning, drift types, offline vs online metric mismatch, cold start.

### 7.4 Behavioural & leadership (decides senior vs mid level)
Prepare **8 STAR stories** with metrics:
- [ ] Led a project end-to-end with ambiguous requirements
- [ ] A model that failed in production: what you learned, what you changed
- [ ] A difficult technical trade-off you drove (e.g. simpler model over a fancy one)
- [ ] Disagreement with a stakeholder or PM
- [ ] Mentoring or raising a team's engineering standards
- [ ] Explaining ML to non-technical leadership
- [ ] Prioritizing under deadline pressure
- [ ] Measurable business impact

### 7.5 Application logistics
- [ ] **By 15 Mar**: résumé rewritten around impact ("cut p95 latency 40%", "lifted recall@5 from .62 to .81"); link both project repos with READMEs, architecture diagrams and results tables
- [ ] Update LinkedIn and GitHub (pin Projects A & B and this repo)
- [ ] 2–3 mock interviews (peers, interviewing.io / Exponent)
- [ ] Target list of 20–30 companies in tiers; **apply to the lowest tier first** as warm-up interviews

---

## 8. Phase checkpoints

| Date | Checkpoint |
|---|---|
| 1 Nov 2026 | Can derive cross-entropy gradient; TODO.md items 1, 2, 4, 6 ✅ |
| 6 Dec 2026 | 5 from-scratch implementations; Project A model + MLflow ✅ |
| 10 Jan 2027 | GPT-from-scratch trained; CNN fine-tune ✅ |
| 14 Feb 2027 | Project B RAG + eval numbers ✅ |
| 14 Mar 2027 | Both projects deployed with CI/CD + monitoring ✅ |
| 4 Apr 2027 | 10 design cases, 8 STAR stories, ~120 LeetCode, résumé final ✅ → **start applying** |

---

## Sources (role & interview research, Oct 2026)
- [GitLab ML Engineering job family](https://handbook.gitlab.com/job-description-library/engineering/development/data-science/machine-learning/)
- [Senior ML Engineer: Virtual Vocations](https://www.virtualvocations.com/job/senior-machine-learning-engineer-3293413-i.html)
- [Senior ML Engineer: Pantheon Data](https://pantheondata.isolvedhire.com/jobsandemployment/common-job-titles/Senior-Machine-Learning-Engineer-jobs-181498.html)
- [Senior ML Engineer, RAG: Klue (via Taro)](https://jointaro.com/jobs/klue/senior-machine-learning-engineer-rag)
- [Senior AI/ML Engineer: Ayasdi](https://jobs.khoslaventures.com/companies/ayasdi-2/jobs/52159080-senior-ai-ml-engineer)
- [Senior ML Engineer: Dice](https://www.dice.com/job-detail/48ac1662-7e2c-42fe-89d5-e7a828be6cd5)
- [ML Engineer job description template: Korebpo](https://korebpo.com/machine-learning-engineers-job-description-template/)
- [ML Engineer Interview 2026: The 5-Axis Grid](https://velyq.com/en/blog/entretien-ml-engineer-2026-grille)
- [System Design Interview 2026: LLM Apps, Vector DBs](https://lastroundai.com/blog/system-design-interview-mastery-2026)
- [LLM System Design Interview Guide 2026](https://www.calibreos.com/blog/genai-llm-system-design-interview-guide)
- [50 AI Engineer Interview Questions 2026](https://letsdatascience.com/blog/50-llm-and-ai-engineer-interview-questions-for-2026)
- [Generative AI System Design Interview Guide](https://www.systemdesignhandbook.com/guides/generative-ai-system-design-interview/)
