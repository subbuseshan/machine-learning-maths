# ML Engineer Maths — What's Next

Gaps left after the Coursera *Mathematics for ML & Data Science* specialization (C1–C3), in priority order.
Tick items off as you go; add notebooks under `C4/` (or similar) to keep the same layout.

## 1. Information theory  ⭐ highest priority
- [ ] Entropy, cross-entropy, KL divergence: definitions + intuition
- [ ] Derive binary/categorical cross-entropy loss from MLE (links to C3 MLE)
- [ ] Implement softmax + cross-entropy in NumPy; compare with `sklearn.metrics.log_loss`
- [ ] Where KL shows up: VAEs, knowledge distillation, RLHF (KL penalty)

## 2. Matrix calculus & vectorized backprop
- [ ] Jacobians and gradients of matrix expressions (`d(Wx)/dW`, `d(xᵀAx)/dx`)
- [ ] Chain rule on computational graphs
- [ ] Re-do the C2 W3 2-layer network fully vectorized, then check gradients against `jax.grad`
- [ ] Build micrograd (Karpathy, *Zero to Hero* ep. 1)

## 3. SVD in depth
- [ ] SVD ↔ eigen-decomposition ↔ PCA relationship (extends C1 W4)
- [ ] Low-rank approximation: compress an image with truncated SVD
- [ ] Pseudo-inverse, least squares, and condition number
- [ ] Connect to LoRA (low-rank weight updates)

## 4. Optimization beyond vanilla gradient descent
- [ ] SGD and mini-batches; noise vs. batch size
- [ ] Momentum, RMSProp, Adam: implement each and compare on the same loss surface
- [ ] Learning-rate schedules (step, cosine, warmup)
- [ ] Convexity basics; why NN losses are non-convex
- [ ] L1/L2 regularization as constraints and as Bayesian priors (links to C3 MAP)

## 5. Multivariate Gaussian
- [ ] Covariance matrix geometry; sampling with Cholesky
- [ ] Mahalanobis distance
- [ ] (Bonus) Gaussian processes intro

## 6. Numerical stability
- [ ] Log-sum-exp trick; stable softmax
- [ ] Float formats: fp32 vs fp16 vs bf16 (range vs. precision)
- [ ] Vanishing/exploding gradients and why init (Xavier/He) matters

## 7. Norms & high-dimensional geometry
- [ ] L1 / L2 / L∞ norms, cosine similarity
- [ ] Curse of dimensionality: distance concentration experiment
- [ ] Embeddings & nearest-neighbour search (the maths behind vector DBs)

## 8. Optional (research / generative roles)
- [ ] Markov chains, MCMC sampling
- [ ] Bayesian inference beyond MAP
- [ ] Diffusion-model maths (forward/reverse noising process)

## Resources
- [ ] Andrej Karpathy: *Neural Networks: Zero to Hero* (covers 1, 2, 6 hands-on)
- [ ] Deisenroth et al.: *Mathematics for Machine Learning* (free PDF): reference for 2, 3, 5
- [ ] Goodfellow et al.: *Deep Learning*, Ch. 2–4 (linear algebra, probability, numerical computation)
