<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind)
[![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind?style=social)](https://github.com/jingyaogong/minimind/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind)](https://github.com/jingyaogong/minimind/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/jingyaogong/minimind/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniMind%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

</div>

<div align="center">

![GitHub Trend](https://trendshift.io/api/badge/repositories/12586)

</div>

<div align="center">
  <h3>"The Great Way is Simple"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

* This open-source project aims to train an ultra-small language model MiniMind with approximately 64M parameters entirely from scratch, using only 3 CNY in cost and 2 hours of training time.
* The MiniMind series is extremely lightweight, with the smallest version on the main branch being approximately $\frac{1}{2700}$ the size of GPT-3, striving to enable even ordinary personal GPUs to quickly complete training and reproduction.
* The project also open-sources the minimalist structure and complete training pipeline of large models, covering the entire process code for MoE, data cleaning, Pretraining, Supervised Fine-Tuning (SFT), LoRA, RLHF (DPO), RLAIF (PPO / GRPO / CISPO), Tool Use, Agentic RL, Adaptive Thinking, and Model Distillation.
* MiniMind has also been extended to a visual multimodal version [MiniMind-V](https://github.com/jingyaogong/minimind-v), a diffusion language model (MiniMind-dLM), and a linear attention model (MiniMind-Linear), See [Discussion](https://github.com/jingyaogong/minimind/discussions) for details.
* All core algorithm code in the project is implemented from scratch using native PyTorch, without relying on high-level abstract interfaces provided by third-party libraries.
* This is not only a full-stage open-source reproduction project for large language models, but also a tutorial oriented towards LLM introduction and practice.
* We hope this project can provide a reproducible, understandable, and extensible starting point for more people, to share the joy of creation together and promote the progress of the broader AI community.

> Note: This project is open-sourced under the Apache 2.0 license and is completely free. "2 hours" refers to the measured time to run `1 epoch` of the SFT stage on a single NVIDIA 3090, and "3 CNY" refers to the corresponding GPU rental cost for that duration.

> **Personal note:** I forked this repo primarily to study the GRPO and DPO training pipelines. If you're also learning from this project, I found it helpful to start with `train_pretrain.py` before diving into the RL-based fine-tuning scripts.

---

<div align="center">

![minimind-3](./images/minimind-3.gif)

[🔗 Online Demo](https://