# Teaching an LLM to Reason: GRPO Fine-Tuning for Chain-of-Thought

## Project Overview

This repository contains the implementation of a reinforcement learning pipeline designed to elicit explicit, step-by-step reasoning in Large Language Models (LLMs). Developed as the foundational project for the comprehensively updated (December 2025) Udacity Generative AI Nanodegree, this codebase focuses on teaching a 3-billion parameter model to accurately count specific characters within strings—a notoriously challenging task for autoregressive models.

By leveraging **Group Relative Policy Optimization (GRPO)** and **Low-Rank Adaptation (LoRA)**, the model is optimized to autonomously break down words letter-by-letter. It is trained to strictly adhere to a Chain-of-Thought (CoT) reasoning format before generating a final answer, bridging the gap between simple instruction-following and algorithmic execution.

## Key Technologies & Concepts

* **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
* **Reinforcement Learning:** GRPO (DeepSeek methodology)
* **Parameter-Efficient Fine-Tuning:** LoRA (Rank 128, targeting attention and MLP projections), 4-bit Quantization (bitsandbytes)
* **Frameworks:** `unsloth` (for memory-efficient training), `vLLM` (for accelerated inference), `trl` (`GRPOTrainer`), Hugging Face `datasets`
* **Core Skills Demonstrated:** Reward Shaping, Prompt Engineering, RLHF/GRPO, Custom Dataset Generation.

## Methodology: Reward Shaping

The core engineering effort in this project lies in **Reward Shaping**. Rather than relying on a computationally expensive secondary critique model, a suite of programmatic reward functions was constructed using complex regular expressions to evaluate multiple generations per prompt. 

The GRPO trainer optimizes the model based on the following criteria:
1.  **Formatting Reward:** Enforces strict adherence to a `<reasoning>...</reasoning><answer>...</answer>` XML structure.
2.  **Spelling Reward:** Evaluates the generated letter-by-letter CoT breakdown, penalizing omissions, hallucinations, or length discrepancies against the ground-truth string.
3.  **Numbering Reward:** Ensures strict sequential indexing for each step in the reasoning block.
4.  **Counting Reward:** Validates the progressive, rolling count of the target character throughout the word breakdown.
5.  **Task Correctness:** Confirms the final extracted integer in the answer block matches the true count.

## Results & Impact

* **Baseline:** Under standard zero-shot prompting, the base model frequently hallucinated character frequencies or failed to maintain accurate counts for longer strings.
* **Post-Training:** The GRPO-optimized model reliably generates a structured, step-by-step spelling and counting breakdown, significantly improving final task accuracy and demonstrating verifiable internal logic.

## Repository Structure

* `grpo_training.ipynb`: The primary executable notebook. Contains the custom dataset generator, the programmatic reward functions, the LoRA/Unsloth configuration, and the GRPO training loop.