# DeepSeek-V3 Technical Overview & Architecture

## 1. Executive Summary
DeepSeek-V3 is an advanced open-weights Mixture-of-Experts (MoE) language model developed with a total of 671 billion parameters, activating 37 billion parameters per token during inference. It introduces Multi-Head Latent Attention (MLA) and DeepSeekMoE architecture to optimize computational efficiency and inference throughput.

## 2. Architectural Innovations

### Multi-Head Latent Attention (MLA)
Traditional Multi-Head Attention (MHA) suffers from high Key-Value (KV) cache memory overhead during inference. DeepSeek-V3 implements Multi-Head Latent Attention (MLA), which compresses the KV cache into a low-rank latent vector:
- **Compression Ratio**: Reduces KV cache memory footprint by approximately 93.3% compared to standard MHA.
- **Inference Speed**: Enables significantly larger batch sizes and higher token throughput on consumer and enterprise GPUs.

### DeepSeekMoE Architecture
- **Fine-Grained Experts**: Employs fine-grained expert division with 256 routed experts and 1 shared expert.
- **Active Parameters**: Routes tokens to the top-8 routed experts alongside the shared expert.
- **Auxiliary-Loss-Free Load Balancing**: Minimizes performance degradation that typically occurs with standard auxiliary load balancing losses.

## 3. Training and Infrastructure
- **Dataset Scale**: Trained on 14.8 trillion high-quality multilingual tokens.
- **Compute Cost**: Trained on a cluster of 2,048 NVIDIA H800 GPUs over approximately 2 months. Total training cost was reported at approximately $5.58 million USD.
- **Precision**: Uses native FP8 mixed precision training framework for stability and throughput.

## 4. Benchmark Performance
- **MMLU**: Achieved 88.5%, outperforming many closed-source frontier models.
- **MATH 500**: Scored 90.2% on competition-level mathematics without tool use.
- **HumanEval (Python Code Generation)**: Reached 82.6% zero-shot pass@1.
- **Codeforces**: Elo rating exceeding 2000 in competitive programming evaluations.

## 5. Deployment and Context Window
- **Context Length**: Supports native context windows up to 128K tokens using YaRN context extension techniques.
- **Hardware Footprint**: Can run quantized 4-bit (AWQ) inference on nodes with multiple consumer RTX 3090/4090 GPUs.
