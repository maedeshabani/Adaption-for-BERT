Adapting BERT for Sentiment Analysis with LoRA and LoRA+

Overview

This project explores parameter-efficient fine-tuning of BERT (bert-base-cased) using LoRA (Low-Rank Adaptation) and LoRA+, comparing their performance against full fine-tuning. Our experiments are conducted on the IMDB sentiment analysis dataset, analyzing the trade-offs between efficiency and accuracy.

Motivation

Fine-tuning large models like BERT is computationally expensive. LoRA and LoRA+ offer efficient alternatives by modifying only small parts of the model while keeping the majority of weights frozen. Our study aims to evaluate their effectiveness compared to full fine-tuning.

Methods

We evaluate three approaches:

Full Fine-Tuning – Updates all model parameters.

LoRA – Introduces low-rank trainable matrices in attention layers, reducing computational cost.

LoRA+ – An enhanced version of LoRA with optimized learning rates and targeted module selection.

Experimental Setup

Dataset: IMDB sentiment analysis dataset from Hugging Face

Training Configurations:

Batch Size: 32

Epochs: 2

Learning Rate: 5e-5 (LoRA+ modifies this for certain parameters)

Evaluation: Every 128 steps

Hardware: GPU/TPU (Google Colab)

LoRA Hyperparameters:

Rank: 8

Alpha: 16

Dropout: 0.1

LoRA+ Hyperparameters:

Rank: 8

Alpha: 32

Dropout: 0.1

Target Modules: ["query", "value"]

Adaptive Learning Rates:

ηA = 5e-5 (same as base model)

ηB = 24 × ηA = 1.2e-3 (higher learning rate for LoRA matrices)

Results

Method

Training Time (s)

Eval Loss ↓

Accuracy ↑

Precision ↑

Recall ↑

Fine-Tuning

2940.13

0.2199

93.60%

93.73%

93.46%

LoRA

2738.39

0.2520

89.74%

89.00%

90.70%

LoRA+

2739.33

0.2396

90.27%

88.89%

92.05%

Key Takeaways

✅ LoRA and LoRA+ reduce training time by ~6.8% compared to full fine-tuning.
✅ LoRA+ improves accuracy and recall over LoRA while maintaining computational efficiency.
✅ Full fine-tuning achieves the best performance but is the most expensive.

Conclusion

Fine-tuning remains the best option for maximum performance.

LoRA provides significant speedup with some accuracy trade-off.

LoRA+ offers a great balance between performance and efficiency, making it a strong alternative to full fine-tuning.

Repository Structure

├── bert_adaptation.ipynb  # Implementation Notebook
├── AdaptionBERT_LoRA_LoRA+_report.pdf  # Detailed report
├── README.md  # This file

How to Use

Clone the repository:

git clone https://github.com/your-username/bert-lora-adaptation.git
cd bert-lora-adaptation

Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook bert_adaptation.ipynb to train and evaluate models.

Contributors

Maede Shabani Samgh Abadi (M2 DS, Université Paris-Saclay)

Negin Heidarifard (M2 AI, Université Paris-Saclay)

Mennaallah Khaled Abdou Salim (M2 IoT, Université Paris-Saclay)

License

MIT License
