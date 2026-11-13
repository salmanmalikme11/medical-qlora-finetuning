# Medical QLoRA Fine-tuning

Fine-tuning Llama-3.1 8B Instruct for medical multiple-choice questions with QLoRA, using [Unsloth](https://github.com/unslothai/unsloth) on a free Colab GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salmanmalikme11/medical-qlora-finetuning/blob/main/Medical_QLoRA_Finetuning_Colab.ipynb)

## What it does

- Loads [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa), a set of real questions from Indian medical entrance exams
- Measures the base model's accuracy on 500 validation questions
- Fine-tunes a 4-bit Llama-3.1 8B with LoRA adapters on 5,000 training questions
- Measures accuracy again on the same 500 questions and saves the adapter

## Approach

- **Model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, LoRA rank 16 on all attention and MLP projections
- **Prompt format:** Llama-3 chat template via `tokenizer.apply_chat_template`. The model answers with one letter (A-D).
- **Loss:** computed only on the answer token (`train_on_responses_only`), not on the question
- **Evaluation:** compares the logits of `A`/`B`/`C`/`D` at the answer position, so the score doesn't depend on parsing generated text

## Running it

Open the notebook in Colab, choose a GPU runtime (a T4 is enough) and run all cells. Training takes about 20-25 minutes.

To train on more data, change `N_TRAIN` in the config cell.
