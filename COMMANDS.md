# lhf-lex (scaffold v0.5)
Authors: Xiaoyang Ming, Jose Hernandez, Thomas Stephan Juzek

Modular scaffold for **LLM lexical preference learning bias**.  
**Scope:** prompt preparation → generation (Base/Instruct) → deletion-only cleaning → POS tagging → **TPS** estimation and scoring.

---

## Contents
- [1. Setup](#1-setup)
  - [1.1 Install](#11-install)
  - [1.2 Access to generation models (Hugging Face)](#12-access-to-generation-models-hugging-face)
  - [1.3 OpenAI token for cleaning](#13-openai-token-for-cleaning)
  - [1.4 What this scaffold currently covers](#14-what-this-scaffold-currently-covers)
- [2. Human abstracts prep](#2-human-abstracts-prep)
- [3. Text generation (Base vs Instruct)](#3-text-generation-base-vs-instruct)
  - [3.1 Output roots](#31-output-roots)
  - [3.2 One-liner examples per model](#32-one-liner-examples-per-model)
- [4. Cleaning raw generations (deletion-only)](#4-cleaning-raw-generations-deletion-only)
  - [4.1 Credentials](#41-credentials)
  - [4.2 Step 1 — Pattern pass](#42-step-1--pattern-pass)
  - [4.3 Step 2 — GPT cleaner](#43-step-2--gpt-cleaner)
  - [4.4 Human cleaning](#44-human-cleaning)
- [5. POS tagging → CoNLL-U](#5-pos-tagging--conll-u)
  - [5.1 Tag with the transformer model](#51-tag-with-the-transformer-model)
- [6. Triangulate with TPS](#6-triangulate-with-tps)
  - [6.1 estimate / discover first](#61-estimate--discover-first)
  - [6.2 Scoring (TPS)](#62-scoring-tps)
    - [Single-file examples](#single-file-examples)
  - [6.3 Score all models](#63-score-all-models)
    - [6.3.1 example 1](#631-example-1-unchanged)
    - [6.3.2 example 2](#632-example-2-unchanged)
- [Provenance](#provenance)

---

## 1. Setup

### 1.1 Install
```bash
git clone https://github.com/<user>/lhf-lex-ablation.git
cd lhf-lex-ablation
python -m venv .venv && source .venv/bin/activate     # recommended
pip install -e .
````

This installs the package and CLI entry points.

---

### 1.2 Access to generation models (Hugging Face)

```bash
hf auth login --token "$HF_TOKEN"
hf auth whoami    # verify
```

---

### 1.3 OpenAI token for cleaning

```bash
export OPENAI_API_KEY=sk-...
```

---

### 1.4 What this scaffold currently covers

* **Prompts** from human halves (PubMed), one abstract per line.
* **Deterministic generation** (temperature 0, greedy) for Base/Instr variants of six families.
* **Symmetric deletion-only cleaning** (patterns → OpenAI cleaner).
* **POS tagging** to CoNLL-U (UPOS).
* **TPS** estimation (**discover**) and scoring (**eval**) at word/sequence/document/corpus levels.

---

## 2. Human abstracts prep

Place yearly text files under `data/human_abstracts/` named `2012.txt` … `2021.txt`, **one abstract per line**.

Sample **N=4200 per year** and split each abstract into halves:

```bash
python scripts/prep_human_abstracts.py data/human_abstracts \
  --years 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 \
  --sample-per-year 4200 \
  --out-jsonl  data/prompts/human_halves.jsonl \
  --out-prompts data/prompts/human_first_halves.txt
```

**Outputs**

* `data/prompts/human_halves.jsonl` with fields
  `year, idx, text, sentences, first_half, second_half`
* `data/prompts/human_first_halves.txt` — one `first_half` per line (generation input)

> **Notes**
> • The JSONL is the ground truth for later cleaning alignment.
> • Keep filenames and relative locations — later scripts assume these defaults.

---

## 3. Text generation (Base vs Instruct)

Deterministic decoding is used to remove sampling noise.

### 3.1 Output roots

Create per-family raw output directories:

```bash
mkdir -p out/pre/raw/{olmo2-1124-7b,mistral-7b-v0.3,llama-3.1-8b,gemma-3-4b,falcon3-7b,yi-1.5-6b}
```

### 3.2 One-liner examples per model

**OLMo 7b Base**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt   --model allenai/OLMo-2-1124-7B  --model-type base   --out out/pre/raw/olmo2-1124-7b/2025-09-07_base_pubmed-halves_greedy_bf16_rev8e91.jsonl   --greedy --bf16
```

**OLMo 7b Instruct**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt   --model allenai/OLMo-2-1124-7B-Instruct  --model-type instruct   --out out/pre/raw/olmo2-1124-7b/2025-09-07_instruct_pubmed-halves_greedy_bf16_rev8e91.jsonl   --greedy --bf16
```

**Mistral 7B v0.3 Base**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model mistralai/Mistral-7B-v0.3  --model-type base  --out out/pre/raw/mistral-7b-v0.3/2025-09-07_base_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Mistral 7B v0.3 Instruct**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model mistralai/Mistral-7B-Instruct-v0.3  --model-type instruct  --out out/pre/raw/mistral-7b-v0.3/2025-09-07_instruct_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Llama 3.1 8B Base**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model meta-llama/Llama-3.1-8B  --model-type base  --out out/pre/raw/llama-3.1-8b/2025-09-07_base_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Llama 3.1 8B Instruct**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model meta-llama/Llama-3.1-8B-Instruct  --model-type instruct  --out out/pre/raw/llama-3.1-8b/2025-09-07_instruct_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Gemma-3 4B**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model google/gemma-3-4b-pt  --model-type base  --out out/pre/raw/gemma-3-4b/2025-09-07_base_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Gemma-3 4B Instruct**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model google/gemma-3-4b-it  --model-type instruct  --out out/pre/raw/gemma-3-4b/2025-09-07_instruct_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Falcon3 7B Base**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model tiiuae/Falcon3-7B-Base  --model-type base  --out out/pre/raw/falcon3-7b/2025-09-07_base_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Falcon3 7B Instruct**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model tiiuae/Falcon3-7B-Instruct  --model-type instruct  --out out/pre/raw/falcon3-7b/2025-09-07_instruct_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Yi-1.5 6B Base**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model 01-ai/Yi-1.5-6B  --model-type base  --out out/pre/raw/yi-1.5-6b/2025-09-07_base_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

**Yi-1.5 6B Instruct**

```bash
python scripts/generate_text.py data/prompts/human_first_halves.txt  --model 01-ai/Yi-1.5-6B-Chat  --model-type instruct  --out out/pre/raw/yi-1.5-6b/2025-09-07_instruct_pubmed-halves_greedy_bf16_rev8e91.jsonl  --greedy --bf16
```

> **Tip:** `--greedy` removes sampling noise. For stochastic decoding, drop it or set `--temperature 0.7 --top-p 0.95 --seed 42`.

---

## 4. Cleaning raw generations (deletion-only)

Goal: remove **meta-commentary** and **obvious loops** from continuations.
Policy: **deletions only** (no paraphrasing, no insertions).

### 4.1 Credentials

```bash
export OPENAI_API_KEY=sk-...   # required for GPT cleaner
```

### 4.2 Step 1 — Pattern pass

Consumes raw generations and prepares cleaner inputs for both models and humans.

```bash
python scripts/clean_generations_patterns.py \
  --repo-root . \
  --in-out-dir out/pre/raw \
  --in-human-halves data/prompts/human_halves.jsonl \
  --out-root out/pre/patterncleaned
```

---

### 4.3 Step 2 — GPT cleaner

Runs the deletion-only pass over the pattern-cleaned inputs.

```bash
python scripts/clean_generations_gpt.py \
  --model gpt-4.1-mini \
  --in-out-dir  out/pre/patterncleaned/cleaning-prep-models \
  --in-human-halves out/pre/patterncleaned/cleaning-prep-human/human_halves.preclean.jsonl \
  --out-root   out/pre/cleaned
```

---

### 4.4 Human cleaning

We also want to clean the human halves cleaned:

```bash
python scripts/clean_generations_gpt.py \
  --model gpt-4.1-mini \
  --in-out-dir out/NONE \
  --in-human-halves out/pre/patterncleaned/cleaning-prep-human/human_halves.preclean.jsonl \
  --out-root out/pre/cleaned
```

---

## 5. POS tagging → CoNLL-U

We tag the **cleaned** continuations (and cleaned human halves) and write per-family `.conllu` files.

### 5.1 Tag with the transformer model (paper settings)

```bash
python scripts/pos_tag.py \
  --model en_core_web_trf \
  --in-out-dir out/pre/cleaned \
  --out-root   out/pre/pos-tr \
  --n-process  1
```

**Notes / gotchas**

* Determinism: spaCy's taggers are deterministic given the same model/version.
* Memory: `en_core_web_trf` benefits from GPU; set `CUDA_VISIBLE_DEVICES` if needed. Make sure cuda and cupy are running.

---


## 6. Triangulate with TPS
We run in two phases:

1. **discover (estimation)** — learn word-level weights from tagged corpora
2. **eval (scoring)** — apply weights to new CoNLL-U to obtain TPS at different levels

### 6.1 estimate / discover first

```bash
python scripts/analyse_tps.py \
  --mode discover \
  --pos-root out-ig/pre/pos-tr-full \
  --out-root out-ig/pre/analyses/tps-all \
  --key lemma_pos \
  --windowk 50 \
  --num-windows 4 \
  --window-policy quantile \
  --trim-common \
  --aggregate both \
  --seed 42
#   --exc-upos-calc NUM \
```

---

## 6.2 Scoring (TPS)

Use the same scorer with different granularities:

* `--level sequences`  → sTPS (sentence-level)
* `--level documents`  → dTPS (continuation-level)
* `--level dataset`    → cTPS (corpus-level)

Pick either a **per-family** weight table or the **aggregate** one from discovery.

**Note:** The results reported in our paper use cTPS (corpus-level) evaluation. To reproduce these results, first perform the discovery (estimation) phase using the commands in [#6.1 Estimate / Discover](#61-estimate--discover-first), then follow the multi-model scoring process detailed in [#6.3.2 Example 2](#632-example-2-unchanged).

### Single-file examples (choose one level)

```bash
# Example: score one file at sentence level (sTPS)
python scripts/analyse_tps.py \
  --mode eval \
  --level sequences \
  --wtable out-ig/pre/analyses/tps-all/policy=quantile_trim=yes_M=4_K=50_jitter=yes/falcon3-7b/tps_word_falcon3-7b.csv \
  --tau-col TPS \
  --tau-rectify \
  --eval-conllu out-ig/pre/pos-tr-full/falcon3-7b/2025-09-07_instruct_...conllu \
  --out-root out-ig/pre/analyses/tps-eval \
  --key lemma_pos \
  --exc-upos-calc PUNCT SYM \
  --source-tag F7B-I

# Same file at document level (dTPS)
python scripts/analyse_tps.py \
  --mode eval \
  --level documents \
  --wtable out-ig/pre/analyses/tps-all/policy=quantile_trim=yes_M=4_K=50_jitter=yes/falcon3-7b/tps_word_falcon3-7b.csv \
  --tau-col TPS \
  --tau-rectify \
  --eval-conllu out-ig/pre/pos-tr-full/falcon3-7b/2025-09-07_instruct_...conllu \
  --out-root out-ig/pre/analyses/tps-eval \
  --key lemma_pos \
  --exc-upos-calc PUNCT SYM \
  --source-tag F7B-I

# Corpus level (cTPS)
python scripts/analyse_tps.py \
  --mode eval \
  --level dataset \
  --wtable out-ig/pre/analyses/tps-all/policy=quantile_trim=yes_M=4_K=50_jitter=yes/aggregate/tps_word_ALLMODELS_macro.csv \
  --tau-col TPS \
  --tau-rectify \
  --eval-conllu out-ig/pre/pos-tr-full/falcon3-7b/2025-09-07_instruct_...conllu \
  --out-root out-ig/pre/analyses/tps-eval \
  --key lemma_pos \
  --exc-upos-calc PUNCT SYM \
  --source-tag F7B-I
```

**Notes**

* `--tau-col TPS` selects the TPS column; `--tau-rectify` zero-clips negative mass (recommended).
* For **per-family** scoring, use `.../<family>/tps_word_<family>.csv`. For pooled scoring, use the `aggregate` table.
* Keep `--key lemma_pos` aligned with your discovery key; exclusions should mirror discovery.

---

## 6.3 Score all models

### 6.3.1 example 1 *(unchanged)*

```bash
set -euo pipefail
shopt -s nullglob

WTABLE="out-ig/pre/analyses/tps-all/policy=quantile_trim=yes_M=4_K=50_jitter=yes/aggregate/tps_word_ALLMODELS_macro.csv"
ROOT="out-ig/pre/pos-tr-full"
OUT="out-ig/pre/analyses/tps-eval-all"

MODELS=(falcon3-7b gemma-3-4b llama-3.1-8b mistral-7b-v0.3 olmo2-1124-7b yi-1.5-6b)

score_variant () {
  local M="$1" VAR="$2" TAG="$3"
  local files=( "$ROOT/$M/"*"_${VAR}_"*.conllu )
  (( ${#files[@]} > 0 )) || { echo "No ${VAR} files for ${M}"; return 1; }

  python scripts/analyse_tps.py \
    --mode eval \
    --level dataset \
    --wtable "$WTABLE" \
    --tau-col TPS \
    --tau-rectify \
    --eval-conllu "${files[@]}" \
    --out-root "$OUT/$M" \
    --key lemma_pos \
    --lp 2.0 \
    --source-tag "$TAG"
}

mkdir -p "$OUT"
for M in "${MODELS[@]}"; do
  echo "== ${M} =="
  score_variant "$M" instruct "${M}-I"
  score_variant "$M" base     "${M}-B"
done
```

### 6.3.2 example 2 *(unchanged)*

```bash
set -euo pipefail
shopt -s nullglob

# Root that contains aggregate/ and per-model subfolders (each with tps_word_<model>.csv)
WROOT="out-ig/pre/analyses/tps-all/policy=quantile_trim=yes_M=4_K=50_jitter=yes"
ROOT="out-ig/pre/pos-tr-full"
OUT="out-ig/pre/analyses/tps-eval-per-family"

MODELS=(falcon3-7b gemma-3-4b llama-3.1-8b mistral-7b-v0.3 olmo2-1124-7b yi-1.5-6b)

score_variant () {
  local M="$1" VAR="$2" TAG="$3"

  # Per-family wtable path e.g. .../falcon3-7b/tps_word_falcon3-7b.csv
  local WTABLE="${WROOT}/${M}/tps_word_${M}.csv"
  [[ -f "$WTABLE" ]] || { echo "Missing wtable for ${M}: $WTABLE" >&2; return 1; }

  local files=( "$ROOT/$M/"*"_${VAR}_"*.conllu )
  (( ${#files[@]} > 0 )) || { echo "No ${VAR} files for ${M}" >&2; return 1; }

  python scripts/analyse_tps.py \
    --mode eval \
    --level dataset \
    --wtable "$WTABLE" \
    --tau-col TPS \
    --tau-rectify \
    --eval-conllu "${files[@]}" \
    --out-root "$OUT/$M" \
    --key lemma_pos \
    --lp 2.0 \
    --source-tag "$TAG"
}

mkdir -p "$OUT"
for M in "${MODELS[@]}"; do
  echo "== ${M} =="
  score_variant "$M" instruct "${M}-I"
  score_variant "$M" base     "${M}-B"
done
```

---

## Provenance

This README was prepared and formatted with GPT assistance (commands left unchanged).