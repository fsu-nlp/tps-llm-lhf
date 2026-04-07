# LLM bias isolation for preference learning stage: Triangulated Preference Shift score
Companion code, configuration files, raw data and supplementary information for the paper: **Isolating LLM Lexical Bias: A Curation-Free Triangulated Metric for Preference-Stage Learning**.

---

## TL;DR: The Pipeline
We provide an end-to-end, **deterministic** pipeline to quantify preference-stage shifts (TPS) in model continuations:
**prompts → generation (Base/Instruct) → deletion-only cleaning → POS → TPS discover & eval**.

- **Purpose:** test whether preference tuning causally shifts lexical choice distributions.
- **Outputs:** word-level weights, sentence/document/corpus-level scores, and batch scripts for model families.
- **Repro path:** fixed decoding; deletion-only cleaning; UD POS → CoNLL-U.

---

## Requirements
- General requirements as per .toml. 
- **Generation** requires a GPU with ≥20 GB VRAM (per 7–8B model), Hugging Face auth.
- Everything else: CPU is sufficient, you can even use our precomputed TPS tables.

---
## Repository Structure
- **src/:** :Core logic for TPS calculation and etymology parsing.
- **scripts/** :Pipeline running scripts.
- **data/** :Prompts for model generation, raw data (42,000 pubmed abstacts and cleaned generations) OSF links.
- **[COMMANDS](./COMMANDS.md)** :Step-by-step CLI instructions for reproduction.
- **[SUPPLEMENTARY](./SUPPLEMENTARY.md)** :Supplemental information for paper's Supporting Materials. 

## Getting started
Use the scripts directly, with the commands provided in [COMMANDS](./COMMANDS.md)

## Supporting Materials
Please check Supporting Materials in file [SUPPLEMENTARY](./SUPPLEMENTARY.md)


---

## License

This repository is released under the **<LICENSE NAME>** license.
See [`LICENSE`](./LICENSE) for details.


