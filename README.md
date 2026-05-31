# LLM bias isolation for preference learning stage: Triangulated Preference Shift score
Companion code, configuration files, raw data and supplementary information for the paper: **Isolating LLM Lexical Bias: A Curation-Free Triangulated Metric for Preference-Stage Learning**.

Published paper: [FLAIRS Proceedings PDF](https://journals.flvc.org/FLAIRS/article/view/141843/147188)

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

## Licence

- **Code** (`src/`, `scripts/`): MIT No Attribution (MIT-0). See [`LICENSE`](LICENSE). Use it freely, no attribution required.
- **Data** (`data/`, `build/`): CC0 1.0 Universal (public domain dedication). See [`LICENSE-DATA`](LICENSE-DATA).

Note: the underlying PubMed abstracts used as prompts are included for research; the CC0 dedication covers only our derivatives (cleaned generations, POS tags, TPS tables, scripts), and the source abstract texts remain under their respective terms.

## Citation

If you use this code or data, a citation is appreciated (though not required; see the licence).

```bibtex
@article{ming-etal-2026-isolating,
  title   = {Isolating LLM Lexical Bias: A Curation-Free Triangulated Metric for Preference-Stage Learning},
  author  = {Ming, Xiaoyang and Hernandez, Jose and Juzek, Thomas Stephan},
  journal = {The International FLAIRS Conference Proceedings},
  volume  = {39},
  number  = {1},
  year    = {2026},
  doi     = {10.32473/flairs.39.1.141843}
}
```

## AI Assistance

Repository polished with Claude Code.
