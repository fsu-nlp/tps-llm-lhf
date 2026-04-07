# Supplemental Information

This document serves as the extended technical and data appendix for the study: **"Isolating instruction-stage effects enables linguistic probing of model behavior."** All materials are provided to ensure the reproducibility and transparency of our findings.

---

## 1. Code, Data, and Computational Setup

### Repository Access
All source code, spaCy/CoNLL-U processing scripts, and raw Triangulated Preference Shift (TPS) results are available at:  
[https://github.com/fsu-nlp/tps-llm-lhf](https://github.com/fsu-nlp/tps-llm-lhf)

### Computational Environment
Experiments were conducted across two primary infrastructures to ensure cross-platform consistency.

**I. GPU Server (High Performance)**
- **Hardware:** NVIDIA H100 PCIe (80 GB VRAM); Intel Xeon Platinum 8480+ CPU.
- **Memory:** 221 GiB system RAM.
- **Environment:** Ubuntu 24.04.2 LTS; NVIDIA Driver 570.148.08; CUDA 12.8.

**II. University HPC Node**
- **Hardware:** NVIDIA RTX A4500 (20 GB VRAM); AMD EPYC 7313 CPU (16 cores).
- **Environment:** Linux kernel 4.18.0-372.32.1; 251 GiB system storage.

### Software Stack
- **Python:** 3.12.3
- **Deep Learning:** PyTorch 2.8.0+cu128, `transformers` 4.56.1, `accelerate` 1.10.1, `peft` 0.17.1.
- **NLP Processing:** `spaCy` 3.8.7.

---

## 2. Ethical Considerations

The risks presented in our research are minimal as we exclusively utilize public data and open-weights models. Our study serves as a diagnostic identification of model behavior bias, conducted at a moderate compute expense (~$1,160). 

A significant ethical concern in the AI industry is the human labor underlying preference-learning datasets, particularly reports regarding exploitative working conditions. While this concern is inherent to the broader industry's data collection choices rather than our specific diagnostic methodology, we believe it requires ongoing transparency and scrutiny from model developers and the research community.

---

## 3. TPS Ranking Results

The following tables list the Top 20 lexical items (lemma + UPOS) exhibiting the highest Triangulated Preference Shift ($wTPS$) per model family.

### Falcon & Gemma
| Rk | Falcon | Rk | Falcon | | Rk | Gemma | Rk | Gemma |
|:---|:---|:---|:---|---|:---|:---|:---|:---|
| 1 | `these_DET` | 11 | `enhance_VERB` | | 1 | `furthermore_ADV` | 11 | `further_ADJ` |
| 2 | `to_PART` | 12 | `potential_ADJ` | | 2 | `specifically_ADV` | 12 | `while_SCONJ` |
| 3 | `for_ADP` | 13 | `into_ADP` | | 3 | `reveal_VERB` | 13 | `these_DET` |
| 4 | `this_DET` | 14 | `importance_NOUN` | | 4 | `to_ADP` | 14 | `such_ADJ` |
| 5 | `further_ADJ` | 15 | `additionally_ADV` | | 5 | `significant_ADJ` | 15 | `'s_PART` |
| 6 | `highlight_VERB` | 16 | `insight_NOUN` | | 6 | `, _PUNCT` | 16 | `observe_VERB` |
| 7 | `research_NOUN` | 17 | `improve_VERB` | | 7 | `a_DET` | 17 | `several_ADJ` |
| 8 | `finding_NOUN` | 18 | `therapeutic_ADJ` | | 8 | `include_VERB` | 18 | `we_PRON` |
| 9 | `could_AUX` | 19 | `outcome_NOUN` | | 9 | `compare_VERB` | 19 | `reduce_VERB` |
| 10 | `strategy_NOUN` | 20 | `understand_VERB` | | 10 | `exhibit_VERB` | 20 | `mechanism_NOUN` |

### Llama & Mistral
| Rk | Llama | Rk | Llama | | Rk | Mistral | Rk | Mistral |
|:---|:---|:---|:---|---|:---|:---|:---|:---|
| 1 | `furthermore_ADV` | 11 | `highlight_VERB` | | 1 | `to_PART` | 11 | `various_ADJ` |
| 2 | `, _PUNCT` | 12 | `as_ADP` | | 2 | `study_NOUN` | 12 | `this_PRON` |
| 3 | `additionally_ADV` | 13 | `a_DET` | | 3 | `use_VERB` | 13 | `such_ADJ` |
| 4 | `to_ADP` | 14 | `lead_VERB` | | 4 | `they_PRON` | 14 | `it_PRON` |
| 5 | `which_PRON` | 15 | `contrast_NOUN` | | 5 | `a_DET` | 15 | `as_ADP` |
| 6 | `such_ADJ` | 16 | `however_ADV` | | 6 | `you_PRON` | 16 | `investigate_VERB` |
| 7 | `indicate_VERB` | 17 | `between_ADP` | | 7 | `researcher_NOUN` | 17 | `research_NOUN` |
| 8 | `significant_ADJ` | 18 | `while_SCONJ` | | 8 | `on_ADP` | 18 | `include_VERB` |
| 9 | `include_VERB` | 19 | `exhibit_VERB` | | 9 | `this_DET` | 19 | `from_ADP` |
| 10 | `compare_VERB` | 20 | `its_PRON` | | 10 | `aim_VERB` | 20 | `conduct_VERB` |

### OLMo & Yi
| Rk | OLMo | Rk | OLMo | | Rk | Yi | Rk | Yi |
|:---|:---|:---|:---|---|:---|:---|:---|:---|
| 1 | `these_DET` | 11 | `which_PRON` | | 1 | `to_PART` | 11 | `research_NOUN` |
| 2 | `to_ADP` | 12 | `lead_VERB` | | 2 | `this_DET` | 12 | `crucial_ADJ` |
| 3 | `finding_NOUN` | 13 | `, _PUNCT` | | 3 | `study_NOUN` | 13 | `which_PRON` |
| 4 | `additionally_ADV` | 14 | `highlight_VERB` | | 4 | `a_DET` | 14 | `aim_VERB` |
| 5 | `to_PART` | 15 | `further_ADJ` | | 5 | `this_PRON` | 15 | `for_ADP` |
| 6 | `for_ADP` | 16 | `as_ADP` | | 6 | `as_ADP` | 16 | `have_AUX` |
| 7 | `indicate_VERB` | 17 | `enhance_VERB` | | 7 | `such_ADJ` | 17 | `researcher_NOUN` |
| 8 | `could_AUX` | 18 | `crucial_ADJ` | | 8 | `their_PRON` | 18 | `include_VERB` |
| 9 | `this_DET` | 19 | `its_PRON` | | 9 | `to_ADP` | 19 | `it_PRON` |
| 10 | `such_ADJ` | 20 | `provide_VERB` | | 10 | `various_ADJ` | 20 | `into_ADP` |

---

## 4. Etymology Analysis Computation and Results

### 4.1 Computation Process
We define the set of content-word categories as $\text{POS} = \{\text{NOUN, VERB, ADJ, ADV}\}$, where $p \in \text{POS}$ denotes an individual category. For each model family, we obtain the lemma $w$ total frequency $c_S(w) = \sum_{i=1}^{R} Y_{iS}(w)$ via the windowed prevalence method. 

The categorical Romance ratio $\rho_{p,S}$ is defined as the Romance-origin mass divided by the total mass of words with known Germanic or Romance origin:

$$\rho_{p,S} = \frac{\sum_{w \in p \cap \text{Romance}} c_S(w)}{\sum_{w \in p \cap \text{Romance}} c_S(w) + \sum_{w \in p \cap \text{Germanic}} c_S(w)}$$

To obtain Figure 1 results, we computed the Odds Ratio ($O_{p,S}$) and final Relative Shift ($RS_p$) as:
$$O_{p,S} = \frac{\rho_{p,S}}{1 - \rho_{p,S}}, \quad RS_p = \frac{O_{p,I}}{O_{p,B}}$$
and $RS_p$ values are plotted in Figure 1 for each model family.

For the overall aggregated shift $Agg_S$, we utilize categorical mass $M_{p,S}$ and weights $W_{p,S}$:
$$M_{p, S} = \sum_{w \in p} c_S(w), \quad W_{p,S} = \frac{M_{p,S}}{\sum_{p'} M_{p',S}}$$
$$Agg_S = \sum_{p \in \text{POS}} (\rho_{p,S} \cdot W_{p,S})$$

### 4.2 Model-wise Distribution Tables
The tables below present granular etymological data for each model family. The fractional values in the "Mass" columns arise from the windowed frequency $c_S(w) = \sum_{i=1}^{R} Y_{iS}(w)$, where $Y_{iS}(w)$ is the mean of binary indicator functions across $N_w = 4$ stratified windows.

#### Falcon
| POS | Variant | Germanic Mass | Romance Mass | Total Mass ($M_{p,S}$) | Romance Ratio ($\rho_{p,S}$) | Odds Ratio |
|:---|:---|:---|:---|:---|:---|:---|
| ADJ | Base | 18,787.50 | 35,850.50 | 54,638.00 | 65.61% | 1.467 |
| | Instruct | 26,604.25 | 74,487.50 | 101,091.75 | 73.68% | |
| ADV | Base | 13,254.00 | 40.25 | 13,294.25 | 0.30% | 2.545 |
| | Instruct | 16,949.00 | 129.25 | 17,078.25 | 0.76% | |
| NOUN | Base | 31,697.25 | 123,387.00 | 155,084.25 | 79.56% | 1.027 |
| | Instruct | 53,502.00 | 213,824.00 | 267,326.00 | 79.99% | |
| VERB | Base | 15,747.00 | 42,878.00 | 58,625.00 | 73.14% | 1.495 |
| | Instruct | 24,185.75 | 98,430.75 | 122,616.50 | 80.28% | |
| **AGGREGATE** | **Base** | **79,485.75** | **202,155.75** | **281,641.50** | **71.78%** | **1.255** |
| | **Instruct** | **121,241.00** | **386,871.50** | **508,112.50** | **76.14%** | |

#### Gemma
| POS | Variant | Germanic Mass | Romance Mass | Total Mass ($M_{p,S}$) | Romance Ratio ($\rho_{p,S}$) | Odds Ratio |
|:---|:---|:---|:---|:---|:---|:---|
| ADJ | Base | 23,838.25 | 45,807.50 | 69,645.75 | 65.77% | 1.080 |
| | Instruct | 27,211.00 | 56,501.00 | 83,712.00 | 67.49% | |
| ADV | Base | 16,801.50 | 53.25 | 16,854.75 | 0.32% | 1.000 |
| | Instruct | 26,608.50 | 85.00 | 26,693.50 | 0.32% | |
| NOUN | Base | 38,539.50 | 156,930.00 | 195,469.50 | 80.28% | 1.062 |
| | Instruct | 43,697.50 | 188,840.25 | 232,537.75 | 81.21% | |
| VERB | Base | 19,881.50 | 59,545.00 | 79,426.50 | 74.97% | 1.267 |
| | Instruct | 25,008.75 | 94,949.00 | 119,957.75 | 79.15% | |
| **AGGREGATE** | **Base** | **119,060.75** | **262,335.75** | **361,396.50** | **72.59%** | **1.049** |
| | **Instruct** | **122,525.75** | **340,375.25** | **462,901.00** | **73.53%** | |

#### Llama
| POS | Variant | Germanic Mass | Romance Mass | Total Mass ($M_{p,S}$) | Romance Ratio ($\rho_{p,S}$) | Odds Ratio |
|:---|:---|:---|:---|:---|:---|:---|
| ADJ | Base | 24,707.25 | 48,589.25 | 73,296.50 | 66.29% | 0.990 |
| | Instruct | 27,933.25 | 54,400.00 | 82,333.25 | 66.07% | |
| ADV | Base | 17,492.00 | 56.50 | 17,548.50 | 0.32% | 1.220 |
| | Instruct | 26,331.25 | 102.50 | 26,433.75 | 0.39% | |
| NOUN | Base | 40,001.25 | 162,412.75 | 202,414.00 | 80.24% | 1.039 |
| | Instruct | 44,433.75 | 187,466.00 | 231,899.75 | 80.84% | |
| VERB | Base | 20,465.25 | 60,069.75 | 80,535.00 | 74.59% | 1.034 |
| | Instruct | 25,923.25 | 78,707.75 | 104,631.00 | 75.22% | |
| **AGGREGATE** | **Base** | **102,665.75** | **271,128.25** | **373,794.00** | **72.53%** | **0.975** |
| | **Instruct** | **124,621.50** | **320,676.25** | **445,297.75** | **72.02%** | |

#### Mistral
| POS | Variant | Germanic Mass | Romance Mass | Total Mass ($M_{p,S}$) | Romance Ratio ($\rho_{p,S}$) | Odds Ratio |
|:---|:---|:---|:---|:---|:---|:---|
| ADJ | Base | 21,309.75 | 39,603.75 | 60,913.50 | 65.02% | 1.414 |
| | Instruct | 19,979.00 | 52,520.75 | 72,499.75 | 72.44% | |
| ADV | Base | 14,246.75 | 53.50 | 14,300.25 | 0.37% | 2.692 |
| | Instruct | 13,360.00 | 133.75 | 13,493.75 | 0.99% | |
| NOUN | Base | 34,817.50 | 137,428.00 | 172,245.50 | 79.79% | 1.082 |
| | Instruct | 41,447.25 | 177,053.50 | 218,500.75 | 81.03% | |
| VERB | Base | 18,007.50 | 49,447.25 | 67,454.75 | 73.30% | 1.268 |
| | Instruct | 24,323.75 | 84,668.50 | 108,992.25 | 77.68% | |
| **AGGREGATE** | **Base** | **88,381.50** | **226,532.50** | **314,914.00** | **71.93%** | **1.238** |
| | **Instruct** | **99,110.00** | **314,376.50** | **413,486.50** | **76.03%** | |

#### Olmo
| POS | Variant | Germanic Mass | Romance Mass | Total Mass ($M_{p,S}$) | Romance Ratio ($\rho_{p,S}$) | Odds Ratio |
|:---|:---|:---|:---|:---|:---|:---|
| ADJ | Base | 22,616.00 | 42,943.50 | 65,559.50 | 65.50% | 1.288 |
| | Instruct | 28,260.50 | 69,097.25 | 97,357.75 | 70.97% | |
| ADV | Base | 16,781.00 | 55.75 | 16,836.75 | 0.33% | 2.374 |
| | Instruct | 18,846.50 | 149.00 | 18,995.50 | 0.78% | |
| NOUN | Base | 36,632.50 | 147,110.25 | 183,742.75 | 80.06% | 1.077 |
| | Instruct | 50,691.50 | 219,207.75 | 269,899.25 | 81.22% | |
| VERB | Base | 17,546.00 | 50,043.25 | 67,589.25 | 74.04% | 1.356 |
| | Instruct | 24,520.25 | 94,840.50 | 119,360.75 | 79.46% | |
| **AGGREGATE** | **Base** | **93,575.50** | **240,152.75** | **333,728.25** | **71.96%** | **1.221** |
| | **Instruct** | **122,318.75** | **383,294.50** | **505,613.25** | **75.81%** | |

#### Yi
| POS | Variant | Germanic Mass | Romance Mass | Total Mass ($M_{p,S}$) | Romance Ratio ($\rho_{p,S}$) | Odds Ratio |
|:---|:---|:---|:---|:---|:---|:---|
| ADJ | Base | 19,782.75 | 37,138.25 | 56,921.00 | 65.25% | 1.441 |
| | Instruct | 20,327.00 | 55,000.25 | 75,327.25 | 73.02% | |
| ADV | Base | 13,324.00 | 41.25 | 13,365.25 | 0.31% | 2.855 |
| | Instruct | 16,323.75 | 145.00 | 16,468.75 | 0.88% | |
| NOUN | Base | 31,484.50 | 125,154.25 | 156,638.75 | 79.90% | 1.095 |
| | Instruct | 38,543.25 | 167,817.00 | 206,360.25 | 81.32% | |
| VERB | Base | 17,156.50 | 45,188.75 | 62,345.25 | 72.48% | 1.149 |
| | Instruct | 25,924.00 | 78,490.00 | 104,414.00 | 75.17% | |
| **AGGREGATE** | **Base** | **81,747.75** | **207,522.50** | **289,270.25** | **71.74%** | **1.174** |
| | **Instruct** | **101,138.00** | **301,452.25** | **402,570.25** | **74.88%** | |
