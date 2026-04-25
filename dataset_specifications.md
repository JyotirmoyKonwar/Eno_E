# Dark Pattern Dataset Specifications

## Scope
This document summarizes the three datasets currently selected for Stage 1 training:
1. `WIPI/deceptive_patterns_synthetic`
2. `apart/darkbench`
3. `itsbaivab/mistral_dark_pattern_dataset`

It focuses on public dataset-page details, visible schema hints, sample structure, and practical implications for preprocessing and training.

---

## 1) WIPI/deceptive_patterns_synthetic

### Source
- Hugging Face dataset: `WIPI/deceptive_patterns_synthetic`

### What the public page shows
- The dataset viewer exposes three visible top-level columns: `input`, `output`, and `file_url`.
- The `input` field contains long OCR-style serialized descriptions of webpage elements.
- The `output` field contains serialized labels and explanations for each detected UI element.
- The `file_url` field points to a CSV-like source artifact such as `unilever_com-0.csv`, `komoot_com-0.csv`, `sleepshop.ca.csv`, or similar filenames.

### Inferred row structure
Each row appears to represent a webpage screenshot or OCR extraction serialized into text.

#### `input`
The `input` field appears to concatenate many UI elements, where each element includes:
- visible text
- element type (`text`, `button`, `checked checkbox`, `unchecked radio button`, etc.)
- bounding boxes / coordinates
- font size
- foreground color
- background color

Example patterns visible in the dataset page:
- `"Yes , that's fine"|"button"|...|"Vida Loca, (RGB: 79, 133, 13)"|"White, (RGB: 255, 255, 255)"`
- `"Only necessary"|"button"|...`
- `"Accept all"|"button"|...`

#### `output`
The `output` field appears to contain repeated triples per UI element:
- deception label: `non-deceptive`, `forced-action`, `interface-interference`, `sneaking`, etc.
- subtype / finer tag such as `not-applicable`, `forced-action`, `nudge`, `hidden-subscription`, `trick-wording`, `disguised-ads`, `hidden-costs`
- natural-language explanation

Example patterns visible in the page:
- `"forced-action"|"forced-action"|"... doesn't offer a direct way to reject cookies ..."`
- `"interface-interference"|"nudge"|"... button is visually more prominent ..."`
- `"sneaking"|"hidden-subscription"|"Offers a discount in exchange for signing up ..."`
- `"sneaking"|"hidden-costs"|"Pricing information that could be misleading ..."`

#### `file_url`
This looks like a source identifier for the originating webpage snapshot or parsed CSV.

### Label space visible from samples
The public viewer shows at least these labels/subtypes:
- `non-deceptive`
- `forced-action`
- `interface-interference`
- `sneaking`
- `not-applicable`
- `nudge`
- `hidden-subscription`
- `hidden-costs`
- `disguised-ads`
- `trick-wording`

### Strengths
- Rich UI-grounded serialized context.
- Contains explanations, not just labels.
- Useful for learning element-level dark-pattern detection in cookies, banners, ads, pricing, and sign-up prompts.

### Limitations
- Highly synthetic by dataset title.
- Serialized OCR format is noisy and long.
- Not naturally aligned to chat-format SFT and must be normalized.
- Not a ready-made action trajectory dataset for RL.

### Best use in your pipeline
- Use as augmentation.
- Convert into consumer-side “analyze this UI text / element list” supervision.
- Derive designer seeds from explanations and element tags, not full trajectories.

---

## 2) apart/darkbench

### Source
- Hugging Face dataset: `apart/darkbench`
- Public README metadata visible on the dataset card.

### Card metadata visible in README
- **License:** `mit`
- **Task categories:** `question-answering`, `text-generation`, `text-classification`, `summarization`
- **Language:** `en`
- **Pretty name:** `DarkBench`
- **Size category:** `n<1K`
- **Config:** `default`
- **Train file:** `darkbench.jsonl`

### Overview
DarkBench is described as a benchmark for detecting dark design patterns in LLMs.
The README says it contains **660 prompts** across **six categories** and was used to evaluate **14 models** from major AI companies.

### Categories listed in the README
1. **Brand Bias**
2. **User Retention**
3. **Sycophancy**
4. **Anthropomorphism**
5. **Harmful Generation**
6. **Sneaking**

### Key findings reported in README
- Dark patterns appeared in **48%** of tested conversations on average.
- The most common pattern was **sneaking** at **79%**.
- The least common was **sycophancy** at **13%**.
- User retention and sneaking were prevalent across models.
- Claude-family models showed the lowest average dark-pattern rates.

### Methodology stated in README
The authors say they:
1. created precise descriptions for each dark pattern,
2. manually wrote adversarial prompts,
3. used few-shot prompting to generate additional prompts,
4. used multiple LLM annotators to evaluate responses.

### Likely row characteristics
From the README and benchmark framing, each example is likely prompt-centric rather than UI-OCR-centric.
This makes the dataset good for reasoning, prompt-conditioned classification, or generation tasks, but not directly for webpage-element parsing.

### Strengths
- Clear benchmark framing.
- Category diversity beyond simple dark vs non-dark.
- Strong fit for prompt reasoning and structured response generation.

### Limitations
- It is a benchmark, not a real browser trajectory dataset.
- Categories are LLM-behavior-centric, so some labels are not identical to webpage UI dark-pattern taxonomies.
- Likely requires schema inspection before exact field mapping.

### Best use in your pipeline
- Use for category-conditioned reasoning and harder consumer SFT prompts.
- Good for teaching structured flagging rationales.
- Useful as a robustness set, less useful as a direct source of UI action trajectories.

---

## 3) itsbaivab/mistral_dark_pattern_dataset

### Source
- Hugging Face dataset: `itsbaivab/mistral_dark_pattern_dataset`

### Visible schema from the dataset viewer
The public viewer shows three fields:
- `input`
- `output`
- `instruction`

### Field specification
#### `input`
A short text string, typically a single snippet from ecommerce or product UI copy.
Examples visible in the viewer include:
- `Hurry! Only 1 left in size M!!`
- `VIP Points!`
- `No thanks, I’d rather pay full price.`
- `Your order is reserved for 09:56 minutes!`
- `Shipping Insurance Remove Insurance`

#### `output`
Binary label with two visible values:
- `Dark`
- `Not Dark`

#### `instruction`
The viewer shows one fixed instruction template:
- `Tell me this string is showing a dark pattern or Not if it is showing a dark pattern then response with Dark and if it is not showing a dark pattern then response with Not Dark`

### Task type
This is a short-text binary classification dataset for dark-pattern recognition.
It is the most plug-and-play of the three for a first-stage classifier-style SFT.

### Patterns visible in positive examples
The examples shown on the public page include strings suggesting:
- scarcity / urgency (`Only 1 left`, `Limited quantity remaining`)
- social proof (`people are viewing this`, `just bought`)
- confirmshaming (`No thanks, I’d rather pay full price`)
- countdown / reservation pressure (`reserved for 09:56 minutes`)
- hidden recurring billing or membership traps
- add-on pressure / upsell language

### Strengths
- Very simple schema.
- Easy to convert into chat-format SFT.
- Strong starter dataset for dark vs non-dark calibration.

### Limitations
- No fine-grained taxonomy beyond binary labels.
- No UI layout, image, or element metadata.
- No long-horizon flow information.

### Best use in your pipeline
- Use as the main Stage 1 bootstrap for consumer recognition.
- Helpful for fast calibration before richer reasoning or self-play.
- Less useful for designer generation unless you synthesize structure on top.

---

## Cross-dataset comparison

| Dataset | Public schema clarity | Data style | Labels visible | Best role |
|---|---|---|---|---|
| `itsbaivab/mistral_dark_pattern_dataset` | High | Short text snippets | `Dark`, `Not Dark` | Recognition bootstrap |
| `apart/darkbench` | Medium | Benchmark prompts | 6 dark-pattern categories | Reasoning / robustness |
| `WIPI/deceptive_patterns_synthetic` | Medium-high from viewer | OCR-like serialized UI elements | deception label + subtype + explanation | UI-grounded augmentation |

---

## Recommended normalization target

To make the three datasets plug-and-play, normalize each example into a common intermediate schema like this:

```json
{
  "record_id": "string",
  "source_dataset": "string",
  "source_kind": "recognition|benchmark|ocr_ui",
  "text": "string",
  "label": "deceptive|non-deceptive",
  "subcategory": "string",
  "harm_types": ["string"],
  "explanation": "string",
  "consumer_sft": {
    "messages": [
      {"role": "user", "content": "string"},
      {"role": "assistant", "content": "string"}
    ]
  },
  "designer_seed": {
    "workflow": "signup|checkout|cancellation|unknown",
    "difficulty": "easy|medium|hard",
    "trap_elements": [
      {"element_hint": "string", "trap_cat": "string"}
    ],
    "goal_hint": "string"
  }
}
```

---

## Training implications

### What these three datasets are good for
- Early-stage SFT.
- Dark-pattern recognition.
- Structured reasoning and explanation formatting.
- Generating seed concepts for later environment episodes.

### What they do not fully provide
- Real multi-step browser trajectories.
- Ground-truth hidden environment state.
- Long-horizon cancellation-maze supervision.
- Direct RL-ready action sequences.

### Practical conclusion
These three datasets are enough for a credible Stage 1 bootstrap, especially if kept small and normalized carefully.
For later self-play and ELO training, your environment templates still need to provide the sequential structure and verifier-backed rewards.
