# AfriMedQA with VLMEvalKit (Quick Start)

The AfriMedQA dataset class extends VLMEvalKit's `ImageMCQDataset` to load a local TSV of questions and
evaluate with **exact matching** or an **LLM judge**. Evaluation support for SAQ is not included (**yet to decide on appropriate evaluaiton metrics**) 

------------------------------------------------------------------------

## Quick Start

### 1) Create New Virtual Environment with Conda and Install package requirements

``` bash
# Create a new conda env with Python 3.10

conda create -n afrimedqa_vlmeval python=3.10 -y
conda activate afrimedqa_vlmeval

# Install packages
pip install -r requirements.txt
# If converting HTML → TSV (e.g. using html_to_tsv.py), install the following packages:
pip install beautifulsoup4 lxml
```

### 2) Prepare the dataset (HTML → TSV)

Optional: If dataset is in **HTML** form, convert it to TSV with the
script below:

``` bash
python html_to_tsv.py   --html All_Pics_Questions/All_Pics_Questions.html   --out AfrimedQA.tsv
```

-   Defaults (omit flags if these match your layout):
    -   `--html All_Pics_Questions/All_Pics_Questions.html`
    -   `--out AfrimedQA.tsv`
-   The script:
    -   The script converts the AfriMedQA dataset from its original HTML format into a clean TSV file, extracting questions, options, and correct answers while embedding any images in base64 format for evaluation with VLMEvalKit.



### 3) Run an evaluation

#### Configuration File Setup

The evaluation commands rely on JSON files stored in the `configs/` directory. These files allow you to easily define the model path, and the dataset class you want to use without modifying the core Python code.

#### Example JSON Structure (`configs/afrimedsaq.json`)

```json
{
    "model": {
        "medgemma-27b-it": {
            "class": "Gemma3", 
            "model_path": "google/medgemma-27b-it",
            "use_vllm": true,
            "tensor_parallel_size": 2
        }
    },
    "data": {
        "ENGLISH_TEST__Sheet1": {
            "class": "AfrimedShortQA",
            "dataset": "ENGLISH_TEST__Sheet1"
        }
    }
}
```
Set LMUData to the appropriate directory (path to dataset directory) at the start of each command.

#### A. Short Answer Questions (SAQ)
Evaluates open-ended diagnostic reasoning on SAQ questions containing both text and images.

``` bash
LMUData=test_files python run.py --config configs/afrimedsaq.json --judge gpt-5.2
```

#### B. Multimodal MCQ
Evaluates vision-language models on MCQ questions containing both text and images.

``` bash
LMUData=test_files python run.py --config configs/gemma_mcq_baseline.json
```


#### C. Text-Only MCQ
Evaluates models using the text-only baseline.

``` bash
LMUData=test_files python run.py --config configs/gemma_text_baseline.json
```


### 4) Outputs

-   Per-question predictions and hits\
-   Accuracy summary CSV: `_acc_all.csv`\
-   Full per-item results: `_full_data.csv`

------------------------------------------------------------------------

## About the `AfrimedQA` Class

-   **Loads local TSV** from `$LMUData/AfrimedQA.tsv` 
-   **Ensures model predictions** are converted into one of the valid answer choices (A, B, C, D, or E)
-   **Judging modes**
    -   Exact matching (default)
    -   LLM judge if an OpenAI key is set
-   **Reports**  test and validation accuracy and writes results to CSV file

------------------------------------------------------------------------

## Notes on Benchmarks, Models, and Versions

-   [**OpenVLM
    Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard):
    [**Download All DETAILED
    Results**](http://opencompass.openxlab.space/assets/OpenVLM.json).\
-   Check **Supported Benchmarks**: [VLMEvalKit Features -- Benchmarks
    (70+)](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)\
-   Check **Supported LMMs**: [VLMEvalKit Features -- LMMs
    (200+)](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)

### Transformers Version Recommendations

Some models require specific `transformers` versions: -
`transformers==4.33.0` → Qwen, Monkey, InternLM-XComposer, mPLUG-Owl2,
OpenFlamingo v2, IDEFICS, VisualGLM, etc.\
- `transformers==4.36.2` → Moondream1\
- `transformers==4.37.0` → LLaVA, ShareGPT4V, CogVLM, EMU2, Yi-VL,
DeepSeek-VL, InternVL, etc.\
- `transformers==4.40.0` → IDEFICS2, Bunny-Llama3, MiniCPM-Llama3-V2.5,
Phi-3-Vision, etc.\
- `transformers==4.42.0` → AKI\
- `transformers==4.44.0` → Moondream2, H2OVL\
- `transformers==4.45.0` → Aria\
- `transformers==latest` → LLaVA-Next, PaliGemma-3B, Chameleon, Ovis,
Mantis, Idefics-3, GLM-4v-9B, etc.

### Torchvision Version

-   Use `torchvision>=0.16` for Moondream series, Aria

### Flash-attn Version

-   Use `pip install flash-attn --no-build-isolation` for Aria

### Demo

``` python
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images?'])
print(ret)  # There are two apples in the provided images.
```

------------------------------------------------------------------------

## Machine Translation Evaluation (`mt_eval/`)

The `mt_eval` pipeline evaluates translation quality for AfriMedQA SAQ question-answer pairs from English into African languages (e.g. Twi). It scores translations using **ChrF** and **SSA-COMET**.

### Running an MT Evaluation (example)

```bash
python -m mt_eval.run mt_eval/configs/en_twi_vllm.json
```

Replace the config path with whichever backend you want to use.

### Config Structure

All configs live in `mt_eval/configs/` and share the same `data` block:

```json
{
    "data": {
        "source_file": "test_files/ENGLISH_TEST__Sheet1.tsv",
        "target_file": "test_files/TWI_TEST__Sheet1.tsv",
        "source_lang": "English",
        "target_lang": "Twi"
    },
    "model": { ... },
    "output_dir": "outputs/mt_eval"
}
```

The `source_file` and `target_file` should be TSV files with parallel question-answer pairs.

### Example Model Backends

| Config file | Backend | Example model |
|---|---|---|
| `en_twi_vllm.json` | vLLM | `Qwen/Qwen2.5-7B-Instruct` |
| `en_twi_gemini.json` | Gemini API | `gemini-2.0-flash` |
| `en_twi_gemma.json` | HuggingFace (chat) | `google/gemma-3-4b-it` |
| `en_twi_nllb.json` | Seq2Seq (NLLB) | `facebook/nllb-200-3.3B` |

**vLLM config example:**
```json
{
    "model": {
        "type": "vllm",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_tokens": 512,
        "temperature": 0.0,
        "vllm_kwargs": { "gpu_memory_utilization": 0.9, "dtype": "bfloat16" }
    }
}
```

**Gemini config example** (requires `GEMINI_API_KEY` env variable or set `"api_key"` in config):
```json
{
    "model": {
        "type": "gemini",
        "model_name": "gemini-2.0-flash",
        "api_key": "GEMINI_API_KEY",
        "max_tokens": 512,
        "temperature": 0.0
    }
}
```

**NLLB (seq2seq) config example:**
```json
{
    "model": {
        "type": "seq2seq",
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "eng_Latn",
        "tgt_lang": "twi_Latn",
        "max_tokens": 512
    }
}
```

### Outputs

Results are written to `outputs/mt_eval/` (or the path set in `output_dir`):

- `{src}_{tgt}_{model}_results.csv` — per-sample translations with sentence-level ChrF and SSA-COMET scores
- `{src}_{tgt}_{model}_summary.csv` — corpus-level summary with mean ChrF (question, answer, combined) and SSA-COMET system scores

### Metrics

- **ChrF**: character n-gram F-score, reported at both sentence and corpus level
- **SSA-COMET**: reference-based neural MT metric fine-tuned for Sub-Saharan African languages
