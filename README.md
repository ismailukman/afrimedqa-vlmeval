# AfriMedQA with VLMEvalKit

This repo adds AfriMedQA-specific dataset/evaluation support on top of VLMEvalKit:
- `AfrimedQA` for MCQ evaluation.
- `AfrimedShortQA` for SAQ evaluation with automatic metrics.
- Dynamic TSV discovery via `--data-dir` (all `.tsv` files are registered at runtime).

## Quick Start

### 1) Environment setup

```bash
conda create -n afrimedqa_vlmeval python=3.10 -y
conda activate afrimedqa_vlmeval
pip install -r requirements.txt
```

### 2) Prepare data

Put one or more TSV files in a folder (example: `test_files/`), then pass that folder to `--data-dir`.

Recommended columns:
- `index`
- `image_path`
- `question`
- `question_type` (`MCQ` or `SAQ`, if mixed file)
- `A`, `B`, `C`, `D` (for MCQ)
- `answer` (or `correct_option` for MCQ)

Optional HTML to TSV conversion:

```bash
python html_to_tsv.py --html All_Pics_Questions/All_Pics_Questions.html --out AfrimedQA_en.tsv
```

### 3) Configure models, datasets, and judge API key

For config-driven runs, define both the evaluated model and dataset explicitly in a JSON file.

SAQ-specific key points:
- The evaluated VLM is defined under `model`.
- The evaluated dataset is defined under `data`.
- For SAQ, set `data.*.class` to `AfrimedShortQA`.
- The dataset name in `data.*.dataset` must match the discovered TSV name (filename without `.tsv`).

Example SAQ config (`configs/afrimedsaq.json` pattern):

```json
{
  "model": {
    "gemma-3-4b-it": {
      "class": "Gemma3",
      "model_path": "google/gemma-3-4b-it"
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

Create a project-root `.env` file for SAQ judge model access:

```bash
OPENAI_API_KEY=...
```

`run.py` calls `load_env()`, so this key is loaded automatically at runtime.

### 4) Run evaluation

`run.py` supports two entry modes:
- `--data` + `--model`
- `--config` (JSON config for model/data class wiring)

If `LMUData` is not set, `run.py` auto-sets it to `--data-dir` (or current directory).

Example (direct args):

```bash
python run.py --model Gemma3-12B --data-dir test_files 
```

Example (config-driven SAQ):

```bash
python run.py --config configs/afrimedsaq.json --data-dir test_files --judge gpt-4o-mini
```

Notes:
- `scripts/run.sh` is outdated and uses flags removed from `run.py` (`--lang`, `--question-type`, `--img_path`).
- In config mode, set each data entry `class` explicitly (`AfrimedQA` vs `AfrimedShortQA`) to choose the evaluation path.

## Evaluation behavior

### MCQ (`AfrimedQA`)
- Normalizes predictions to single-choice labels (`A`-`E`).
- Supports exact matching and OpenAI-judge flow (`chatgpt-0125`/`gpt-4-0125`), with fallback to exact matching when key/judge is unavailable.
- Writes:
  - `*_acc_all.csv` (aggregated accuracy)
  - `*_full_data.csv` (per-question results)

### SAQ (`AfrimedShortQA`)
- If `question_type` exists, evaluates only rows with `SAQ`.
- Computes automatic metrics including BLEU, ChrF++, SSA-COMET (if model is available), and DeepEval clinical-axis scoring.
- Writes:
  - `*_judged.xlsx` (row-level judged outputs)
  - `*_metrics.csv` (summary metrics)
  - `*_eval_report.md` (DeepEval report)

## Output layout

Outputs are stored under:
- `outputs/{model_name}/{run_id}/` (full run artifacts)
- `outputs/{model_name}/` (symlinks to latest files for easier access)

## Notes on Benchmarks, Models, and Versions

- [OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [Download All DETAILED Results](http://opencompass.openxlab.space/assets/OpenVLM.json)
- Check Supported Benchmarks: [VLMEvalKit Features -- Benchmarks (70+)](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)
- Check Supported LMMs: [VLMEvalKit Features -- LMMs (200+)](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)

### Transformers Version Recommendations

Some models require specific `transformers` versions:

- `transformers==4.33.0` -> Qwen, Monkey, InternLM-XComposer, mPLUG-Owl2, OpenFlamingo v2, IDEFICS, VisualGLM, etc.
- `transformers==4.36.2` -> Moondream1
- `transformers==4.37.0` -> LLaVA, ShareGPT4V, CogVLM, EMU2, Yi-VL, DeepSeek-VL, InternVL, etc.
- `transformers==4.40.0` -> IDEFICS2, Bunny-Llama3, MiniCPM-Llama3-V2.5, Phi-3-Vision, etc.
- `transformers==4.42.0` -> AKI
- `transformers==4.44.0` -> Moondream2, H2OVL
- `transformers==4.45.0` -> Aria
- `transformers==latest` -> LLaVA-Next, PaliGemma-3B, Chameleon, Ovis, Mantis, Idefics-3, GLM-4v-9B, etc.

### Torchvision Version

- Use `torchvision>=0.16` for Moondream series, Aria.

### Flash-attn Version

- Use `pip install flash-attn --no-build-isolation` for Aria.

### Demo

```python
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images?'])
print(ret)  # There are two apples in the provided images.
```
