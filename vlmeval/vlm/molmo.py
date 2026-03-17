import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

TYPE_PROMPTS = {
    'Y/N':'vqa2:',
    'VQA':'vqa2:',
    'MCQ':'a_okvqa_mc:',
}

DATASET_PROMPTS = {
    'AI2D_TEST':'ai2_diagram:',
    'AI2D_TEST_NO_MASK':'ai2_diagram:',
    'COCO_VAL':'coco_captioning:',
    'ChartQA_TEST':'chart_qa:',
    'ChartQA_VAL':'chart_qa:',
    'DocVQA_VAL':'doc_qa:',
    'DocVQA_TEST':'doc_qa:',
    'InfoVQA_TEST':'info_qa:',
    'InfoVQA_VAL':'info_qa:',
    'OCRVQA_TEST':'ocr_vqa:',
    'OCRVQA_TESTCORE':'ocr_vqa:',
    'ScienceQA_VAL':'science_qa:',
    'ScienceQA_TEST':'science_qa:',
    'TableVQABench':'tabwmp_da:',
    'TextVQA_VAL':'text_vqa:'
}


class molmo(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='allenai/Molmo-7B-D-0924', **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            import einops
        except Exception as e:
            logging.critical('Please install transformer and einops before using molmo.')
            raise e

        # transformers 5.x removed 'default' from ROPE_INIT_FUNCTIONS; add it back
        # so remote model code (e.g. Molmo2) that relies on it still works.
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            if 'default' not in ROPE_INIT_FUNCTIONS:
                def _default_rope(config, device=None, **kwargs):
                    import math
                    base = getattr(config, 'rope_theta', 10000.0)
                    head_dim = getattr(config, 'head_dim',
                                       config.hidden_size // config.num_attention_heads)
                    partial = getattr(config, 'partial_rotary_factor', 1.0)
                    dim = int(head_dim * partial)
                    inv_freq = 1.0 / (base ** (
                        torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim
                    ))
                    return inv_freq, 1.0
                ROPE_INIT_FUNCTIONS['default'] = _default_rope
        except ImportError:
            pass

        device_map = "auto" if '72b' in model_path.lower() else 'cuda'
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device_map)
        except ValueError:
            # Molmo2 and other custom-arch models have an auto_map but aren't registered
            # in AutoModelForCausalLM's registry. Load via the dynamic module directly.
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            auto_map = getattr(config, 'auto_map', {})
            # Try common Auto model keys in priority order
            model_key = next(
                (k for k in ('AutoModelForCausalLM', 'AutoModelForImageTextToText',
                              'AutoModel', 'AutoModelForVision2Seq')
                 if k in auto_map),
                None
            )
            assert model_key is not None, \
                f'No usable model class in auto_map {list(auto_map.keys())} for {model_path}'
            model_cls = get_class_from_dynamic_module(auto_map[model_key], model_path)
            self.model = model_cls.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device_map)

        # transformers 5.x ProcessorMixin.__init__ rejects unexpected kwargs that
        # older remote processor code (e.g. Molmo2) passes to super().__init__().
        # Patch it to accept all kwargs: pass valid ones to the original init,
        # then manually set the extras as attributes so subclass code can use them.
        import inspect
        from transformers.processing_utils import ProcessorMixin
        _orig_proc_init = ProcessorMixin.__init__
        _valid_proc_params = set(inspect.signature(_orig_proc_init).parameters) - {'self'}
        def _tolerant_proc_init(self_, *a, **kw):
            valid_kw = {k: v for k, v in kw.items() if k in _valid_proc_params}
            extra_kw = {k: v for k, v in kw.items() if k not in _valid_proc_params}
            _orig_proc_init(self_, *a, **valid_kw)
            for k, v in extra_kw.items():
                setattr(self_, k, v)
        ProcessorMixin.__init__ = _tolerant_proc_init
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        finally:
            ProcessorMixin.__init__ = _orig_proc_init
        self.kwargs = kwargs
        self.model_name = model_path
        # set default maximum number of crops to 36
        self.max_crops = kwargs.get('max_crops', 36)

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) in ['Y/N', 'MCQ', 'VQA']:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        prefix = None
        if dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            prompt = self.build_prompt_mcq_vqa(line)
        elif dataset in ['MathVista_MINI']:
            prompt = self.build_prompt_mathvista(line)
        elif dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            prompt = self.build_prompt_ai2d(line)
        elif dataset is not None and listinstr(list(DATASET_PROMPTS.keys()), dataset):
            prefix = DATASET_PROMPTS[dataset]  # rest of supervised datasets are in VQA format
            prompt = self.build_prompt_vqa(line, prefix)
        elif dataset is not None and listinstr(['MCQ'], DATASET_TYPE(dataset)):
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            message = MMMUDataset.split_MMMU(message)
        return message

    def build_prompt_mathvista(self, line):
        if line['question_type'] == 'multi_choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_ai2d(self, line):
        def option_is_abc(line):
            for cand in string.ascii_uppercase:
                if cand in line and not pd.isna(line[cand]):
                    # check if option is single letter
                    if not line[cand].strip().isalpha() or len(line[cand].strip()) > 1:
                        return False
            return True

        if line['abcLabel'] and option_is_abc(line):
            prompt = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                prompt += f'\n{item}'
            prompt = f"ai2_diagram_no_letter: {prompt}"
            # prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram_no_letter:')
        else:
            prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram:')
        return prompt

    def build_prompt_mcq_vqa(self, line):
        if line['question_type'] == 'multiple-choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_multiple_choice(self, line, prefix=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}: {item}'
        if prefix is not None:
            prompt = f"{prefix} {question}"
        elif hasattr(self, 'model') and not hasattr(self.model, 'generate_from_batch'):
            # Molmo2: use a plain instruction instead of Molmo1-specific task prefix
            prompt = f"{question}\nAnswer with only the letter of the correct option (A, B, C, D, or E)."
        else:
            prompt = f"{TYPE_PROMPTS['MCQ']} {question}"

        return prompt

    def build_prompt_vqa(self, line, prefix=None):
        question = line['question']
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['VQA']} {question}"
        else:
            prompt = f"{prefix} {question}"
        return prompt

    def generate_inner(self, message, dataset=None):
        from transformers import GenerationConfig
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        if image_path is not None:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images = [image]
        else:
            images = None

        tokenizer = self.processor.tokenizer

        # process the image and text
        # Molmo1 uses .process(); Molmo2 uses apply_chat_template via standard __call__
        max_crops = self.max_crops
        if hasattr(self.processor, 'process'):
            proc_kwargs = dict(text=prompt, images=images,
                               images_kwargs={"max_crops": max_crops})
            inputs = self.processor.process(**proc_kwargs)
            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        else:
            # Molmo2: build messages list and use apply_chat_template for proper formatting
            content = []
            if images is not None:
                content.append({"type": "image", "image": images[0]})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if hasattr(self.model, 'generate_from_batch'):
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=tokenizer
                )
                generated_tokens = output[0, inputs['input_ids'].size(1):]
            else:
                input_len = inputs['input_ids'].shape[1]
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    stop_strings=["<|endoftext|>", "<|im_end|>"],
                    tokenizer=tokenizer
                )
                generated_tokens = output[0, input_len:]

        # only get generated tokens; decode them to text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # AI2D: map direct answer to letter option
        if dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            # 'ai2_diagram_no_letter: Which of the following is the magma chamber?\nK\nB\nC\nH'
            if 'ai2_diagram_no_letter' in prompt:
                options = prompt.split('\n')[1:]
                answer = options.index(generated_text)
                generated_text = chr(answer + ord('A'))

        # print(dataset, prompt, generated_text, inputs['images'].size()) # uncomment to debug

        return generated_text
