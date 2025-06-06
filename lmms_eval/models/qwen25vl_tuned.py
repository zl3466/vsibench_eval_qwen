import logging
import re
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

from datetime import timedelta

from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams

import os
import sys
sys.path.append('Qwen2-VL/')
sys.path.append('Qwen2-VL/qwen-vl-utils/src')
from qwen_vl_utils import process_vision_info

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
@register_model("qwen25vl_tuned")
class Qwen25VL_tuned(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        download_dir: str = None,
        modality: str = "image",
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: str = "1",
        max_frames_num: int = None,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        if download_dir is not None:
            self._model = LLM(
                self.path,
                download_dir=download_dir,
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=20480,
                gpu_memory_utilization=0.8
            )
        else:
            self._model = LLM(
                self.path,
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=20480,
                gpu_memory_utilization=0.8
            )
        self._processor = AutoProcessor.from_pretrained(self.path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        print("Done loading processor and tokenizer")
        # self.sampling_params = SamplingParams(temperature=0.8, max_tokens=1024, top_p=0.95)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=1024)

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL2, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        self._config = None

        self._device = "cuda"
        self._rank = 0
        self._world_size = 1

        self.modality = modality
        self.max_frames_num = max_frames_num

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if self.modality == "image":
                raise NotImplementedError("Image inference for Qwen2VL is not supported yet.")
            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"{video_path}",
                                # "max_pixels": 360 * 420,
                                # "nframes": self.max_frames_num,
                            },
                            {"type": "text", "text": f"{contexts}"},
                        ],
                    }
                ]
                if self.max_frames_num:
                    messages[0]['content'][0]['nframes'] = self.max_frames_num
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                _, video_inputs = process_vision_info(messages)
                generated_ids = self._model.generate(
                    {
                        "prompt": text,
                        "multi_modal_data": {
                            "video": video_inputs
                        },
                    },
                    sampling_params=self.sampling_params
                )
                output_text = generated_ids[0].outputs[0].text
            else:
                raise NotImplementedError
            print(text)
            print(f"output text:")
            print(output_text)
            if int(os.getenv("VSI_THOUGHT_PROCESS")) == 1:
                output_text = extract_answer(output_text)
                print(f"extracted answer:")
                print(output_text)
            res.append(output_text)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."
