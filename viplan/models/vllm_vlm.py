import requests
import json
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta
import os
import sys
import time
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams
from datetime import datetime, timedelta
import io
from PIL import Image
import base64
import io
from collections import namedtuple


from transformers import AutoProcessor, AutoModelForCausalLM

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import viplan.code_helpers as code_helpers

class VllmVLM():
    """
    Class for loading and using a VLLM Vision-Language Model (VLM) for image to text generation.
    """
    def __init__(self, model_name, device, dtype, img_tag="{image}", cache_dir=None, logger=DEFAULT_LOGGER, tensor_parallel_size=1, **kwargs):
        """
        Initialize the Huggingface VLM model.
        
        Parameters:
            model_name (str): The name of the model to load (from the Huggingface model hub).
            device (str): The device to use for inference.
            dtype (torch.dtype): The data type to use for inference.
            img_tag (str): The tag to use in the prompt for the image.
            cache_dir (str): The directory to use for caching model files. Defaults to None.
            logger (MultiLogger): The logger to use for logging. Defaults to a console logger.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.img_tag = img_tag
        self.logger = logger
        
        if cache_dir is None:
            cache_dir = os.environ.get("HF_HOME", None)
        self.logger.debug(f"Huggingface cache dir: {cache_dir}")
        print(f"Huggingface cache dir: {cache_dir}")
        
        self.logger.info(f"Loading model {self.model_name}")
        token = os.environ.get("HF_TOKEN", None)
        self.chat_special_tokens = None
        if "mistral-small" in self.model_name.lower(): # https://docs.vllm.ai/en/latest/getting_started/examples/vision_language_multi_image.html
            self.model = LLM(
                model=model_name, 
                tokenizer_mode="mistral",
                config_format="mistral",
                load_format="mistral",
                allowed_local_media_path="/scratch/cs/world-models/predicate_datasets",  # Fix: provide actual path instead of boolean
                dtype=self.dtype,
                # enforce_eager=True,
                disable_mm_preprocessor_cache=True,
                download_dir=cache_dir,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=65536,  # ↓ Reduce from 128000 to something manageable - NOT SURE THIS IS ALLOWED 
            )
            self.dictionary_size=131072 # number of tokens in dictionary, needs to be manually set based on the model
            self.tokenizer = self.model.get_tokenizer()
            self.chat_template = None
        elif "deepseek" in self.model_name.lower():
            self.model = LLM(
                model=model_name, 
                allowed_local_media_path="/scratch/cs/world-models/predicate_datasets",  # Fix: provide actual path instead of boolean
                dtype=self.dtype,
                # enforce_eager=True,
                # disable_mm_preprocessor_cache=True,
                download_dir=cache_dir,
                hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
                tensor_parallel_size=tensor_parallel_size,)
            self.chat_template_path = f'{PROJECT_ROOT}/data/chat_templates/template_deepseek_vl2.jija'
            self.chat_template = open(self.chat_template_path, 'r').read()
            self.tokenizer = self.model.get_tokenizer()
        elif "aya-vision" in self.model_name.lower():
            self.model = LLM(
                model=model_name, 
                allowed_local_media_path="/scratch/cs/world-models/predicate_datasets",  # Fix: provide actual path instead of boolean
                dtype=self.dtype,
                mm_processor_kwargs={"crop_to_patches": True},
                download_dir=cache_dir,
                tensor_parallel_size=tensor_parallel_size,)
            self.chat_template_path = None
            self.chat_template = None
            self.tokenizer = self.model.get_tokenizer()
        elif "molmo" in self.model_name.lower():
            self.model = LLM(
                model=model_name,
                allowed_local_media_path="/scratch/cs/world-models/predicate_datasets",  # Fix: provide actual path instead of boolean
                dtype=self.dtype,
                download_dir=cache_dir,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,)
            self.tokenizer = self.model.get_tokenizer()
            self.chat_template = None
        elif "internvl" in self.model_name.lower():
            self.model = LLM(
                model=model_name, 
                trust_remote_code=True,
                max_model_len=4096,
                dtype=self.dtype,
                mm_processor_kwargs={"max_dynamic_patch": 4},
                download_dir=cache_dir,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=True # experimental, to make InternVL fit in 2 H200 with 141GB each
            )
            self.chat_template_path = None
            self.chat_template = None
            self.tokenizer = self.model.get_tokenizer()
        else:
            raise ValueError(f'{self.model_name.lower()} not implemented')
        

        self.logger.info(f"Model {self.model_name} loaded on device {self.device} with {self.dtype} precision.")

        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.num_beams = kwargs.get("num_beams", 1)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 1024)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)

    def _format_messages(self, messages, images):
        all_messages_with_images_url = []
        for i, message in enumerate(messages):
            new_message = []
            for turn in message:
                if turn['role'] != 'user':
                    new_message.append(turn)
                else:
                    new_turn = {'role':'user', 'content':[]}
                    for one_content in turn['content']:
                        if one_content['type'] != 'image':
                            new_turn['content'].append(one_content)
                        else:
                            one_new_content = {
                                'type':'image_url',
                                'image_url':{'url': images[i]}
                            }
                            new_turn['content'].append(one_new_content)
                    new_message.append(new_turn)
            all_messages_with_images_url.append(new_message)
        return all_messages_with_images_url

    def format_messages(self, messages, images):
        all_messages = self._format_messages(messages, images)
        return all_messages
        

    def generate(self, prompts, return_prompt=False, images=None, image_files=None, temperature=None, max_new_tokens=None, return_probs=False, **kwargs):
        """
        Generate text from the given prompts and images.
        
        Parameters:
            prompts (str or list): The prompts to generate text for. A list of prompts can be provided for batch generation.
            return_prompt (bool): Whether to return the prompt with the generated text. Defaults to False.
            images (list): A list of PIL images to use for the prompts. Defaults to None.
            image_files (str or list): The paths to the images to use for the prompts. Provide this only if images is None and the images have to be directly loaded from files. If images is provided, this is ignored. Defaults to None.
            temperature (float): The temperature to use for sampling. Defaults to None, in which case the default temperature provided at initialization is used.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to None, in which case the default value provided at initialization is used.
            return_probs (bool): Whether to return the probabilities of the output tokens. Currently only works for binary classification tasks and returns the logit probabilities of the "Yes" and "No" tokens. Defaults to False.
            
            If other keyword arguments are provided, they are passed to the model for generation.
            
        Returns:
            list: The generated text for each prompt. Includes the probability of the "Yes" and "No" tokens if return_probs is True.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if temperature is None:
            temperature = self.temperature
            
        def format_image_path(path: str):
            with Image.open(path).convert("RGB") as img:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data_uri = f'data:image/png;base64,{img_base64}'
            return image_data_uri

        def format_image(img):
            img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data_uri = f'data:image/png;base64,{img_base64}'
            return image_data_uri
        
        if images is None and image_files is not None:
            if isinstance(image_files, list):
                images = [format_image_path(path) for path in image_files] if isinstance(image_files[0], str) else image_files
            else:
                images = [format_image_path(image_files)] * len(prompts)  # Broadcast a single image to all prompts
        elif images is not None:
            if len(images) < len(prompts):
                images = images + [images[-1]] * (len(prompts) - len(images)) # Broadcast the last image to all remaining prompts
            images = [ format_image(img) for img in images ]
            
        if isinstance(prompts, str):
            prompts = [prompts]
        
        all_messages = [code_helpers.get_messages(prompt, img_tag=self.img_tag) for prompt in prompts]
        self.logger.debug(f"Messages: {all_messages}")
        all_messages_with_images_url = self.format_messages(all_messages, images)
        stop_token_ids = [self.tokenizer.eos_token_id]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens, 
            min_tokens=self.min_new_tokens, 
            n=self.num_return_sequences,
            best_of=self.num_return_sequences,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            logprobs=20, # self.dictionary_size would be the best choice. As of today (vllm 0.7.0), 20 is the maximum
            stop_token_ids=stop_token_ids
        )
        outputs = self.model.chat(all_messages_with_images_url, sampling_params=sampling_params,
                                  chat_template=self.chat_template)
        batch_decoded_outputs = []
        for output in outputs:
            decoded_output = output.outputs[0].text
            batch_decoded_outputs.append(decoded_output)
        if return_probs:
            yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
            batch_probs = []
            first_token_index = 0
            for i in range(len(prompts)):
                from math import exp
                Prob = namedtuple("Prob", ["logprob"])
                yes_prob = float(exp(outputs[i].outputs[0].logprobs[first_token_index].get(yes_token_id, Prob(logprob=-10**27) ).logprob))
                no_prob = float(exp(outputs[i].outputs[0].logprobs[first_token_index].get(no_token_id,  Prob(logprob=-10**27)).logprob))
                batch_probs.append((batch_decoded_outputs[i], yes_prob, no_prob))
            return batch_probs
        
        return batch_decoded_outputs
