import os
import sys
import time
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, Gemma3ForConditionalGeneration

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import viplan.code_helpers as code_helpers

def format_messages(messages, image_paths):
    for msg_list, img_path in zip(messages, image_paths):
        for msg in msg_list:
            if msg["role"] == "system" and isinstance(msg["content"], str):
                # Convert system content to the required format
                msg["content"] = [{"type": "text", "text": msg["content"]}]
            elif msg["role"] == "user" and isinstance(msg["content"], list):
                # Replace image placeholder with actual image path
                for content in msg["content"]:
                    if content["type"] == "image":
                        content["image"] = img_path  
    return messages

class Gemma3Model():
    """
    Class for loading and using a Huggingface Vision-Language Model (VLM) for image to text generation.
    """
        
    def __init__(self, model_name, device, dtype, img_tag="{image}", cache_dir=None, logger=DEFAULT_LOGGER, **kwargs):
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
        
        self.logger.info(f"Loading model {self.model_name}")
        use_flash_attention = kwargs.get("use_flash_attn", True)
        attn_implementation = "flash_attention_2" if use_flash_attention else None
        self.logger.debug(f"Using flash attention: {use_flash_attention}")
        
        token = os.environ.get("HF_TOKEN", None)
        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir, token=token, trust_remote_code=True, use_fast=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True, token=token,
                                                device_map='auto', cache_dir=cache_dir, attn_implementation=attn_implementation, trust_remote_code=True)
        self.model.eval()
        self.logger.info(f"Model {self.model_name} loaded on device {self.device} with {self.dtype} precision.")

        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", None)
        self.top_p = kwargs.get("top_p", None)
        self.num_beams = kwargs.get("num_beams", 1)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 1024)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)
        
        if self.temperature == 0.0:
            self.model.generation_config.temperature=None
            self.model.generation_config.top_p=None
            self.model.generation_config.top_k=None
        
        self.processor.tokenizer.pad_token = kwargs.get("tokenizer_pad", self.processor.tokenizer.eos_token) # TODO verify if other choices are better
        self.processor.tokenizer.padding_side = kwargs.get("tokenizer_padding_side", "left")

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
        
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        if images is None and image_files is not None:
            images = image_files  # Directly assign the list of image paths

        elif images is not None and len(images) < len(prompts):
            images = images + [images[-1]] * (len(prompts) - len(images))  # Broadcast the last image path to all remaining prompts
            
        if isinstance(prompts, str):
            prompts = [prompts]

        all_messages = [code_helpers.get_messages(prompt, img_tag=self.img_tag, merge_system=False) for prompt in prompts]
        all_messages = format_messages(all_messages, images)
        self.logger.debug(f"Messages: {all_messages}")

        inputs = self.processor.apply_chat_template(all_messages, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True, padding=True).to(self.device)
        if temperature == 0:
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=self.min_new_tokens, 
                                            pad_token_id=self.processor.tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True, **kwargs)
        else:
            outputs = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_k=self.top_k, top_p=self.top_p, 
                                            num_beams=self.num_beams, num_return_sequences=self.num_return_sequences, max_new_tokens=max_new_tokens, 
                                            min_new_tokens=self.min_new_tokens, pad_token_id=self.processor.tokenizer.pad_token_id, 
                                            return_dict_in_generate=True, output_scores=True, **kwargs)

        batch_decoded_outputs = []
        for i in range(len(prompts)):
            output_tokens = outputs.sequences[i][len(inputs['input_ids'][i]):] if not return_prompt else outputs.sequences[i]
            decoded_output = self.processor.tokenizer.decode(output_tokens, skip_special_tokens=True)
            batch_decoded_outputs.append(decoded_output)

        if return_probs:
            yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
            batch_probs = []
            for i in range(len(prompts)):
                probs = torch.softmax(outputs.scores[0][i], dim=-1)  # Get the probabilities of the first output token
                yes_prob = probs[yes_token_id].item()
                no_prob = probs[no_token_id].item()
                batch_probs.append((batch_decoded_outputs[i], yes_prob, no_prob))
            return batch_probs
        
        return batch_decoded_outputs
