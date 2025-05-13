from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import torch
import numpy as np
import logging
import torch.nn.functional as F
import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import List, Dict

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_helpers import get_messages

class MolmoModel(object):
    def __init__(self, model_name, device, dtype, img_tag="{image}", cache_dir=None, **kwargs):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.img_tag = img_tag
        print(f"Cache dir: {cache_dir}")
        # use_flash_attention = kwargs.get("use_flash_attn", True)
        use_flash_attention = False # MolmoForCausalLM does not support Flash Attention 2.0 yet
        attn_implementation = "flash_attention_2" if use_flash_attention else None

        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True, 
                                                                       device_map='auto', cache_dir=cache_dir, attn_implementation=attn_implementation, trust_remote_code=True)
        self.model.eval()
        print(f"Model loaded on device: {self.model.device}")

        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.num_beams = kwargs.get("num_beams", 1)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 1024)
        self.min_new_tokens =    kwargs.get("min_new_tokens", 0)
        
        if self.temperature == 0.0:
            self.model.generation_config.temperature=None
            self.model.generation_config.top_p=None
            self.model.generation_config.top_k=None
        
        self.processor.tokenizer.padding_side = "left"


    def process_batch(
        self,
        processor: AutoProcessor,
        texts: List[str],
        images_list: List[List[Image.Image]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process in batch.
        
        Args:
            processor: The original processor.
            texts: List of text inputs
            images_list: List of lists containing PIL images.
            
        Returns:
            Dict with padded input_ids, images, image_input_idx, image_masks.
        """
        batch_size = len(texts)
        tokens_list = []
        for text in texts:
            tokens = processor.tokenizer.encode(" " + text, add_special_tokens=False)
            tokens_list.append(tokens)
        images_arrays_list = []
        image_idxs_list = []
        for images in images_list:
            if images:
                image_arrays = []
                for image in images:
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                        image = ImageOps.exif_transpose(image)
                        image_arrays.append(np.array(image))
                    else:
                        assert len(image.shape) == 3 and image.shape[-1] == 3
                        image_arrays.append(image.astype(np.uint8))
                images_arrays_list.append(image_arrays)
                image_idx = [-1] * len(image_arrays)
                image_idxs_list.append(image_idx)
            else:
                images_arrays_list.append(None)
                image_idxs_list.append(None)
        images_kwargs = {
            "max_crops": 12,
            "overlap_margins": [4, 4],
            "base_image_input_size": [336, 336],
            "image_token_length_w": 12,
            "image_token_length_h": 12,
            "image_patch_size": 14,
            "image_padding_mask": True,
        }
        outputs_list = []
        for i in range(batch_size):
            tokens = tokens_list[i]
            images = images_arrays_list[i]
            image_idx = image_idxs_list[i]
            out = processor.image_processor.multimodal_preprocess(
                images=images,
                image_idx=image_idx,
                tokens=np.asarray(tokens).astype(np.int32),
                sequence_length=1536,
                image_patch_token_id=processor.special_token_ids["<im_patch>"],
                image_col_token_id=processor.special_token_ids["<im_col>"],
                image_start_token_id=processor.special_token_ids["<im_start>"],
                image_end_token_id=processor.special_token_ids["<im_end>"],
                **images_kwargs,
            )
            outputs_list.append(out)

        batch_outputs = {}
        for key in outputs_list[0].keys():
            tensors = [torch.from_numpy(out[key]) for out in outputs_list]
            batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=-1
            )
        bos = processor.tokenizer.bos_token_id or processor.tokenizer.eos_token_id
        batch_outputs["input_ids"] = torch.nn.functional.pad(
            batch_outputs["input_ids"], (1, 0), value=bos
        )
        if "image_input_idx" in batch_outputs:
            image_input_idx = batch_outputs["image_input_idx"]
            batch_outputs["image_input_idx"] = torch.where(
                image_input_idx < 0, image_input_idx, image_input_idx + 1
            )
        return batch_outputs
        
    def generate(self, prompts, return_prompt=False, images=None, image_files=None, temperature=None, max_new_tokens=None, return_probs=False, verbose=False):
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if verbose: print('\n\nimage_files: ', image_files)
        if image_files is not None:
            if isinstance(image_files, list):
                images = [[Image.open(path)] for path in image_files] if isinstance(image_files[0], str) else image_files
            else:
                images = [[Image.open(image_files)]] * len(prompts)  # Broadcast a single image to all prompts
                
        if images is not None:
            if type(images[0]) != list:
                images = [[images[i]] for i in range(len(images))]
                
        if verbose: print('\nimages:', images)
        inputs = self.process_batch(processor=self.processor, texts=prompts, images_list=images)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if verbose: print('inputs: ', inputs)
        if verbose: print('self.processor.tokenizer.eos_token: ', self.processor.tokenizer.eos_token)
        if verbose: print('self.processor.tokenizer.pad_token_id: ', self.processor.tokenizer.pad_token_id)
        if self.dtype==torch.bfloat16: 
            inputs["images"] = inputs["images"].to(torch.bfloat16)
        # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        if True:
            if temperature == 0:
                if verbose: print('min_new_token:', self.min_new_tokens)
                outputs = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(
                                    max_new_tokens=self.max_new_tokens,
                                    min_new_tokens=self.min_new_tokens,
                                    temperature=temperature+0.001,
                                    top_k=self.top_k,
                                    top_p=self.top_p,
                                    num_beams=self.num_beams,
                                    num_return_sequences=self.num_return_sequences,
                                    pad_token_id=self.processor.tokenizer.pad_token_id,
                                    output_scores=True,
                                    output_logits=True,
                                    return_dict_in_generate=True, 
                                    return_past_key_values = False,
                                    do_sample=False
                                    ),
                    tokenizer=self.processor.tokenizer
            )          
            else:                               
                outputs = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(# stop_strings=['<system>', 'user'],# self.processor.tokenizer.eos_token, # "<|endoftext|>",
                                    max_new_tokens=self.max_new_tokens,
                                    min_new_tokens=self.min_new_tokens,
                                    temperature=temperature,
                                    top_k=self.top_k,
                                    top_p=self.top_p,
                                    num_beams=self.num_beams,
                                    num_return_sequences=self.num_return_sequences,
                                    pad_token_id=self.processor.tokenizer.pad_token_id,
                                    output_scores=True,
                                    output_logits=True,
                                    return_dict_in_generate=True, 
                                    return_past_key_values = False,
                                    do_sample=True
                                    ),
                    tokenizer=self.processor.tokenizer
                )
                pass
        if verbose: print('output: ', outputs)
        
        batch_decoded_outputs = []
        for i in range(len(prompts)):
            if verbose: print('outputs.sequences[i]: ', outputs.sequences[i])
            output_tokens = outputs.sequences[i][len(inputs['input_ids'][i]):] if not return_prompt else outputs.sequences[i]
            decoded_output = self.processor.tokenizer.decode(output_tokens, skip_special_tokens=True)
            if verbose: print('decoded_output: ', decoded_output)
            batch_decoded_outputs.append(decoded_output)
        if verbose: print('generated_texts: ', batch_decoded_outputs)
        

        if return_probs:
            yes_token_id = self.processor.tokenizer.encode(" Yes", add_special_tokens=False)[0]
            no_token_id = self.processor.tokenizer.encode(" No", add_special_tokens=False)[0]
            batch_probs = []
            for i in range(len(prompts)):
                if verbose: print('prompts[i]: ', prompts[i])
                if verbose: print('len(outputs.scores): ', len(outputs.scores))
                if verbose: print('outputs.scores[0].shape: ', outputs.scores[0].shape)
                if verbose: print('len(outputs.logits): ', len(outputs.logits))
                if verbose: print('outputs.logits[0].shape: ', outputs.logits[0].shape)
                if verbose: print('encoded prompts[i]: ', self.processor.tokenizer.encode(prompts[i], add_special_tokens=False))
                if verbose: print('output_tokens: ', outputs.sequences[i][len(inputs['input_ids'][i]):])
                if verbose: print('batch_decoded_outputs[i]: ', batch_decoded_outputs[i])
                probs = torch.softmax(outputs.logits[0][i], dim=-1)  # Get the probabilities of the first output token
                if verbose: print('len(probs): ', len(probs))
                values, indices = torch.topk(probs, k=5)
                if verbose: print("Indices of top 5 values:", indices)
                if verbose: print("Top 5 values:", values)

                yes_prob = probs[yes_token_id].item()
                if verbose: print('insided yes_prob: ', yes_prob)
                no_prob = probs[no_token_id].item()
                if verbose: print('insided no_prob: ', no_prob)
                batch_probs.append((batch_decoded_outputs[i], yes_prob, no_prob))
                if verbose: print()
            return batch_probs
        
        return batch_decoded_outputs
