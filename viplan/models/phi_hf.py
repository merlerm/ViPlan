import copy
import torch

from PIL import Image

from viplan import code_helpers
from viplan.models.huggingface_vlm import HuggingFaceVLM


class PhiModel(HuggingFaceVLM):

    def convert_system_to_user_messages(self, messages):
        messages = copy.deepcopy(messages)
        for i in range(len(messages)):
            if messages[i]['role'] == 'system':
                messages[i]['role'] = 'user'
        return messages

    # def convert_to_inline_images(self, messages):
    #     """
    #     Converts from {'role': ROLE, 'content': [{'type': 'image'}, {'type': 'text', 'text': 'TEXT'}]}
    #     to {'role': ROLE, 'content': "<image_1>TEXT"} as expected by Phi4's processor.
    #     """
    #     # Remove image tags
    #     messages = copy.deepcopy(messages)
    #     img_idx = 1
    #     for i in range(len(messages)):
    #         if isinstance(messages[i]['content'], list):
    #             message_content = ""
    #             text_content = [content for content in messages[i]['content'] if content['type'] == 'text']
    #             assert len(text_content) <= 1, 'Multiple text contents in one message, formatting not implemented'
    #             if len(text_content) == 1:
    #                 message_content += text_content[0]['text']
    #             image_content = [content for content in messages[i]['content'] if content['type'] == 'image']
    #             assert len(image_content) <= 1, 'Multiple images in one message, formatting not implemented'
    #             if len(image_content) == 1:
    #                 message_content = f'<|image_{img_idx}|>' + message_content
    #                 img_idx += 1
    #             messages[i]['content'] = message_content
    #     return messages
    
    def convert_to_inline_images(self, messages):
        """
        Converts messages like:
          {'role': ROLE, 'content': [
              {'type': 'text',  'text': 'Hello, '},
              {'type': 'image'},
              {'type': 'text',  'text': 'world!'},
              {'type': 'image'},
              {'type': 'text',  'text': 'Bye.'},
          ]}
        into:
          {'role': ROLE, 'content': "Hello, <|image_1|>world!<|image_2|>Bye."}
        """
        messages = copy.deepcopy(messages)
        img_idx = 1
    
        for msg in messages:
            cont = msg.get('content')
            # only transform if content is a list of fragments
            if isinstance(cont, list):
                parts = []
                for fragment in cont:
                    t = fragment.get('type')
                    if t == 'text':
                        parts.append(fragment.get('text', ''))
                    elif t == 'image':
                        parts.append(f'<|image_{img_idx}|>')
                        img_idx += 1
                    else:
                        raise ValueError(f"Unsupported content type: {t!r}")
                msg['content'] = ''.join(parts)
    
        return messages

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
            if isinstance(image_files, list):
                # Ensure each image path corresponds to a prompt, or broadcast to all prompts if only one image
                images = [Image.open(path) for path in image_files] if isinstance(image_files[0], str) else image_files
            else:
                images = [Image.open(image_files)] * len(prompts)  # Broadcast a single image to all prompts
                
        elif images is not None and len(images) < len(prompts):
            images = images + [images[-1]] * (len(prompts) - len(images)) # Broadcast the last image to all remaining prompts
            
        if isinstance(prompts, str):
            prompts = [prompts]

        all_messages = [code_helpers.get_messages(prompt, img_tag=self.img_tag) for prompt in prompts]
        # According to https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/6, system messages are not supported with vision and speech inputs.
        # TODO: Should consecutive user messages be merged?
        self.logger.debug(f"All messages: {all_messages}")
        all_messages = [self.convert_system_to_user_messages(messages) for messages in all_messages]
        all_messages = [self.convert_to_inline_images(messages) for messages in all_messages]

        self.logger.debug(f"Messages: {all_messages}")
        batch_prompts = [self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in all_messages]
        for i, prompt in enumerate(batch_prompts):
            if prompt.endswith('<|endoftext|>'):
                batch_prompts[i] = prompt.rstrip('<|endoftext|>')
        inputs = self.processor(text=batch_prompts, images=images, return_tensors="pt", padding=True).to(self.device)
        # inputs['pixel_values'] = inputs['pixel_values'].type(self.dtype) if 'pixel_values' in inputs else None # Fix for llava onevision having float32 pixel values
        if temperature == 0:
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=self.min_new_tokens, 
                                            pad_token_id=self.processor.tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True, num_logits_to_keep=0, **kwargs)
        else:
            outputs = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_k=self.top_k, top_p=self.top_p, 
                                            num_beams=self.num_beams, num_return_sequences=self.num_return_sequences, max_new_tokens=max_new_tokens, 
                                            min_new_tokens=self.min_new_tokens, pad_token_id=self.processor.tokenizer.pad_token_id, 
                                            return_dict_in_generate=True, output_scores=True, num_logits_to_keep=0, **kwargs)

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