import os
import sys
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import viplan.code_helpers as code_helpers

class HuggingFaceLLM():
    def __init__(self, model_name: str, cache_dir: str = None, logger=DEFAULT_LOGGER, **kwargs):
        """
        Parameters:
        - model_name (str): The name of the model to load.
        - cache_dir (str, optional): Directory to cache the model.
        - logger (optional): Logger instance for logging messages.
        - kwargs: Additional keyword arguments:
            - use_flash_attention (bool): Whether to use flash attention. Default is False.
            - padding_side (str): Padding side for the tokenizer. Default is 'left'.
            - max_new_tokens (int): Maximum number of new tokens to generate. Default is 1500.
            - min_new_tokens (int): Minimum number of new tokens to generate. Default is 0.
            - temperature (float): Sampling temperature. Default is 1.0.
            - top_k (int): Top-k sampling. Default is 100.
            - top_p (float): Top-p (nucleus) sampling. Default is 0.8.
            - num_return_sequences (int): Number of sequences to return. Default is 1.
            - num_beams (int): Number of beams for beam search. Default is 1.
            - tokenizer_pad (str): Token to use for padding. Default is the tokenizer's eos_token.
            - tokenizer_padding_side (str): Padding side for the tokenizer. Default is 'left'.
        """
        self.model_name = code_helpers.get_model_name(model_name)
        self.logger = logger
        self.logger.info(f"Loading model {self.model_name}...")
        token = os.environ.get("HF_TOKEN", None) # Token needed for gated models (Llama 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Using bfloat16 results in RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16' with Deepseek Coder
        # (https://github.com/huggingface/diffusers/issues/3453)
        # Float16 on the other hand can result in RuntimeError: probability tensor contains either `inf`, `nan` or element < 0, likely due to overflow
        # (https://github.com/meta-llama/llama/issues/380)
        # self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if 'mistralai' in self.model_name or 'llama' in self.model_name:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

        # TODO this currently is not working, we need to figure out how to use flash attention on AMD GPUs
        # https://huggingface.co/docs/transformers/perf_infer_gpu_one?install=AMD
        use_flash_attention = kwargs.get("use_flash_attention", False)
        attn_implementation = "flash_attention_2" if use_flash_attention else None
        self.logger.debug(f"Using flash attention: {use_flash_attention}")

        if cache_dir is None:
            cache_dir = os.environ.get("HF_HOME", None)
        self.logger.debug(f"Huggingface cache dir: {cache_dir}")

        # Tokenizer params
        padding_side = kwargs.get("padding_side", "left")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir, torch_dtype=self.dtype, token=token, padding_side=padding_side)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=cache_dir, torch_dtype=self.dtype, token=token, device_map='auto', attn_implementation=attn_implementation)
        self.model.eval()

        self.logger.info(f"Model {self.model_name} loaded on device {self.device} with {self.dtype} precision.")

        # WizardCoder extra token fix
        # https://github.com/huggingface/transformers/issues/24843
        if "WizardLM" in self.model_name:
            special_token_dict = self.tokenizer.special_tokens_map
            self.tokenizer.add_special_tokens(special_token_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.logger.debug(f"Special tokens added to tokenizer: {special_token_dict}")

        # Generation params
        self.max_new_tokens = kwargs.get("max_new_tokens", 1500)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)
        
        # Trying to solve warning
        # UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
        temperature = kwargs.get("temperature", 1.0)
        if temperature == 0.0:
            self.model.generation_config.temperature=None
            self.model.generation_config.top_p=None
        self.temperature = temperature
        
        self.top_k = kwargs.get("top_k", 100)
        self.top_p = kwargs.get("top_p", 0.8)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.num_beams = kwargs.get("num_beams", 1)

        self.logger.debug(f"Generation params: max_new_tokens={self.max_new_tokens}, min_new_tokens={self.min_new_tokens}, temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}, num_return_sequences={self.num_return_sequences}, num_beams={self.num_beams}")

        # Possibly switch token to unk_token to avoid issues? (https://github.com/meta-llama/llama/issues/380)
        self.tokenizer.pad_token = kwargs.get("tokenizer_pad", self.tokenizer.eos_token)
        self.tokenizer.padding_side = kwargs.get("tokenizer_padding_side", "left")

    def get_completion(self, prompts, exclude_prompt=True, messages=None, **kwargs):
        start = time.perf_counter()
        
        # Ensure prompts and messages are lists
        if not isinstance(prompts, list):
            prompts = [prompts]
        if messages is None:
            messages = [None] * len(prompts)
        
        # Prepare messages and determine add_generation_prompt for each prompt
        processed_messages = []
        add_generation_prompts = []
        for prompt, message in zip(prompts, messages):
            if message is None:
                merge_system = True if "mistralai" in self.model_name.lower() else False
                message = code_helpers.get_messages(prompt, merge_system=merge_system)
            processed_messages.append(message)
            
            last_message = message[-1]
            if last_message["role"] == "assistant":
                add_generation_prompts.append(False)
            else:
                add_generation_prompts.append(True)
        
        # Tokenize the batched messages
        # ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 
        #'truncation=True' to have batched tensors with the same length.
        # => trying with padding=True
        inputs = self.tokenizer.apply_chat_template(processed_messages, add_generation_prompt=add_generation_prompts, padding=True, return_dict=True, return_tensors="pt").to(self.device)
        self.logger.debug(f"Input tokens: {self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")
        self.logger.debug(f"Number of input tokens: {len(inputs['input_ids'][0])}")
        
        # Generate outputs for the batched inputs
        if self.temperature > 0.0:
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, min_new_tokens=self.min_new_tokens,
                                          do_sample=True, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                                          num_return_sequences=self.num_return_sequences, num_beams=self.num_beams, pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, min_new_tokens=self.min_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, **kwargs)

        # Decode input prompts and outputs
        input_prompts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Adjust for chat-specific templates
        completions = []
        for i, (input_prompt, output) in enumerate(zip(input_prompts, decoded_outputs)):
            # Find the first occurrence of the last user message in the output
            last_user_message = input_prompt.rsplit("User:", 1)[-1].strip()  # Get the last user message
            self.logger.debug(f"last_user_message (i={i}): {last_user_message}")
            self.logger.debug(f"last_user_message in output: {last_user_message in output}")
            if last_user_message in output:
                # Remove everything up to and including the last user message
                cleaned_output = output.split(last_user_message, 1)[-1].strip()
            else:
                # Fallback: Remove the entire input prompt if it's present
                cleaned_output = output[len(input_prompt):].strip() if output.startswith(input_prompt) else output

            self.logger.debug(f"cleaned_output (i={i}): {cleaned_output}")
            completions.append(cleaned_output)
        
        end = time.perf_counter()
        self.logger.debug(f"Completions generated in {end-start:.2f}s")

        return completions

    def get_response(self, prompt, **kwargs):
        return self.get_completion(prompt, **kwargs)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run HuggingFace LLM with specified parameters.")
    parser.add_argument("--model_name", type=str, default="mistralai", help="Name of the model to load.")
    parser.add_argument("--prompt", type=str, nargs='+', default=["What is the capital of France?"], help="Prompt(s) to get completion for. Can be a single string or list of strings.")
    
    args = parser.parse_args()
    logger = ConsoleLogger(LogLevel.DEBUG)
    model = HuggingFaceLLM(args.model_name, logger=logger)
    print(model.get_completion(args.prompt))