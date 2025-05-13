import os
import re
import glob
import time
import torch

from typing import List, Dict, Tuple, Union
from mloggers import LogLevel, ConsoleLogger, FileLogger, MultiLogger

def get_unique_id(logger):
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    job_id = array_job_id if array_job_id is not None else os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        unique_id = f"{job_id}_{task_id}" if task_id is not None else job_id
        logger.info(f"Found SLURM job id: {unique_id}")
    else:
        import datetime
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"Using datetime as unique id: {unique_id}")
        
    return unique_id

def get_model_name(alias: str) -> str:
    """
    Returns the model name for a given alias.
    
    Parameters:
        alias (str): The alias of the model.
        
    Returns:
        str: The model name.
        
    Raises:
        ValueError: If the alias does not match any known model.
    """
    alias = alias.lower()
    # HuggingFace aliases
    if "deepseek" in alias or "deepcoder" in alias:
        return "deepseek-ai/deepseek-coder-33b-instruct"
    if "wizard" in alias:
        return "WizardLM/WizardCoder-33B-V1.1"
    if "mistral" in alias or "mixtral" in alias:
        if "7" in alias:
            return "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif "22" in alias:
            return "mistralai/Mixtral-8x22B-Instruct-v0.1"
        else:
            return "mistralai/Mixtral-8x7B-Instruct-v0.1" # Default to the smaller model
    if "llama" in alias:
        if "70" in alias:
            return "meta-llama/Meta-Llama-3.1-70B-Instruct" 
        else:
            return "meta-llama/Meta-Llama-3.1-8B-Instruct" # Default to the smaller model
    # OpenAI aliases
    if "gpt3" in alias or "gpt-3" in alias or "gpt3.5" in alias or "gpt-3.5" in alias:
        return "gpt-3.5-turbo"
    if "gpt4" in alias or "gpt-4" in alias:
        if "turbo" in alias:
            return "gpt-4-turbo"
        elif "mini" in alias:
            return "gpt-4o-mini"
        else:
            return "gpt-4o" # Default to the full model
    
    # VLM aliases
    if "llava" in alias:
        if "1.6" in alias:
            return "llava-hf/llava-v1.6-mistral-7b-hf"
        elif "1.5" in alias:
            return "llava-hf/llava-v1.5-7b-hf"
        else:
            return "llava-hf/llava-onevision-qwen2-7b-ov-hf" # Default to the newest model
        
    if "qwenvl" in alias:
        if "72" in alias:
            return "Qwen/Qwen2.5-VL-72B-Instruct"
        else:
            return "Qwen/Qwen2.5-VL-7B-Instruct" # Default to the smaller model
    
    # If no known alias is matched, raise an error
    valid_aliases = [
        "deepseek", "deepcoder", "wizard", "mistral", "mixtral", "llama",
        "gpt3", "gpt-3", "gpt3.5", "gpt-3.5", "gpt4", "gpt-4",
        "llava", "qwenvl",
    ]
    import warnings
    message = f"Unknown model alias: {alias}. Valid aliases are: {', '.join(valid_aliases)}."
    warnings.warn(message)
    return alias

def load_vlm(
        model_name: str,
        hf_cache_dir: str = None,
        logger: MultiLogger = None,
        device: str = "auto",
        dtype: str = "auto",
        use_flash_attn: bool = None,
        temperature: float = 0.0,
        **kwargs
    ):
    """
    Loads a model from HuggingFace or OpenAI using custom model classes.
    """
    # For now let's bypass the alias and use the full model name
    # model_name = get_model_name(model_name)
    if logger is None:
        logger = get_logger("info")
        logger.info(f"Loading model {model_name}...")
    if hf_cache_dir is None:
        hf_cache_dir = os.environ.get("HF_HOME", None)
        logger.debug(f"Using HF cache dir: {hf_cache_dir}")
    
    if use_flash_attn is None:
        if torch.cuda.is_available() and ('A100' in torch.cuda.get_device_name() or 'H100' in torch.cuda.get_device_name() or 'H200' in torch.cuda.get_device_name()):
            use_flash_attn = True
        else:
            use_flash_attn = False

    logger.info(f"Use flash attention: {use_flash_attn}")
    
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
    logger.info(f"Using GPU: {torch.cuda.get_device_name()}." if torch.cuda.is_available() else "Using CPU.")
    logger.info(f"Using dtype: {dtype}.")
    
    # Remove eventual duplicate args from kwargs
    if "cache_dir" in kwargs:
        kwargs.pop("cache_dir")
    if "temperature" in kwargs:
        kwargs.pop("temperature")
    if "device" in kwargs:
        kwargs.pop("device")
    if "dtype" in kwargs:
        kwargs.pop("dtype")
    if "use_flash_attn" in kwargs:
        kwargs.pop("use_flash_attn")
    if "logger" in kwargs:
        kwargs.pop("logger")
    
    # VLLM parallelism
    # https://docs.vllm.ai/en/latest/serving/distributed_serving.html
    tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
    if "tensor_parallel_size" in kwargs:
        kwargs.pop("tensor_parallel_size")
    
    if "gpt" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower() or "o4" in model_name.lower():
        from viplan.models import OpenAIModel
        model = OpenAIModel(model_name=model_name, temperature=temperature, **kwargs)
    elif "molmo" in model_name.lower():
        from viplan.models import MolmoModel
        model = MolmoModel(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=temperature, device=device, dtype=dtype, use_flash_attn=use_flash_attn, **kwargs)
    elif "phi" in model_name.lower():
        from viplan.models import PhiModel
        model = PhiModel(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=temperature, device=device, dtype=dtype, use_flash_attn=use_flash_attn, **kwargs)
    elif "gemma" in model_name.lower():
        from viplan.models import Gemma3Model
        model = Gemma3Model(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=temperature, device=device, dtype=dtype, use_flash_attn=use_flash_attn, **kwargs)
    elif "mistral-small" in model_name.lower():
        from viplan.models import VllmVLM
        model = VllmVLM(model_name, cache_dir=hf_cache_dir, logger=logger, tensor_parallel_size=tensor_parallel_size, temperature=temperature, device=device, dtype=dtype, use_flash_attn=use_flash_attn, **kwargs)
    elif "deepseek-vl2" in model_name.lower():
        from viplan.models import VllmVLM
        model = VllmVLM(model_name, cache_dir=hf_cache_dir, logger=logger, tensor_parallel_size=tensor_parallel_size, temperature=temperature, device=device, dtype=dtype, use_flash_attn=use_flash_attn, **kwargs)
    elif "internvl" in model_name.lower():
        from viplan.models import VllmVLM
        model = VllmVLM(model_name, cache_dir=hf_cache_dir, logger=logger, tensor_parallel_size=tensor_parallel_size, temperature=temperature, device=device, dtype=dtype, use_flash_attn=use_flash_attn, **kwargs)
    else:
        from viplan.models import HuggingFaceVLM
        model = HuggingFaceVLM(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=temperature, device=device, dtype=dtype, use_flash_attn=use_flash_attn, **kwargs)
    
    if logger is not None:
        logger.info(f"Loaded model {model_name}.")
    return model

def get_messages(prompt: str, splits: List[str] = ["system", "user", "assistant"], merge_system: bool = False, img_tag: str = "{image}") -> List[Dict[str, str]]:
    """
    Converts a prompt string into a list of messages for each split.
    
    Parameters:
        prompt (str): The prompt string.x
        splits (list[str]): A list of the splits to parse. Defaults to ["system", "user", "assistant"].
        merge_system (bool): Whether to merge the system messages into the user messages. Defaults to False.
        img_tag (str): The tag to use for images. Defaults to "{image}".
        
    Returns:
        list[dict[str, str]]: A dictionary of the messages for each split.
    """
    
    def _get_content(text: str, img_tag: str = "{image}"):
        if not img_tag in text:
            return text.strip()
        else:
            content = []
            for part in text.split(img_tag):
                if part:
                    content.append({"type": "text", "text": part.strip()})
                content.append({"type": "image"})
            return content[:-1]
    
    messages = []
    for split in splits:
        start_tag = f"<{split}>"
        end_tag = f"</{split}>"

        start_idx = prompt.find(start_tag)
        end_idx = prompt.find(end_tag)
        if end_idx == -1:
            end_tag = f"<\\{split}>"
            end_idx = prompt.find(end_tag)
        
        # Skip if the split is not in the prompt (e.g. no system prompt)
        if start_idx == -1 and end_idx == -1:
            continue
        messages.append({
            "role": split,
            "content": _get_content(prompt[start_idx + len(start_tag):end_idx].strip(), img_tag=img_tag)
        })
    
    # If no splits at all, assume the whole prompt is a user message
    if len(messages) == 0:
        messages.append({
            "role": "user",
            "content": _get_content(prompt.strip(), img_tag=img_tag)
        })
        
    if merge_system:
        for i, message in enumerate(messages):
            if message["role"] == "system" and messages[i+1] and messages[i+1]["role"] == "user":
                if isinstance(messages[i+1]["content"], str) and isinstance(message["content"], str):
                    messages[i+1]["content"] = message["content"] + " " + messages[i+1]["content"]
                elif isinstance(messages[i+1]["content"], dict) and isinstance(message["content"], dict):
                    messages[i+1]["content"].update(message["content"])

        messages = [message for message in messages if message["role"] != "system"]

    return messages

def parse_prompt(
        file_path: os.PathLike,
        splits: List[str] = ["system", "user"],
        subs: Dict[str, str] = None
    ):
    """
    Parses a prompt file and returns a dictionary of the messages for each split.

    Parameters:
        file_path (os.PathLike): The path to the prompt file.
        splits (list[str]): A list of the splits to parse. Defaults to ["system", "user"].
        subs (dict[str, str]): A dictionary of substitutions to make in the prompt file. In the prompt file, the keys
                                should be surrounded by curly braces, e.g. {key}. Defaults to None.

    Returns:
        list[dict[str, str]]: A dictionary of the messages for each split.
    """

    with open(file_path, "r") as f:
        prompt = f.read()

    if subs is not None:
        for key, value in subs.items():
            if key not in prompt:
                raise ValueError(f"Key {key} not found in prompt file.")
            prompt = prompt.replace(f"{{{key}}}", value)
    
    return get_messages(prompt, splits)

def parse_output(output: str, answer_tags: List[str] = ["answer"]) -> Tuple[Union[str, Dict[str, str]], bool]:
    """
    Parses the output of a model contained within multiple XML-style answer tags.
    
    Parameters:
        output (str): The output string.
        answer_tags (list[str]): The tags to look for (without the brackets, which are automatically added). Defaults to ["answer"].

    Returns:
        dict[str, str]: A dictionary with the tags as keys and the content as values. If no tags are found, the full output is returned unchanged.
        bool: Whether any tags were found.
    """
    parsed_output = {}
    tags_found = False

    for answer_tag in answer_tags:
        start_tag = f"<{answer_tag}>"
        end_tag = f"</{answer_tag}>"

        start_idx = output.find(start_tag)
        end_idx = output.find(end_tag)
        if end_idx == -1:
            end_tag = f"<\\{answer_tag}>" # Try with a backslash
            end_idx = output.find(end_tag)

        if start_idx != -1 and end_idx != -1:
            parsed_output[answer_tag] = output[start_idx + len(start_tag):end_idx].strip()
            tags_found = True
        else:
            parsed_output[answer_tag] = None

    if not tags_found:
        # Return full output (without dict) for backwards compatibility when no tags are present
        return output.strip(), tags_found

    return parsed_output, tags_found

def get_completion(
    client, # openai.Client
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 1024,
    temperature: float = 0.6,
    n: int = 1,
):

    start = time.perf_counter()
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )
    end = time.perf_counter()
    print(f"Time taken for inference: {end - start:.2f}s")
    if n==1:
        return response.choices[0].message.content.strip()
    else:
        return [choice.message.content.strip() for choice in response.choices]
    
def get_log_level(log_level: str) -> LogLevel:
    """
    Returns the log level for a given string.
    
    Parameters:
        log_level (str): The log level as a string.
        
    Returns:
        LogLevel: The log level.
        
    Raises:
        ValueError: If the log level is not recognized.
    """
    log_level = log_level.lower().strip()
    if log_level == "debug":
        return LogLevel.DEBUG
    if log_level == "info":
        return LogLevel.INFO
    if log_level == "warning":
        return LogLevel.WARNING
    if log_level == "error":
        return LogLevel.ERROR
    raise ValueError(f"Unknown log level: {log_level}. Valid log levels are: debug, info, warning and error.")

def get_logger(log_level: str, log_file: str = None) -> MultiLogger:
    """
    Returns a logger with the given log level and log file.
    
    Parameters:
        log_level (str): The log level as a string.
        log_file (str): The path to the log file. Defaults to None. If None, only console logging is used.
        
    Returns:
        MultiLogger: The logger.
    """
    log_level = get_log_level(log_level)
    console_logger = ConsoleLogger(default_priority=log_level)
    loggers = [console_logger]
    if log_file:
        file_logger = FileLogger(file_path=log_file, default_priority=log_level)
        loggers.append(file_logger)

    return MultiLogger(loggers, default_priority=log_level)