import os, sys
import base64
import time
import numpy as np
from PIL import Image
from openai import OpenAI
from io import BytesIO

from viplan.code_helpers import get_messages
from typing import List, Dict, Tuple, Union

def get_completion(
    client,  # openai.Client
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 1024,
    temperature: float = 0.6,
    logprobs: bool = True,
    top_logprobs: int = 1,
):
    import math
    import time

    start = time.perf_counter()
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs if logprobs else None,
        n=1,
    )
    end = time.perf_counter()
    #print(f"Time taken for inference: {end - start:.2f}s")

    choice = response.choices[0]
    raw_text = choice.message.content.strip()

    # If no logprobs are requested, return only the raw text
    if not logprobs:
        return raw_text

    text = raw_text.lower().strip().strip(".,!?\"'")
    try:
        # Sum logprobs of the full generated response
        total_logprob = sum(t.logprob for t in choice.logprobs.content)
        prob = math.exp(total_logprob)

        # Check if "yes" or "no" is in the generated text
        if "yes" in text:
            yes_prob = prob
            no_prob = 1 - prob
        elif "no" in text:
            no_prob = prob
            yes_prob = 1 - prob
        else:
            yes_prob = 0.5
            no_prob = 0.5

        return (raw_text, yes_prob, no_prob)

    except Exception as e:
        print(f"Warning: could not extract logprobs — {e}")
        return (raw_text, 0.5, 0.5)


class OpenAIModel:
    def __init__(self, model_name="gpt-4.1-nano", temperature=0.0, **kwargs):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
    
    def _format_messages(self, messages, images):
        for msg_list, img in zip(messages, images):
            for msg in msg_list:
                if msg["role"] == "system" and isinstance(msg["content"], str):
                    msg["content"] = [{"type": "text", "text": msg["content"]}]
                elif msg["role"] == "user" and isinstance(msg["content"], list):
                    for i, content in enumerate(msg["content"]):
                        if content["type"] == "image":
                            base64_img = self._encode_image(img)
                            msg["content"][i] = {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_img}"
                                }
                            }
        return messages

    def _encode_image(self, image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate(self, prompts, return_prompt=False, images=None, image_files=None, return_probs=False, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        if image_files and not images:
            if isinstance(image_files, str):
                image_files = [image_files]
            images = [Image.open(p) for p in image_files]
        if images and len(images) != len(prompts):
            raise ValueError("Number of prompts and images must match.")

        all_messages = [get_messages(prompt, img_tag="{image}", merge_system=False) for prompt in prompts]
        all_messages = self._format_messages(all_messages, images)

        results = []
        for i, prompt in enumerate(all_messages):
            output = get_completion(
                client=self.client,
                messages=prompt,
                model=self.model_name,
                temperature=self.temperature,
                logprobs=return_probs,
            )
            results.append(output)
        return results
