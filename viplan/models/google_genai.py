import math
import os
from google import genai
from google.genai import types

from viplan import code_helpers


class GoogleAPIModel:
    def __init__(self, model_name: str, temperature: float, **kwargs):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.img_tag = "{image}"

    def _split_messages_by_role(self, messages):
        sysmsg = "\n\n".join([msg['content'] for msg in messages if msg['role'] == 'system'])
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        return sysmsg, user_messages

    def _get_yes_no_probabilities(self, raw_text, avg_logprob):
        text = raw_text.lower().strip().strip(".,!?\"'")
        prob = min(1, math.exp(avg_logprob)) if avg_logprob is not None else 0.5
        if "yes" in text:
            yes_prob = prob
            no_prob = 1 - prob
        elif "no" in text:
            no_prob = prob
            yes_prob = 1 - prob
        else:
            yes_prob = 0.5
            no_prob = 0.5
        return yes_prob, no_prob

    def generate(self, prompts, return_prompt=False, images=None, image_files=None, temperature=None, max_new_tokens=None, return_probs=False, **kwargs):
        if temperature is None:
            temperature = self.temperature
        if isinstance(prompts, str):
            prompts = [prompts]
        if images is not None:
            raise NotImplementedError("Inline image processing not implemented for GoogleAPIModel.")
        if return_prompt:
            raise NotImplementedError("Return prompt not implemented for GoogleAPIModel.")

        all_messages = [code_helpers.get_messages(prompt, img_tag=self.img_tag) for prompt in prompts]

        results = []
        for i, prompt in enumerate(all_messages):
            sysmsg, user_messages = self._split_messages_by_role(prompt)
            image = self.client.files.upload(file=image_files[i])
            contents_list = []
            for user_message in user_messages:
                contents_list.extend([image if content['type'] == 'image' else content['text'] for content in user_message])

            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=sysmsg,
                    temperature=temperature,
                    max_output_tokens=max_new_tokens,
                ),
                contents=contents_list,
            )
            completion = response.candidates[0]
            raw_text = ''.join([part.text for part in completion.content.parts]).strip()

            # If no logprobs are requested, return only the raw text
            if not return_probs:
                results.append(raw_text)
                continue

            yes_prob, no_prob = self._get_yes_no_probabilities(raw_text, completion.avg_logprobs)
            results.append((raw_text, yes_prob, no_prob))

        return results