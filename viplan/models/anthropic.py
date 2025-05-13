import anthropic
import base64
import io
import os

from PIL import Image
from viplan import code_helpers


class AnthropicAPIModel:
    def __init__(self, model_name: str, temperature: float, **kwargs):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.img_tag = "{image}"
        self.max_new_tokens = kwargs.get("max_new_tokens", 1024)

    def _split_messages_by_role(self, messages):
        sysmsg = "\n\n".join([msg['content'] for msg in messages if msg['role'] == 'system'])
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        return sysmsg, user_messages

    def _encode_image(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _insert_image(self, messages, image):
        for msg in messages:
            if msg["role"] == "user":
                for content in msg["content"]:
                    if content["type"] == "image":
                        content["source"] = {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": self._encode_image(image),
                        }
        return messages

    def generate(self, prompts, return_prompt=False, images=None, image_files=None, temperature=None, max_new_tokens=None, return_probs=False, **kwargs):
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if return_prompt:
            raise NotImplementedError("Return prompt not implemented for AnthropicAPIModel.")
        if image_files and not images:
            if isinstance(image_files, str):
                image_files = [image_files]
            images = [Image.open(p) for p in image_files]
        all_messages = [code_helpers.get_messages(prompt, img_tag=self.img_tag) for prompt in prompts]

        results = []
        for i, prompt in enumerate(all_messages):
            sysmsg, user_messages = self._split_messages_by_role(prompt)
            messages_with_image = self._insert_image(user_messages, images[i]) if images else user_messages

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_new_tokens,
                system=sysmsg,
                temperature=temperature,
                messages=messages_with_image,
            )
            raw_text = response.content[0].text
            if not return_probs:
                results.append(raw_text)
                continue

            # Probabilities are not available in the response
            yes_prob, no_prob = 0.5, 0.5
            results.append((raw_text, yes_prob, no_prob))

        return results