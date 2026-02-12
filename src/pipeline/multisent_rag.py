from typing import List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)


class MultiSentRAG:
    """
    MultiSent-RAG reader model using instruction-tuned LLMs
    (Mistral or LLaMA-3) in a generative inference setup.

    Supports:
    - Few-shot binary classification (8 seen languages)
    - Zero-shot 3-class classification (4 unseen languages)
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        max_new_tokens: int = 100,
        temperature: float = 0.1,
    ):

        self.model_name = model_name

        # 4-bit quantization (as described in paper)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
            do_sample=True,
            temperature=temperature,
            return_full_text=False,
            max_new_tokens=max_new_tokens,
        )

    # -----------------------------
    # Prompt Builders
    # -----------------------------

    def build_fewshot_prompt(self, question: str) -> str:
        """
        Few-shot prompt (used for 8 seen languages).
        Matches the formulation described in the paper.
        """

        prompt = f"""System:
You are a multilingual sentiment analysis expert in English, French, Arabic,
Spanish, German, Hindi, Portuguese, and Italian. Your task is to classify
the emotional sentiment of a given text as:
- Positive: happiness, satisfaction, praise, optimism
- Negative: dissatisfaction, sadness, anger, frustration
Be strict and objective. Respond with one word only.

User: Here are some examples:

English
Text: "The movie was a total waste of time."
→ Negative

French
Text: "Le produit est arrivé rapidement et fonctionne bien."
→ Positive

Spanish
Text: "La película fue aburrida y una completa decepción."
→ Negative

Now classify this:

Text: "{question}"

What is the sentiment of the text?
"""
        return prompt.strip()

    def build_zeroshot_prompt(self, question: str) -> str:
        """
        Zero-shot prompt (used for unseen languages).
        Matches the zero-shot evaluation description in the paper.
        """

        prompt = f"""System:
You are a multilingual sentiment analysis expert.

Your task is to classify the emotional sentiment of a given text in any language as:
- Neutral: states facts or balanced opinions without strong emotion
- Positive: happiness, satisfaction, praise, optimism
- Negative: dissatisfaction, sadness, anger, frustration

Be strict and objective. Respond with one word only.

User:
Now classify this:

Text: "{question}"

What is the sentiment of the text?
"""
        return prompt.strip()

    # -----------------------------
    # Inference
    # -----------------------------

    def predict(self, texts: List[str], mode: str = "fewshot") -> List[str]:
    """
    Predict sentiment labels for a list of texts.

    Returns:
        List[str]  -> raw model outputs (e.g. "Positive")
    """

    predictions = []

    for text in texts:

        if mode == "fewshot":
            prompt = self.build_fewshot_prompt(text)
        else:
            prompt = self.build_zeroshot_prompt(text)

        output = self.generator(prompt)[0]["generated_text"].strip()

        # Return only first word (enforced by prompt)
        predictions.append(output.split()[0])

    return predictions
