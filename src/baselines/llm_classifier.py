from typing import List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


class LLMClassifier:
    """
    Quantized inference-only classifier for decoder-based LLMs
    (BLOOMZ, LLaMA-3, Mistral) used in baseline experiments.

    Models are treated as sequence classifiers rather than text generators.
    """

    def __init__(self, model_name: str, num_labels: int = 2, device: Optional[str] = None):

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 4-bit quantization (as reported in the paper)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quant_config,
            num_labels=num_labels,
        )

        # Fix pad token for decoder-based models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 8) -> List[int]:
        """
        Predict sentiment labels for a list of texts.
        """

        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().tolist())

        return all_preds
