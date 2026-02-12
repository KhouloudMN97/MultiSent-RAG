import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


class LLMClassifier:
    """
    Here we use LLMs in a non-generative setting, treating them as
    sequence classification models rather than text generators.

    Quantized inference-only classifier for:
    - BLOOMZ
    - LLaMA-3
    - Mistral

    Used for baseline results reported in the paper.
    """

    def __init__(self, model_name: str, num_labels: int = 2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 4-bit quantization (as reported in the paper)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quant_config,
            num_labels=num_labels
        )

        # Fix pad token (important for LLaMA/Mistral/BLOOM)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts, batch_size: int = 8):
        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())

        return all_preds
