from typing import List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EncoderClassifier:
    """
    Inference-only wrapper for encoder-based models
    (mBERT, XLM-R).
    """

    def __init__(self, model_name: str, device: Optional[str] = None):

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 32):

        all_outputs = []

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
                all_outputs.append(outputs.logits.cpu())

        logits = torch.cat(all_outputs, dim=0)
        predictions = logits.argmax(dim=1).numpy()

        return predictions
