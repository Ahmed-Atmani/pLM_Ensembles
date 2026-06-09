# Based on: https://huggingface.co/ElnaggarLab/ankh3-xl

import torch
from transformers import T5Tokenizer, T5EncoderModel

from plm_models.plm_super import PLM


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Ankh3_PLM(PLM):
    def __init__(self, small=False):
        if small:
            version = "ElnaggarLab/ankh-base"
        else:
            version = "ElnaggarLab/ankh3-large"
            # version = "ElnaggarLab/ankh3-xl" # Too large

        self.tokenizer = T5Tokenizer.from_pretrained(version)
        model = T5EncoderModel.from_pretrained(version).to(device)
        model.eval()

        super().__init__("Ankh3", model)

    def generate_raw_embeddings(self, sequences, per_residue=False):

        if isinstance(sequences, str):
            sequences = [sequences]

        # Ankh3 expects an [NLU] prefix for sequence-level tasks
        nlu_sequences = ["[NLU]" + seq for seq in sequences]

        outputs = self.tokenizer(
            nlu_sequences, 
            add_special_tokens=True, 
            padding=True, 
            return_tensors="pt",
        ).to(device)
            
        with torch.no_grad():
            outputs = self.model(
                input_ids=outputs["input_ids"], 
                attention_mask=outputs["attention_mask"]
            )

            if per_residue:
                return outputs.last_hidden_state[:, 1:, :] # Remove first token
            
            return outputs.last_hidden_state[:, 1:, :].mean(dim=1)

    def get_embedding_size(self):
        """ Returns the hidden size from the model configuration """
        return self.model.config.d_model