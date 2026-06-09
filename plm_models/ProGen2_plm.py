# Based on: https://huggingface.co/hugohrban/progen2-xlarge

from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F

from plm_models.plm_super import PLM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ProGen2_PLM(PLM):
    def __init__(self, small=False):
        model_version = "progen2-small" if small else "progen2-xlarge" 
        self.model = AutoModelForCausalLM.from_pretrained(
            f"hugohrban/{model_version}", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if not small else torch.float32,
            output_hidden_states=True
        ).to(device)
        
        self.tokenizer = Tokenizer.from_pretrained(f"hugohrban/{model_version}")

        # Enable padding explicitly since we're using the custom tokenizer (for batch processing)
        self.tokenizer.enable_padding(direction='right', pad_id=0, pad_token="[PAD]")

        super().__init__("ProGen2", self.model)

    def generate_raw_embeddings(self, sequences, per_residue=False):
        if isinstance(sequences, str):
            sequences = [sequences]

        # Use encode_batch for the 'tokenizers' library
        encodings = self.tokenizer.encode_batch(sequences)
        
        # Convert python lists to tensors
        input_ids = torch.tensor([e.ids for e in encodings]).to(device)
        attention_mask = torch.tensor([e.attention_mask for e in encodings]).to(device)
            
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.hidden_states[-1] # last (-1) hidden state

            if per_residue:
                return embeddings
            else:
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask

    def get_embedding_size(self):
        return self.model.config.n_embd
