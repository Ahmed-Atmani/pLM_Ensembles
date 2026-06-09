# Based on: https://github.com/mheinzinger/ProstT5

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from plm_models.plm_super import PLM


class ProtT5_PLM(PLM):
    def __init__(self, small=False):


        if small:
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_base_mt_uniref50").to(device) # 200M params
            self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_base_mt_uniref50", do_lower_case=False)
        else:
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device) # 3B params
            self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)

        # # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
        # model.float() if device.type=='cpu' else model.half()

        super().__init__("ProtT5", model)

    def generate_raw_embeddings(self, sequences, per_residue=False):

        # If it is a single sequence, put it in a singleton list
        if isinstance(sequences, str):
            sequences = [sequences]

        # Replacing invalid/unsupported residues/tokens
        sequences = [
            " ".join(list(re.sub(r"[UZOB]", "X", seq)))
            for seq in sequences
        ]

        sequences = [
            "<AA2fold> " + s if s.isupper()
            else "<fold2AA> " + s
            for s in sequences
        ]

        ids = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self.model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            )
            
            embeddings_per_residue = outputs.last_hidden_state

            # Embeddings per residue
            if per_residue:
                return embeddings_per_residue # TODO: check this
        
            # Embeddings per sequence
            embeddings_per_sequence = embeddings_per_residue[:, 0, :]
            return embeddings_per_sequence

    def get_embedding_size(self):
        """ Returns the size of the dimension of the embeddings"""
        return self.model.config.d_model