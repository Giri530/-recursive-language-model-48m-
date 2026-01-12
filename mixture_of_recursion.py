import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_sequence_length=512, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding_dim = embedding_dim
    
    def forward(self, input_ids):
        sequence_length = input_ids.size(1)
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.dropout(embeddings)


class RecursiveLanguageModelConfig(PretrainedConfig):
    model_type = "recursive_language_model"

    def __init__(
        self,
        vocab_size=50000,
        embedding_dim=512,
        num_layers=6,
        num_attention_heads=8,
        max_recursion_steps=2,
        dropout_rate=0.1,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_recursion_steps = max_recursion_steps
        self.dropout_rate = dropout_rate
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = True


class RecursionDepthRouter(nn.Module):
    def __init__(self, embedding_dim, max_recursion_steps):
        super().__init__()
        self.max_recursion_steps = max_recursion_steps
        self.depth_classifier = nn.Linear(embedding_dim, max_recursion_steps + 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        pooled_representation = hidden_states.mean(dim=1)
        depth_logits = self.depth_classifier(self.dropout(pooled_representation))
        return F.softmax(depth_logits, dim=-1)


class RecursiveLanguageModel(PreTrainedModel):
    config_class = RecursiveLanguageModelConfig
    _tied_weights_keys = ["language_model_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embedding_layer = TokenPositionEmbedding(
            config.vocab_size,
            config.embedding_dim,
            config.max_position_embeddings,
            config.dropout_rate
        )

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.embedding_dim * 4,
                dropout=config.dropout_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(config.num_layers)
        ])

        self.recursive_processing_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.recursion_router = RecursionDepthRouter(config.embedding_dim, config.max_recursion_steps)
        self.output_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.language_model_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def tie_weights(self):
        self.language_model_head.weight = self.embedding_layer.token_embedding.weight

    def forward(self, input_ids, labels=None, attention_mask=None, return_dict=True, **kwargs):
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device

        hidden_states = self.embedding_layer(input_ids)

        causal_attention_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=device) * float('-inf'),
            diagonal=1
        )

        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(hidden_states, src_mask=causal_attention_mask)

        recursion_depth_probabilities = self.recursion_router(hidden_states)
        
        recursive_hidden_states = hidden_states.clone()
        for recursion_step in range(self.config.max_recursion_steps):
            recursive_hidden_states = self.recursive_processing_layer(
                recursive_hidden_states, 
                src_mask=causal_attention_mask
            )
            step_weight = recursion_depth_probabilities[:, recursion_step + 1].view(batch_size, 1, 1)
            hidden_states = hidden_states + recursive_hidden_states * step_weight

        hidden_states = self.output_layer_norm(hidden_states)
        logits = self.language_model_head(hidden_states)

        loss = None
        if labels is not None:
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            
            loss_function = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_function(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

    @torch.no_grad()
    def generate(
        self, 
        input_ids, 
        max_new_tokens=50, 
        temperature=1.0, 
        top_k=50, 
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        eos_token_id=None
    ):
        self.eval()
        
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        for _ in range(max_new_tokens):
            max_context_length = self.config.max_position_embeddings
            context_input_ids = (
                input_ids if input_ids.size(1) <= max_context_length
                else input_ids[:, -max_context_length:]
            )

            model_outputs = self(context_input_ids)
            next_token_logits = model_outputs.logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty

            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, min(top_k, next_token_logits.size(-1))
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probabilities = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probabilities > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')

                token_probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(token_probabilities, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token.item() == eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def get_input_embeddings(self):
        return self.embedding_layer.token_embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding_layer.token_embedding = new_embeddings

    def get_output_embeddings(self):
        return self.language_model_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model_head = new_embeddings


from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("recursive_language_model", RecursiveLanguageModelConfig)
AutoModel.register(RecursiveLanguageModelConfig, RecursiveLanguageModel)
AutoModelForCausalLM.register(RecursiveLanguageModelConfig, RecursiveLanguageModel)
