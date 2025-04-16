import torch.nn as nn
from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertModel

from src.utils.shogi import pieces_list

config = {
    'vocab_size': len(pieces_list) + 4,  # MASK_TOKEN_ID, MASK, CLS, SEP
    'bos_token_id': len(pieces_list) + 2,  # CLS
    'eos_token_id': len(pieces_list) + 3,  # SEP
    'cls_token_id': len(pieces_list) + 2,  # CLS
    'sep_token_id': len(pieces_list) + 3,  # SEP
    'pad_token_id': len(pieces_list) + 3,  # SEP
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,  # hidden_size * 4が目安
    'global_attn_every_n_layers': 1,
    # 'hidden_act': 'gelu',
    'mlp_dropout': 0.1,
    # 'attention_probs_dropout_prob': 0.1,
    'attention_dropout': 0.1,
    'embedding_dropout': 0.1,
    'classifier_dropout': 0.1,
    'max_position_embeddings': 95,  # 95(=81(マス目)+7(先手持駒)+7(後手持駒))でいいかも
    # 'type_vocab_size': 1,  # 対の文章を入れない。つまりtoken_type_embeddingsは完全に無駄になっている。
    'initializer_range': 0.02,
}
config = ModernBertConfig.from_dict(config)


class ModernBertMLM(nn.Module):
    def __init__(self, model_dir=None):
        super().__init__()
        if model_dir is None:
            self.bert = ModernBertForMaskedLM(config)
        else:
            self.bert = ModernBertForMaskedLM.from_pretrained(model_dir)

    def forward(self, input_ids, labels):
        return self.bert(input_ids=input_ids, labels=labels)


class ModernBertPolicyValue(nn.Module):
    def __init__(self, model_dir=None):
        super().__init__()
        if model_dir is None:
            self.bert = ModernBertModel(config)
        else:
            self.bert = ModernBertModel.from_pretrained(model_dir)

        self.policy_head = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.Tanh(),
            nn.Linear(768 * 2, 9 * 9 * 27)
        )

        self.value_head = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.Tanh(),
            nn.Linear(768 * 2, 1),
            nn.Sigmoid()
        )

        self.loss_policy_fn = nn.CrossEntropyLoss()
        self.loss_value_fn = nn.MSELoss()

    def forward(self, input_ids, labels=None):
        features = self.bert(input_ids=input_ids)['last_hidden_state']
        policy = self.policy_head(features).mean(axis=1)
        value = self.value_head(features).mean(axis=1).squeeze(1)
        if labels is None:
            return {'policy': policy, 'value': value}
        else:
            loss_policy = self.loss_policy_fn(policy, labels['labels'])
            loss_value = self.loss_value_fn(value, labels['values'])
            loss = loss_policy + loss_value
            return {'loss_policy': loss_policy, 'loss_value': loss_value, 'loss': loss}
