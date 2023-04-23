import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    BertConfig,
    BertPreTrainedModel,
    BertModel,
    BertForPreTraining
)
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions
)
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from dataset import CDADataset, get_dataloaders
from config import ModelArguments, DEVICE


class MLPLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super(MLPLayer, self).__init__()
        self.hidden1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        return x


class Pooler(nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()

    def forward(self, outputs: BaseModelOutputWithPoolingAndCrossAttentions):
        return outputs.last_hidden_state[:, 0]


class MABELBert(BertForPreTraining):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: BertConfig, **model_kwargs):
        super().__init__(config)
        self.config = config
        model_args = model_kwargs.get('model_args')
        self.encoder = BertModel.from_pretrained(config._name_or_path,
                                                 add_pooling_layer=False)

        self.masked_head = BertLMPredictionHead(config)
        self.pooler = Pooler()
        self.mlp = MLPLayer(config)

        # Init weights
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            mlm_input_ids=None,
            mlm_labels=None
    ):
        mlm_outputs = None
        num_sent = input_ids.size(1)
        max_len = input_ids.size(-1)

        # Separate input_ids, attention_masks and token_type_ids
        # for each of the 4 sentences

        og_z0_input_ids, og_z0_mask, og_z0_tok_type = (
            input_ids[:, 0, :],
            attention_mask[:, 0, :],
            token_type_ids[:, 0, :]
        )
        og_z1_input_ids, og_z1_mask, og_z1_tok_type = (
            input_ids[:, 1, :],
            attention_mask[:, 1, :],
            token_type_ids[:, 1, :]
        )
        aug_z0_input_ids, aug_z0_mask, aug_z0_tok_type = (
            input_ids[:, 2, :],
            attention_mask[:, 2, :],
            token_type_ids[:, 2, :]
        )
        aug_z1_input_ids, aug_z1_mask, aug_z1_tok_type = (
            input_ids[:, 3, :],
            attention_mask[:, 3, :],
            token_type_ids[:, 3, :]
        )

        # Pass all the sentences through the encoder
        og_z0_outputs = self.encoder(
            input_ids=og_z0_input_ids,
            attention_mask=og_z0_mask,
            token_type_ids=og_z0_tok_type
        )

        og_z1_outputs = self.encoder(
            input_ids=og_z1_input_ids,
            attention_mask=og_z1_mask,
            token_type_ids=og_z1_tok_type
        )

        aug_z0_outputs = self.encoder(
            input_ids=aug_z0_input_ids,
            attention_mask=aug_z0_mask,
            token_type_ids=aug_z0_tok_type
        )

        aug_z1_outputs = self.encoder(
            input_ids=aug_z1_input_ids,
            attention_mask=aug_z1_mask,
            token_type_ids=aug_z1_tok_type
        )

        # Masked language model
        mlm_outputs = self.encoder(
            input_ids=mlm_input_ids,
            attention_mask=attention_mask.view(-1, ModelArguments.max_len),
            token_type_ids=token_type_ids.view(-1, ModelArguments.max_len),
            return_dict=True,
            output_hidden_states=False
        )
        mlm_outputs = self.masked_head(mlm_outputs.last_hidden_state)
        mlm_outputs = mlm_outputs.view(-1, self.config.vocab_size)

        # Pass the outputs through a pooler
        og_z0 = self.pooler(og_z0_outputs)
        og_z1 = self.pooler(og_z1_outputs)
        aug_z0 = self.pooler(aug_z0_outputs)
        aug_z1 = self.pooler(aug_z1_outputs)

        og_z0 = self.mlp(og_z0)
        og_z1 = self.mlp(og_z1)
        aug_z0 = self.mlp(aug_z0)
        aug_z1 = self.mlp(aug_z1)

        return og_z0, og_z1, aug_z0, aug_z1, mlm_outputs


def similarity(x, y, temp=0.05):
    cos = nn.CosineSimilarity(dim=-1)
    return cos(x, y) / temp


def contrastive_loss(og0, og1, aug0, aug1):
    z0 = torch.cat([og0, aug0])
    z1 = torch.cat([og1, aug1])

    cos_sim = similarity(z0.unsqueeze(1), z1.unsqueeze(0))
    labels = torch.arange(cos_sim.size(0)).long().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    cl_loss = criterion(cos_sim, labels)

    return cl_loss


def alignment_loss(og0, og1, aug0, aug1, alpha=2):
    og_cos_sim = similarity(og0.unsqueeze(1), og1.unsqueeze(0))
    aug_cos_sim = similarity(aug0.unsqueeze(1), aug1.unsqueeze(0))
    align_loss = (og_cos_sim - aug_cos_sim).norm(p=2, dim=1).pow(alpha).mean()

    return align_loss


def mlm_loss(mlm_outputs, mlm_labels):
    criterion = nn.CrossEntropyLoss()
    masked_loss = criterion(mlm_outputs, mlm_labels.view(-1))
    return masked_loss


def training_objective(og0, og1, aug0, aug1, mlm_outputs, mlm_labels,
                       align_temp=0.05, mlm_weight=0.1, **kwargs):
    cl_loss = contrastive_loss(og0, og1, aug0, aug1)
    al_loss = alignment_loss(og0, og1, aug0, aug1, **kwargs)
    mlm_loss_ = mlm_loss(mlm_outputs, mlm_labels)

    loss_comb = ((1 - align_temp) * al_loss
                 + align_temp * cl_loss
                 + mlm_weight * mlm_loss_)

    return loss_comb


if __name__ == '__main__':
    model = MABELBert(AutoConfig.from_pretrained(ModelArguments.hf_model))
    model = model.to(DEVICE)
    dataset = CDADataset('./data/data_aug.csv')
    dataloader = get_dataloaders(dataset)
    inputs = next(iter(dataloader))

    for key, val in inputs.items():
        inputs[key] = val.to(DEVICE)

    og_z0, og_z1, aug_z0, aug_z1, mlm_outputs = model(**inputs)
    print(contrastive_loss(og_z0, og_z1, aug_z0, aug_z1))
    print(alignment_loss(og_z0, og_z1, aug_z0, aug_z1))
    print(mlm_loss(mlm_outputs, inputs['mlm_labels']))

