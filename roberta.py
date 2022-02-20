import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import RobertaForSequenceClassification, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaClassificationHeadCustomSimple(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.dense_1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_2 = nn.Linear(2 * hidden_size, hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.dense_2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        return x


class RobertaForSequenceClassificationCustomSimple(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHeadCustomSimple(config)

        self.init_weights()


class RobertaClassificationHeadCustom(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, use_hidden_states: bool = False):
        super().__init__()
        self.use_hidden_states = use_hidden_states
        hidden_size = config.hidden_size
        if self.use_hidden_states:
            hidden_size *= 2

        self.dense_1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_2 = nn.Linear(2 * hidden_size, hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        if 'hidden_states' in kwargs and kwargs['hidden_states'] is not None:
            if self.use_hidden_states:
                x = torch.cat(
                    (
                        features[:, 0, :],
                        # take <s> token (equiv. to [CLS]) from hidden states from last layer
                        kwargs['hidden_states'][-1][:, 0, :]
                    ),
                    dim=1
                )
            else:
                x = features[:, 0, :] + kwargs['hidden_states'][-1][:, 0, :]
        else:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
            if self.use_hidden_states:
                x = torch.cat(
                    (
                        features[:, 0, :],
                        torch.zeros(x.size(), dtype=x.dtype, device=x.device)
                    ),
                    dim=1
                )

        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.dense_2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        return x


class RobertaForSequenceClassificationCustom(RobertaForSequenceClassification):
    def __init__(self, config, use_hidden_states: bool):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHeadCustom(config, use_hidden_states=use_hidden_states)

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
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if return_dict:
            logits = self.classifier(sequence_output, hidden_states=outputs.hidden_states)
        else:
            raise NotImplemented('Not implemented for using non-dictionary object')

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaClassificationHeadCustomAlternative(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_1_input = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_1_hidden = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_2 = nn.Linear(4 * hidden_size, hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if 'hidden_states' in kwargs and kwargs['hidden_states'] is not None:
            # take <s> token (equiv. to [CLS]) from hidden states from last layer
            hidden = kwargs['hidden_states'][-1][:, 0, :]
        else:
            hidden = torch.zeros(x.size(), dtype=x.dtype, device=x.device)

        x = self.dense_1_input(x)
        x = torch.relu(x)
        x = self.dropout(x)

        hidden = self.dense_1_hidden(hidden)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)

        x = torch.cat((x, hidden), dim=1)
        x = self.dense_2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        return x


class RobertaForSequenceClassificationCustomAlternative(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHeadCustomAlternative(config)

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
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if return_dict:
            logits = self.classifier(sequence_output, hidden_states=outputs.hidden_states)
        else:
            raise NotImplemented('Not implemented for using non-dictionary object')

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )