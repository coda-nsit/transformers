from torch import nn
from transformers import BertPreTrainedModel, BertForNextSentencePrediction


class BertEnsemble(BertPreTrainedModel):
    """BERT model for Ensemble classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels1, num_labels2, num_labels3, num_labels4,
                 num_labels5, model_dir1, model_dir2, model_dir3, model_dir4):
        super(BertForEnsemble, self).__init__(config)
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.num_labels3 = num_labels3
        self.num_labels4 = num_labels4
        self.num_labels5 = num_labels5

        self.bert1 = BertForSequenceClassification.from_pretrained(model_dir1,
                                                                   num_labels=num_labels1).bert
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.bert2 = BertForSequenceClassification.from_pretrained(model_dir2,
                                                                   num_labels=num_labels2).bert
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.bert3 = BertForSequenceClassification.from_pretrained(model_dir3,
                                                                   num_labels=num_labels3).bert
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)
        self.bert4 = BertForSequenceClassification.from_pretrained(model_dir4,
                                                                   num_labels=num_labels4).bert
        self.dropout4 = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(4 * 768,
                                    num_labels5)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output1 = self.bert1(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output1 = self.dropout1(pooled_output1)
        _, pooled_output2 = self.bert2(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output2 = self.dropout2(pooled_output2)
        _, pooled_output3 = self.bert3(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output3 = self.dropout3(pooled_output3)
        _, pooled_output4 = self.bert4(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output4 = self.dropout4(pooled_output4)

        pooled_output = torch.cat((pooled_output1, pooled_output2,
                                   pooled_output3, pooled_output4), dim=1)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels5), labels.view(-1))
            return loss
        else:
            return logits