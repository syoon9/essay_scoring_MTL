import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers.models.bert.modeling_bert import  BertPreTrainedModel, BertPredictionHeadTransform
from transformers.models.roberta.modeling_roberta import  RobertaPreTrainedModel, RobertaPredictionHeadTransform

class PredictionHead(nn.Module):
    '''
    A prediction head for a single objective of the SpeechGraderModel.

    Args:
        config (BertConfig): the config for the the pre-trained BERT model
        num_labels (int): the number of labels that can be predicted

    Attributes:
        transform (transformers.modeling_bert.BertPredictionHeadTransform): a dense linear layer with gelu activation
            function
        decoder (torch.nn.Linear): a linear layer that makes predictions across the labels
        bias (torch.nn.Parameter): biases per label
    '''
    def __init__(self, config, num_labels, modeltype):
        super(PredictionHead, self).__init__()
        self.modeltype = modeltype
        if self.modeltype=='bert':
            self.transform = BertPredictionHeadTransform(config)
        else:
            self.transform = RobertaPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, num_labels)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class SpeechGraderModelBert(BertPreTrainedModel):
    '''
    BERT Model for automated speech scoring of transcripts.

    Attributes:
        model_dir (string) : directory to store/load this model to/from
        args : arguments from argparse
        bert_tokenizer : the tokenizer to use for the BERT model
        training_objectives_decoders (dict of str: (int, float)): a mapping of training objectives to their training
            parameters (a tuple containing the number of labels (i.e. the decoder output size) and the weight to give to
            this objective in the combined weighted loss function)
    '''
    def __init__(self, config, model_path):
        super(SpeechGraderModel, self).__init__(config)
        self.model_path = model_path

        self.bert_config = config
        self.bert = BertModel.from_pretrained(self.model_path)
        self.scoring_model_type = config.training_objectives['score'][0]

        # Creates a prediction head per objective.
        self.decoder_objectives = config.training_objectives.keys()
        for objective, objective_params in config.training_objectives.items():
            num_predictions, _ = objective_params
            decoder = PredictionHead(self.bert_config, num_predictions, 'bert')
            setattr(self, objective + '_decoder', decoder)

        # The score scaler is used to force the result of the score prediction head to be within the range of possible
        # scores.
        self.score_scaler = nn.Hardtanh(min_val=0, max_val=config.max_score)

    def forward(self, batch):
        """
        Returns:
        training_objective_predictions (dict of str: [float]): mapping of training objective to the predicted label
        """
        bert_sequence_output, bert_pooled_output = self.bert(**batch, return_dict=False)
        training_objective_predictions = {}

        for objective in self.decoder_objectives:
            input = bert_pooled_output if objective == 'score' else bert_sequence_output
            decoded_objective = getattr(self, objective + '_decoder')(input)
            decoded_objective = self.score_scaler(decoded_objective) if (objective == 'score' and self.scoring_model_type ==1)  else decoded_objective
            if objective == 'score': 
                if self.scoring_model_type ==1: # regression
                    training_objective_predictions[objective] = decoded_objective.squeeze()
                else: # classification
                    training_objective_predictions[objective] = decoded_objective.view(-1, self.scoring_model_type) 
            else:
                 training_objective_predictions[objective] = decoded_objective.view(-1, decoded_objective.shape[2]) 

        return training_objective_prediction


class SpeechGraderModelRoberta(RobertaPreTrainedModel):
    '''
    Roberta Model for automated speech scoring of transcripts.

    Attributes:
        model_dir (string) : directory to store/load this model to/from
        args : arguments from argparse
        bert_tokenizer : the tokenizer to use for the BERT model
        training_objectives_decoders (dict of str: (int, float)): a mapping of training objectives to their training
            parameters (a tuple containing the number of labels (i.e. the decoder output size) and the weight to give to
            this objective in the combined weighted loss function)
    '''
    def __init__(self, config, model_path):
        super(SpeechGraderModelRoberta, self).__init__(config)
        self.model_path = model_path

        self.bert_config = config
        self.bert = RobertaModel.from_pretrained(self.model_path)
        self.scoring_model_type = config.training_objectives['score'][0]

        # Creates a prediction head per objective.
        self.decoder_objectives = config.training_objectives.keys()
        for objective, objective_params in config.training_objectives.items():
            num_predictions, _ = objective_params
            decoder = PredictionHead(self.bert_config, num_predictions, 'roberta')
            setattr(self, objective + '_decoder', decoder)

        # The score scaler is used to force the result of the score prediction head to be within the range of possible
        # scores.
        self.score_scaler = nn.Hardtanh(min_val=0, max_val=config.max_score)

    def forward(self, batch):
        """
        Returns:
        training_objective_predictions (dict of str: [float]): mapping of training objective to the predicted label
        """
        bert_sequence_output, bert_pooled_output = self.bert(**batch, return_dict=False)
        training_objective_predictions = {}

        for objective in self.decoder_objectives:
            input = bert_pooled_output if objective == 'score' else bert_sequence_output
            decoded_objective = getattr(self, objective + '_decoder')(input)
            decoded_objective = self.score_scaler(decoded_objective) if (objective == 'score' and self.scoring_model_type ==1)  else decoded_objective
            if objective == 'score':
                if self.scoring_model_type ==1: # regression
                    training_objective_predictions[objective] = decoded_objective.squeeze()
                else: # classification
                    training_objective_predictions[objective] = decoded_objective.view(-1, self.scoring_model_type)
            else:
                 training_objective_predictions[objective] = decoded_objective.view(-1, decoded_objective.shape[2])

        return training_objective_predictions
