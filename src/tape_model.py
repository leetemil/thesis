import torch
import torch.nn as nn

from tape import ProteinModel, ProteinConfig
from tape.models.modeling_utils import SequenceClassificationHead, SequenceToSequenceClassificationHead, ValuePredictionHead, PairwiseContactPredictionHead
from tape.registry import registry

from scipy import stats
from models import WaveNet

# class UniRepReimpConfig(ProteinConfig):
class WaveNetConfig(ProteinConfig):
    def __init__(self,
        input_channels: int = 30,
        residual_channels: int = 48,
        out_channels: int = 30,
        stacks: int = 6,
        layers_per_stack: int = 9,
        total_samples: int = 0, # I think we can make do by just setting this to 0
        l2_lambda : int = 0,
        bias : bool = True,
        dropout : float = 0.5,
        bayesian : bool = False,
        backwards : bool = False,
        **kwargs):

        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.out_channels = out_channels
        self.stacks = stacks
        self.layers_per_stack = layers_per_stack
        self.total_samples = total_samples
        self.l2_lambda = l2_lambda
        self.bias = bias
        self.dropout = dropout
        self.bayesian = bayesian
        self.backwards = backwards
        self.initializer_range = 0.02 # stolen from unirep

class WaveNetAbstractModel(ProteinModel):
    config_class = WaveNetConfig
    base_model_prefix = 'wavenet'

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@registry.register_task_model('embed', 'wavenet')
class WaveNetModel(WaveNetAbstractModel):
    # init expects only a single argument - the config
    def __init__(self, config: WaveNetConfig):
        super().__init__(config)
        self.inner_model = WaveNet(
            config.input_channels,
            config.residual_channels,
            config.out_channels,
            config.stacks,
            config.layers_per_stack,
            config.total_samples,
            config.l2_lambda,
            config.bias,
            config.dropout,
            config.bayesian,
            config.backwards
        )
        self.init_weights()

    def forward(self, input_ids, input_mask = None):
        # if input_mask is None:
        #     input_mask = torch.ones_like(input_ids)
        representations = self.inner_model.get_representation(input_ids)
        return representations

@registry.register_task_model('fluorescence', 'wavenet')
@registry.register_task_model('stability', 'wavenet')
class WaveNetForValuePrediction(WaveNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.wavenet = WaveNetModel(config)
        predict_size = config.residual_channels # guess we use this as representation size
        self.predict = ValuePredictionHead(predict_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):
        representations = self.wavenet(input_ids, input_mask=input_mask)
        prediction, *_ = self.predict(representations)
        outputs = (prediction,)

        if targets is not None:
            loss = nn.MSELoss()(prediction, targets)
            metrics = {'spearman_rho': stats.spearmanr(prediction.cpu().detach(), targets.cpu().detach()).correlation}
            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction

@registry.register_task_model('remote_homology', 'wavenet')
class WaveNetForSequenceClassification(WaveNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.wavenet = WaveNetModel(config)

        if self.wavenet.inner_model.rnn_type == "GRU":
            predict_size = config.hidden_size
        else:
            predict_size = config.hidden_size * 2

        self.classify = SequenceClassificationHead(predict_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.wavenet(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # outputs = self.classify(pooled_output, targets) + outputs[2:]
        # # (loss), prediction_scores, (hidden_states)
        # return outputs

        prediction, *_ = self.classify(pooled_output)

        outputs = (prediction,)

        if targets is not None:
            loss = nn.CrossEntropyLoss()(prediction, targets)
            is_correct = prediction.float().argmax(-1) == targets
            is_valid_position = targets != -1

            # cast to float b/c otherwise torch does integer division
            num_correct = torch.sum(is_correct * is_valid_position).float()
            accuracy = num_correct / torch.sum(is_valid_position).float()
            metrics = {'acc': accuracy}

            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction

@registry.register_task_model('secondary_structure', 'wavenet')
class WaveNetForSequenceToSequenceClassification(WaveNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.wavenet = WaveNetModel(config)
        self.classify = SequenceToSequenceClassificationHead(config.hidden_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.wavenet(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # outputs = self.classify(sequence_output, targets) + outputs[2:]
        # # (loss), prediction_scores, (hidden_states)
        # return outputs

        prediction, *_ = self.classify(sequence_output)

        outputs = (prediction,)

        if targets is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-1)(prediction.view(-1, prediction.size(2)), targets.view(-1))
            # cast to float b/c float16 does not have argmax support
            is_correct = prediction.float().argmax(-1) == targets
            is_valid_position = targets != -1

            # cast to float b/c otherwise torch does integer division
            num_correct = torch.sum(is_correct * is_valid_position).float()
            accuracy = num_correct / torch.sum(is_valid_position).float()
            metrics = {'acc': accuracy}

            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction

@registry.register_task_model('contact_prediction', 'wavenet')
class WaveNetForContactPrediction(WaveNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.wavenet = WaveNetModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.wavenet(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
