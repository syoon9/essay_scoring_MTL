import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn import metrics

def get_losses(training_objectives, training_objective_predictions, labels, device):
    """
    Returns a dict containing training objectives and their respective losses over their predictions and the combined
    weighted loss of the training objectives.
    """
    losses = {'overall': torch.tensor([0.], device=device)}
    scoring_model_type = training_objectives['score'][0]
    for objective, prediction in training_objective_predictions.items():
        _, alpha = training_objectives[objective]
        # Scoring objective uses MSE loss and all other objectives use CrossEntropy loss
        criterion = nn.MSELoss().to(device) if (objective == 'score' and scoring_model_type==1) else nn.CrossEntropyLoss(
            ignore_index=-1).to(device)
        batch_labels = labels[objective].reshape(-1)
        losses[objective] = criterion(prediction, batch_labels)
        losses['overall'] += (losses[objective] * alpha)
    return losses

def compute_metrics(total_losses, all_score_predictions, all_score_targets, device,
                    all_tag_predictions=None, all_tag_targets=None,
                    all_nl_predictions=None, all_nl_targets=None):
    """ Computes Pearson correlation and accuracy within 0.5 and 1 of target score and adds each to total_losses dict. """
    total_losses['pearson'] = stats.pearsonr(all_score_predictions.cpu(), all_score_targets.cpu())[0]
    total_losses['within_0.5'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 0.5,
                                                              device)
    total_losses['within_1'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 1,
                                                                device)
    if all_tag_targets:
        assert len(all_tag_targets) == len(all_tag_predictions)
        new_targets = []
        new_preds = []
        for i,label in enumerate(all_tag_targets):
            if i != -1:
                new_targets.append(label)
                new_preds.append(all_tag_predictions[i])
        total_losses['tag_accuracy'] = metrics.accuracy_score(all_tag_predictions, all_tag_targets)

    if all_nl_targets:
        assert len(all_nl_targets) == len(all_nl_predictions)
        total_losses['nl_accuracy'] = metrics.accuracy_score(all_nl_predictions, all_nl_targets)

def _accuracy_within_margin(score_predictions, score_target, margin, device):
    """ Returns the percentage of predicted scores that are within the provided margin from the target score. """
    return torch.sum(
        torch.where(
            torch.abs(score_predictions - score_target) <= margin,
            torch.ones(len(score_predictions), device=device),
            torch.zeros(len(score_predictions), device=device))).item() / len(score_predictions) * 100

