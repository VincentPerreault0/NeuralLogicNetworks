import random
from statistics import mean
from typing import List, Union, Tuple
import warnings
import copy
from math import ceil, exp, floor, log2
from matplotlib import pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from NLN_Logging import get_log_files, close_log_files, print_log

VERBOSE = False

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

NB_RULES = 128

TRAIN_FORW_WEIGHT_QUANT = ""
APPROX_PARAMS = None

DISCRETIZATION_METHOD = "sel_desc"

USE_RULE_MODULE = True

RESET_UNUSED_CONCEPTS = True
UNUSED_CONCEPT_THRESHOLD = 1e-5

RANDOM_INIT_OBS = True
RANDOM_INIT_UNOBS = False

EMPTY_INIT_TARGETS = False
EMPTY_RESET_IN_CONCEPTS = True

NB_DICHOTOMIES_PER_CONTINUOUS = 32


def find_lexicographical_order(matrix, order_outs=True, output_list=None, equivalency_classes=None):
    if equivalency_classes == None:
        nb_order_idcs = matrix.size(0) if order_outs else matrix.size(1)
        return find_lexicographical_order(
            matrix,
            order_outs=order_outs,
            output_list=[],
            equivalency_classes=[(0, list(range(nb_order_idcs)))],
        )
    elif equivalency_classes == []:
        return output_list
    current_value_idx, current_equivalency_class = equivalency_classes.pop(0)
    if len(current_equivalency_class) == 1 or current_value_idx == -1:
        output_list += current_equivalency_class
    else:
        next_used_value_idx_idx_to_order_pairs = []
        for idx_to_order in current_equivalency_class:
            next_used_value_idx = -1
            if order_outs:
                for value_idx in range(current_value_idx, matrix.size(1)):
                    if matrix[idx_to_order, value_idx] > 0:
                        next_used_value_idx = value_idx
                        break
                    elif matrix[idx_to_order, value_idx] < 0:
                        next_used_value_idx = value_idx + 0.5
                        break
            else:
                for value_idx in range(current_value_idx, matrix.size(0)):
                    if matrix[value_idx, idx_to_order] > 0:
                        next_used_value_idx = value_idx
                        break
                    elif matrix[value_idx, idx_to_order] < 0:
                        next_used_value_idx = value_idx + 0.5
                        break
            next_used_value_idx_idx_to_order_pairs.append((next_used_value_idx, idx_to_order))
        next_used_value_idx_idx_to_order_pairs.sort(key=lambda pair: pair[0])
        new_equivalency_classes = []
        current_next_used_value_idx = -2
        for next_used_value_idx, idx_to_order in next_used_value_idx_idx_to_order_pairs:
            if next_used_value_idx == current_next_used_value_idx:
                new_equivalency_classes[-1][1].append(idx_to_order)
            else:
                if next_used_value_idx != -1:
                    new_equivalency_classes.append((floor(next_used_value_idx) + 1, [idx_to_order]))
                else:
                    new_equivalency_classes.append((-1, [idx_to_order]))
                current_next_used_value_idx = next_used_value_idx
        equivalency_classes = new_equivalency_classes + equivalency_classes

    return find_lexicographical_order(
        matrix,
        order_outs=order_outs,
        output_list=output_list,
        equivalency_classes=equivalency_classes,
    )


class GradGraft(torch.autograd.Function):
    """Implements the Gradient Grafting."""

    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class CombinationConcepts(nn.Module):
    """Parent class of AND concepts and OR concepts."""

    def __init__(
        self,
        is_AND: bool,
        nb_in_concepts: int,
        nb_out_concepts: int,
        use_negation: bool = True,
        use_unobserved: bool = True,
        use_missing_values: bool = False,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        empty_reset_in_concepts: bool = EMPTY_RESET_IN_CONCEPTS,
        train_forw_weight_quant: str = TRAIN_FORW_WEIGHT_QUANT,
        approx_AND_OR_params: Union[None, Tuple[float, float, float]] = APPROX_PARAMS,
        in_concepts_group_first_stop_pairs: List[Tuple[int, int]] = [],
        device=DEVICE,
        verbose: bool = VERBOSE,
    ):
        super().__init__()
        self.is_AND = is_AND
        if is_AND:
            self.junction_display_string = "AND"
            self.full_unobserved_value = 1
        else:
            self.junction_display_string = "OR"
            self.full_unobserved_value = 0
        self.nb_in_concepts = nb_in_concepts
        self.nb_out_concepts = nb_out_concepts
        self.use_negation = use_negation
        self.use_unobserved = use_unobserved
        self.use_missing_values = use_missing_values
        self.random_init_obs = random_init_obs
        self.random_init_unobs = random_init_unobs
        self.empty_reset_in_concepts = empty_reset_in_concepts
        self.train_forw_weight_quant = train_forw_weight_quant
        self.approx_AND_OR_params = approx_AND_OR_params
        self.in_concepts_group_first_stop_pairs = in_concepts_group_first_stop_pairs
        self.is_grouped = USE_RULE_MODULE and len(self.in_concepts_group_first_stop_pairs) > 0
        if self.is_grouped:
            self.nb_ungrouped_in_concepts = self.in_concepts_group_first_stop_pairs[0][0]
            nb_in_concepts = self.nb_ungrouped_in_concepts
        self.device = device
        self.verbose = verbose
        if random_init_obs:
            if use_negation:
                self.observed_concepts = nn.Parameter(torch.Tensor(nb_out_concepts, nb_in_concepts).uniform_(-1, 1))
            else:
                self.observed_concepts = nn.Parameter(torch.Tensor(nb_out_concepts, nb_in_concepts).uniform_(0, 1))
            if use_missing_values:
                self.missing_observed_concepts = nn.Parameter(torch.Tensor(nb_out_concepts).uniform_(0, 1))
        else:
            self._reinitialize_next_resets(first_time=True)
            self.observed_concepts = nn.Parameter(torch.zeros(nb_out_concepts, nb_in_concepts))
            for out_concept in range(nb_out_concepts):
                in_concept, sign = self._get_next_reset()
                if not self.is_grouped or in_concept < self.nb_ungrouped_in_concepts:
                    self.observed_concepts.data[out_concept, in_concept] = sign
            if use_missing_values:
                self.missing_observed_concepts = nn.Parameter(torch.zeros(nb_out_concepts))
        self.observed_concepts.data = self.observed_concepts.data.to(device)
        if use_missing_values:
            self.missing_observed_concepts.data = self.missing_observed_concepts.data.to(device)
        if self.is_grouped:
            self.observed_grouped_concepts = nn.Parameter(torch.ones(nb_out_concepts, len(self.in_concepts_group_first_stop_pairs)))
            self.observed_grouped_concepts.data = self.observed_grouped_concepts.data.to(device)
        if use_unobserved:
            if random_init_unobs:
                self.unobserved_concepts = nn.Parameter(torch.Tensor(nb_out_concepts).uniform_(0, 1))
            else:
                self.unobserved_concepts = nn.Parameter(self.full_unobserved_value * torch.ones(nb_out_concepts))
            self.unobserved_concepts.data = self.unobserved_concepts.data.to(device)
        self.overridden = ""
        self.to(device)

    def set(
        self,
        empty_reset_in_concepts=None,
        train_forw_weight_quant=None,
        approx_AND_OR_params="",
        device=None,
        verbose=None,
    ):
        if empty_reset_in_concepts != None and empty_reset_in_concepts != self.empty_reset_in_concepts:
            self.empty_reset_in_concepts = empty_reset_in_concepts
        if train_forw_weight_quant != None and train_forw_weight_quant != self.train_forw_weight_quant:
            self.train_forw_weight_quant = train_forw_weight_quant
        if approx_AND_OR_params != "" and approx_AND_OR_params != self.approx_AND_OR_params:
            self.approx_AND_OR_params = approx_AND_OR_params
        if verbose != None and verbose != self.verbose:
            self.verbose = verbose
        if device != None and device != self.device:
            self.device = device
            self.observed_concepts.data = self.observed_concepts.data.to(device)
            if self.use_missing_values:
                self.missing_observed_concepts.data = self.missing_observed_concepts.data.to(device)
            if self.is_grouped:
                self.observed_grouped_concepts.data = self.observed_grouped_concepts.data.to(device)
            if self.use_unobserved:
                self.unobserved_concepts.data = self.unobserved_concepts.data.to(device)
            self.to(device)

    def _get_in_concept_from_out_concept_in_group(self, out_concept, in_group):
        return self.nb_ungrouped_in_concepts + self.nb_out_concepts * in_group + out_concept

    def _get_out_concept_in_group_from_in_concept(self, in_concept):
        for in_concepts_group_idx, first_stop_pair in enumerate(self.in_concepts_group_first_stop_pairs):
            first_in_concept_idx, stop_in_concept_idx = first_stop_pair
            if in_concept >= first_in_concept_idx and in_concept < stop_in_concept_idx:
                break
        return in_concept - first_in_concept_idx, in_concepts_group_idx

    def _get_in_concepts_from_grouped_nonzero(self, tensor_to_nonzero, used_out_concepts=-1):
        out_concept_in_group_pairs = torch.nonzero(tensor_to_nonzero).tolist()
        if used_out_concepts != -1:
            out_concept_in_group_pairs = [
                (used_out_concepts[used_out_concept_idx_in_group_pair[0]], used_out_concept_idx_in_group_pair[1])
                for used_out_concept_idx_in_group_pair in out_concept_in_group_pairs
            ]
        return [
            self._get_in_concept_from_out_concept_in_group(out_concept_in_group_pair[0], out_concept_in_group_pair[1]) for out_concept_in_group_pair in out_concept_in_group_pairs
        ]

    def ungroup(self):
        if self.is_grouped:
            old_observed_concepts = 1 * self.observed_concepts.data
            self.observed_concepts.data = torch.zeros((self.nb_out_concepts, self.nb_in_concepts)).to(self.device)
            self.observed_concepts.data[:, : self.nb_ungrouped_in_concepts] = old_observed_concepts
            for out_concept in range(self.nb_out_concepts):
                for in_group in range(len(self.in_concepts_group_first_stop_pairs)):
                    in_concept = self._get_in_concept_from_out_concept_in_group(out_concept, in_group)
                    self.observed_concepts.data[out_concept, in_concept] = self.observed_grouped_concepts.data[out_concept, in_group]
            self.is_grouped = False
            del self.nb_ungrouped_in_concepts
            del self.observed_grouped_concepts

    def get_weight_value(self, out_concept, in_concept):
        if not self.is_grouped or in_concept < self.nb_ungrouped_in_concepts:
            return self.observed_concepts.data[out_concept, in_concept].item()
        else:
            out_concept, in_group = self._get_out_concept_in_group_from_in_concept(in_concept)
            return self.observed_grouped_concepts.data[out_concept, in_group].item()

    def _reinitialize_next_resets(self, first_time=False):
        next_resets_idcs = list(range(self.nb_in_concepts))
        if not first_time:
            random.shuffle(next_resets_idcs)
        if self.use_negation:
            self.next_pos_resets = next_resets_idcs
            self.next_neg_resets = next_resets_idcs.copy()
        else:
            self.next_pos_resets = next_resets_idcs

    def _get_next_reset(self):
        if len(self.next_pos_resets) == 0:
            if not self.use_negation or len(self.next_neg_resets) == 0:
                self._reinitialize_next_resets()
                return self.next_pos_resets.pop(0), 1
            else:
                return self.next_neg_resets.pop(0), -1
        else:
            return self.next_pos_resets.pop(0), 1

    def reset_out_concept(self, out_concept):
        nb_in_concepts = self.nb_in_concepts if not self.is_grouped else self.nb_ungrouped_in_concepts
        if self.random_init_obs:
            if self.use_negation:
                self.observed_concepts.data[out_concept, :] = torch.Tensor(1, nb_in_concepts).uniform_(-1, 1)
            else:
                self.observed_concepts.data[out_concept, :] = torch.Tensor(1, nb_in_concepts).uniform_(0, 1)
            if self.use_missing_values:
                self.missing_observed_concepts.data[out_concept] = torch.Tensor(1).uniform_(0, 1)
        else:
            self.observed_concepts.data[out_concept, :] = torch.zeros(1, nb_in_concepts)
            in_concept, sign = self._get_next_reset()
            if not self.is_grouped or in_concept < self.nb_ungrouped_in_concepts:
                self.observed_concepts.data[out_concept, in_concept] = sign
            if self.use_missing_values:
                self.missing_observed_concepts.data[out_concept] = 0
        if self.is_grouped:
            self.observed_grouped_concepts.data[out_concept, :] = torch.ones(1, len(self.in_concepts_group_first_stop_pairs))
        if self.use_unobserved:
            if self.random_init_unobs:
                self.unobserved_concepts.data[out_concept] = torch.Tensor(1).uniform_(0, 1)
            else:
                self.unobserved_concepts.data[out_concept] = self.full_unobserved_value

    def reset_in_concept(self, in_concept):
        if not self.is_grouped or in_concept < self.nb_ungrouped_in_concepts:
            if self.random_init_obs:
                if self.use_negation:
                    self.observed_concepts.data[:, in_concept] = torch.Tensor(self.nb_out_concepts).uniform_(-1, 1)
                else:
                    self.observed_concepts.data[:, in_concept] = torch.Tensor(self.nb_out_concepts).uniform_(0, 1)
            else:
                self.observed_concepts.data[:, in_concept] = torch.zeros(self.nb_out_concepts)
        else:
            out_concept, in_group = self._get_out_concept_in_group_from_in_concept(in_concept)
            self.observed_grouped_concepts.data[out_concept, in_group] = 1

    def update_parameters(self):
        if self.use_negation:
            self.observed_concepts.data.clamp_(-1, 1)
            if self.is_grouped:
                self.observed_grouped_concepts.data.clamp_(-1, 1)
        else:
            self.observed_concepts.data.clamp_(0, 1)
            if self.is_grouped:
                self.observed_grouped_concepts.data.clamp_(0, 1)
        if self.use_unobserved:
            self.unobserved_concepts.data.clamp_(0, 1)
        if self.use_missing_values:
            self.missing_observed_concepts.data.clamp_(0, 1)

    def override(self, new_override):
        self.overridden = new_override

    @torch.no_grad()
    def _sample_weights(self, observed_concepts_data):
        weight_prob_thresholds = torch.abs(observed_concepts_data)
        weight_signs = torch.sign(observed_concepts_data)
        probs = torch.FloatTensor(observed_concepts_data.shape).to(self.device).uniform_()
        mask = probs <= weight_prob_thresholds
        sampled_observed_concepts = mask * weight_signs
        return sampled_observed_concepts

    @torch.no_grad()
    def _threshold_weights(self, observed_concepts_data, threshold=0.5):
        mask_pos = observed_concepts_data >= threshold
        thresholded_observed_concepts = torch.zeros_like(observed_concepts_data)
        thresholded_observed_concepts[mask_pos] = 1
        if self.use_negation:
            mask_neg = observed_concepts_data <= -threshold
            thresholded_observed_concepts[mask_neg] = -1
        return thresholded_observed_concepts

    def review_unused_concepts(self, used_out_concepts, also_unused_in_concepts, do_check_in_concepts=True):
        if RESET_UNUSED_CONCEPTS:
            if do_check_in_concepts:
                if len(used_out_concepts) > 0:
                    in_concepts_max_observed = torch.max(
                        torch.abs(self.observed_concepts[used_out_concepts, :].detach()),
                        dim=0,
                    )[0]
                    used_in_concepts = torch.nonzero(in_concepts_max_observed > UNUSED_CONCEPT_THRESHOLD).view(-1).tolist()
                    if self.is_grouped:
                        used_in_concepts += self._get_in_concepts_from_grouped_nonzero(
                            torch.abs(self.observed_grouped_concepts[used_out_concepts, :].detach()) > UNUSED_CONCEPT_THRESHOLD, used_out_concepts=used_out_concepts
                        )
                    used_in_concepts = [used_in_concept for used_in_concept in used_in_concepts if used_in_concept not in also_unused_in_concepts]
                    used_in_concepts.sort()
                else:
                    used_in_concepts = []
                if self.verbose:
                    if len(used_in_concepts) < self.nb_in_concepts:
                        if len(used_in_concepts) == self.nb_in_concepts - 1:
                            display_string = "Unused " + self.junction_display_string + " in concept: "
                        else:
                            display_string = "Unused " + self.junction_display_string + " in concepts: "
                        for i, unused_in_concept in enumerate([unused_in_concept for unused_in_concept in range(self.nb_in_concepts) if unused_in_concept not in used_in_concepts]):
                            if i == 0:
                                display_string += str(unused_in_concept)
                            else:
                                display_string += ", " + str(unused_in_concept)
                        print(display_string)
            if do_check_in_concepts:
                for in_concept in range(self.nb_in_concepts):
                    if in_concept not in used_in_concepts:
                        if self.empty_reset_in_concepts:
                            if not self.is_grouped or in_concept < self.nb_ungrouped_in_concepts:
                                self.observed_concepts.data[:, in_concept] = 0
                            else:
                                out_concept, in_group = self._get_out_concept_in_group_from_in_concept(in_concept)
                                self.observed_grouped_concepts.data[out_concept, in_group] = 0
                        else:
                            self.reset_in_concept(in_concept)
            for out_concept in range(self.nb_out_concepts):
                if out_concept not in used_out_concepts:
                    self.reset_out_concept(out_concept)
            if do_check_in_concepts:
                return used_in_concepts

    def forward(self, x):
        if not self.training or self.train_forw_weight_quant == "":
            result = self._forward_pass(x)
        else:
            train_back_result = self._forward_pass(x)

            self.override(self.train_forw_weight_quant)
            with torch.no_grad():
                train_forw_result = self._forward_pass(x)
            self.override("")

            result = GradGraft.apply(train_forw_result, train_back_result)

        return result

    def _forward_pass(self, x):
        """Forward pass"""
        if self.overridden == "stoch" or self.overridden == "thresh" or self.overridden == "sthresh":
            if self.overridden == "stoch":
                observed_concepts = self._sample_weights(self.observed_concepts.data)
                if self.is_grouped:
                    observed_grouped_concepts = self._sample_weights(self.observed_grouped_concepts.data)
            elif self.overridden == "thresh":
                observed_concepts = self._threshold_weights(self.observed_concepts.data)
                if self.is_grouped:
                    observed_grouped_concepts = self._threshold_weights(self.observed_grouped_concepts.data)
            else:  # self.overridden == "sthresh":
                stochastic_threshold = random.random()
                observed_concepts = self._threshold_weights(self.observed_concepts.data, threshold=stochastic_threshold)
                if self.is_grouped:
                    observed_grouped_concepts = self._threshold_weights(self.observed_grouped_concepts.data, threshold=stochastic_threshold)
        else:
            observed_concepts = self.observed_concepts
            if self.is_grouped:
                observed_grouped_concepts = self.observed_grouped_concepts

        x = x.view(-1, self.nb_in_concepts)
        if not self.is_grouped:
            if self.approx_AND_OR_params == None or not self.training:
                x_v = x.view(-1, 1, self.nb_in_concepts)
                observed_concepts_v = observed_concepts.view(1, self.nb_out_concepts, self.nb_in_concepts)
                result = self.combine_observed_concepts(x_v, observed_concepts_v)
            else:
                result = self.combine_observed_concepts_approximately(x, observed_concepts)
        else:
            if self.nb_ungrouped_in_concepts > 0:
                if self.approx_AND_OR_params == None or not self.training:
                    x_v = x[:, : self.nb_ungrouped_in_concepts].view(-1, 1, self.nb_ungrouped_in_concepts)
                    observed_concepts_v = observed_concepts.view(1, self.nb_out_concepts, self.nb_ungrouped_in_concepts)
                    result = self.combine_observed_concepts(x_v, observed_concepts_v)
                else:
                    x_v = x[:, : self.nb_ungrouped_in_concepts].view(-1, self.nb_ungrouped_in_concepts)
                    result = self.combine_observed_concepts_approximately(x_v, observed_concepts)

            grouped_x_v = x[:, self.nb_ungrouped_in_concepts :].view(-1, len(self.in_concepts_group_first_stop_pairs), self.nb_out_concepts)
            grouped_x_v = grouped_x_v.transpose(1, 2)
            observed_grouped_concepts_v = observed_grouped_concepts.view(1, self.nb_out_concepts, len(self.in_concepts_group_first_stop_pairs))
            if self.approx_AND_OR_params == None or not self.training:
                grouped_result = self.combine_observed_concepts(grouped_x_v, observed_grouped_concepts_v)
            else:
                grouped_result = self.combine_observed_concepts_approximately(grouped_x_v, observed_grouped_concepts_v, combine_grouped=True)

            if self.nb_ungrouped_in_concepts > 0:
                result = result * grouped_result
            else:
                result = grouped_result

        if self.approx_AND_OR_params != None and self.training:
            alpha, beta, gamma = self.approx_AND_OR_params
            result = 1.0 / (1.0 + result) ** gamma

        if self.use_missing_values:
            if len(self.missing_idcs) > 0:
                result = result.clone()
                if self.is_AND:
                    result[self.missing_idcs, :] = self.missing_observed_concepts.unsqueeze(0)
                else:
                    result[self.missing_idcs, :] = 1 - self.missing_observed_concepts.unsqueeze(0)

        result = self.combine_with_unobserved_concepts(result)

        return result

    def add_regularization(self, loss):
        if self.use_negation:
            less_than_one_observed_concepts_sums = torch.sum(torch.abs(self.observed_concepts), dim=1)
        else:
            less_than_one_observed_concepts_sums = torch.sum(self.observed_concepts, dim=1)
        if self.is_grouped:
            if self.use_negation:
                less_than_one_observed_concepts_sums += torch.sum(torch.abs(self.observed_grouped_concepts), dim=1)
            else:
                less_than_one_observed_concepts_sums += torch.sum(self.observed_grouped_concepts, dim=1)
        less_than_one_observed_concepts_sums = less_than_one_observed_concepts_sums[less_than_one_observed_concepts_sums < 1]
        if less_than_one_observed_concepts_sums.size(0) > 0:
            loss += 1e-1 * torch.nn.functional.mse_loss(
                less_than_one_observed_concepts_sums,
                torch.ones_like(less_than_one_observed_concepts_sums),
            )
        if not self.is_grouped or self.nb_ungrouped_in_concepts > 0:
            loss += 1e-3 * torch.nn.functional.l1_loss(self.observed_concepts, torch.zeros_like(self.observed_concepts))
        if self.is_grouped:
            loss += 1e-3 * torch.nn.functional.l1_loss(self.observed_grouped_concepts, torch.zeros_like(self.observed_grouped_concepts))
        return loss

    def get_observed_unobserved_parameters(self):
        observed_concepts = []
        unobserved_concepts = []
        for name, parameter in self.named_parameters():
            if "unobserved" in name:
                unobserved_concepts.append(parameter)
            elif "missing" in name:
                unobserved_concepts.append(parameter)
            else:
                observed_concepts.append(parameter)
        return observed_concepts, unobserved_concepts

    def __repr__(self):
        observed_string = str([[round(val, 3) for val in sublist] for sublist in (1 * self.observed_concepts.data).detach().cpu().numpy().tolist()])
        if self.is_grouped:
            observed_string = (
                observed_string + ", " + str([[round(val, 3) for val in sublist] for sublist in (1 * self.observed_grouped_concepts.data).detach().cpu().numpy().tolist()])
            )
        if self.use_unobserved:
            unobserved_string = str([round(val, 3) for val in (1 * self.unobserved_concepts.data).detach().cpu().numpy().tolist()])
            return "(" + unobserved_string + ", " + observed_string + ")"
        else:
            if self.use_missing_values:
                missing_observed_string = str([round(val, 3) for val in (1 * self.missing_observed_concepts.data).detach().cpu().numpy().tolist()])
                return "(" + missing_observed_string + ", " + observed_string + ")"
            else:
                return "(" + observed_string + ")"

    def load_string(self, init_string):
        unobserved_observed_tuple = eval(init_string)
        if isinstance(unobserved_observed_tuple, list):
            self.is_grouped = False
            observed_concepts_lists = unobserved_observed_tuple
            self.observed_concepts.data = torch.tensor(observed_concepts_lists, device=self.device).float()
            self.nb_in_concepts = self.observed_concepts.data.size(1)
        elif len(unobserved_observed_tuple) == 1:  # is tuple
            self.is_grouped = False
            observed_concepts_lists = unobserved_observed_tuple[0]
            self.observed_concepts.data = torch.tensor(observed_concepts_lists, device=self.device).float()
            self.nb_in_concepts = self.observed_concepts.data.size(1)
        elif len(unobserved_observed_tuple) == 2:  # is tuple
            self.is_grouped = False
            if self.use_unobserved:
                unobserved_concepts_lists = unobserved_observed_tuple[0]
                self.unobserved_concepts.data = torch.tensor(unobserved_concepts_lists, device=self.device)
            else:  # self.use_missing_values
                missing_observed_concepts_lists = unobserved_observed_tuple[0]
                self.missing_observed_concepts.data = torch.tensor(missing_observed_concepts_lists, device=self.device)
            observed_concepts_lists = unobserved_observed_tuple[1]
            self.observed_concepts.data = torch.tensor(observed_concepts_lists, device=self.device).float()
            self.nb_in_concepts = self.observed_concepts.data.size(1)
        else:
            self.is_grouped = True
            unobserved_concepts_lists = unobserved_observed_tuple[0]
            observed_concepts_lists = unobserved_observed_tuple[1]
            observed_grouped_concepts_lists = unobserved_observed_tuple[2]
            self.unobserved_concepts.data = torch.tensor(unobserved_concepts_lists, device=self.device)
            self.observed_concepts.data = torch.tensor(observed_concepts_lists, device=self.device).float()
            self.observed_grouped_concepts.data = torch.tensor(observed_grouped_concepts_lists, device=self.device).float()
            self.nb_in_concepts = self.observed_concepts.data.size(1) + self.observed_grouped_concepts.data.numel()
            self.nb_ungrouped_in_concepts = self.observed_concepts.data.size(1)
        self.nb_out_concepts = self.observed_concepts.data.size(0)

    def backward_simplify(self, used_out_concepts=[-1], keep_all_in_concepts=False):
        if self.is_grouped:
            self.ungroup()
        if used_out_concepts == [-1]:
            used_out_concepts = list(range(self.nb_out_concepts))
        if len(used_out_concepts) > 0:
            self.nb_out_concepts = len(used_out_concepts)
            self.observed_concepts.data = self.observed_concepts.data[used_out_concepts, :]
            if self.use_unobserved:
                self.unobserved_concepts.data = self.unobserved_concepts.data[used_out_concepts]
            if self.use_missing_values:
                self.missing_observed_concepts.data = self.missing_observed_concepts.data[used_out_concepts]
            used_in_concepts = torch.nonzero(torch.max(torch.abs(self.observed_concepts.data), dim=0)[0]).view(-1).tolist()
        else:
            self.nb_out_concepts = 1
            self.observed_concepts.data = torch.zeros((self.nb_in_concepts, 1), device=self.device)
            if self.use_unobserved:
                self.unobserved_concepts.data = (1 - self.full_unobserved_value) * torch.ones((1), device=self.device)
            if self.use_missing_values:
                self.missing_observed_concepts.data = torch.zeros((1), device=self.device)
            used_in_concepts = []
        if keep_all_in_concepts:
            pass
        elif len(used_in_concepts) > 0:
            self.nb_in_concepts = len(used_in_concepts)
            self.observed_concepts.data = self.observed_concepts.data[:, used_in_concepts]
        else:
            self.nb_in_concepts = 1
            self.observed_concepts.data = self.observed_concepts.data[:, [0]]
        return used_in_concepts

    def forward_simplify(
        self,
        used_in_concepts=[-1],
        unused_in_idx_prob_pairs=[],
        keep_all_out_concepts=False,
        in_concepts_group_first_stop_pairs=[-1],
    ):
        if in_concepts_group_first_stop_pairs != [-1]:
            self.in_concepts_group_first_stop_pairs = in_concepts_group_first_stop_pairs
            if USE_RULE_MODULE and len(self.in_concepts_group_first_stop_pairs) > 0:
                self.nb_ungrouped_in_concepts = self.in_concepts_group_first_stop_pairs[0][0]
        unused_out_idx_prob_pairs = []
        if not self.use_unobserved:
            extra_unused_out_concepts = []
        if len(unused_in_idx_prob_pairs) > 0:
            for unused_in_concept, prob in unused_in_idx_prob_pairs:
                for out_concept in range(self.nb_out_concepts):
                    if self.use_unobserved:
                        self.update_unobserved_concepts(out_concept, unused_in_concept, prob)
                    else:
                        if prob == 0 or prob == 1:
                            if self.observed_concepts.data[out_concept, unused_in_concept] == 1:
                                if self.is_AND:
                                    if prob == 0:
                                        unused_out_idx_prob_pairs.append((out_concept, 0))
                                        extra_unused_out_concepts.append(out_concept)
                                else:
                                    if prob == 1:
                                        unused_out_idx_prob_pairs.append((out_concept, 1))
                                        extra_unused_out_concepts.append(out_concept)
                            elif self.observed_concepts.data[out_concept, unused_in_concept] == -1:
                                if self.is_AND:
                                    if prob == 1:
                                        unused_out_idx_prob_pairs.append((out_concept, 0))
                                        extra_unused_out_concepts.append(out_concept)
                                else:
                                    if prob == 0:
                                        unused_out_idx_prob_pairs.append((out_concept, 1))
                                        extra_unused_out_concepts.append(out_concept)
                            elif self.observed_concepts.data[out_concept, unused_in_concept] != 0:
                                warnings.warn(
                                    "Case where deleted in-concept with observed concept not 0, 1 or -1 for layer without unobserved concepts is ambiguous. observed concept of "
                                    + str(
                                        round(
                                            self.observed_concepts.data[out_concept, unused_in_concept].item(),
                                            3,
                                        )
                                    )
                                    + " is ignored."
                                )
                        else:
                            warnings.warn(
                                "Case where deleted in-concept with probability not 0 or 1 for layer without unobserved concepts is ambiguous. Probability of "
                                + str(round(prob, 3))
                                + " is ignored."
                            )
        if used_in_concepts == [-1]:
            used_in_concepts = list(range(self.nb_in_concepts))
        if len(used_in_concepts) > 0:
            self.nb_in_concepts = len(used_in_concepts)
            self.observed_concepts.data = self.observed_concepts.data[:, used_in_concepts]
            used_out_concepts = torch.nonzero(torch.max(torch.abs(self.observed_concepts.data), dim=1)[0]).view(-1).tolist()
            if self.use_unobserved:
                used_out_concepts = [used_out_concept for used_out_concept in used_out_concepts if self.unobserved_concepts.data[used_out_concept] > 0]
            else:
                used_out_concepts = [used_out_concept for used_out_concept in used_out_concepts if not used_out_concept in extra_unused_out_concepts]
            if self.use_missing_values:
                used_missing_out_concepts = torch.nonzero(self.missing_observed_concepts.data).view(-1).tolist()
                used_out_concepts = sorted(list(set(used_out_concepts) | set(used_missing_out_concepts)))
        else:
            self.nb_in_concepts = 1
            if keep_all_out_concepts:
                self.observed_concepts.data = torch.zeros((self.nb_out_concepts, self.nb_in_concepts), device=self.device)
            else:
                self.observed_concepts.data = torch.zeros((1, self.nb_in_concepts), device=self.device)
            used_out_concepts = []
        if keep_all_out_concepts:
            used_out_concepts = list(range(self.nb_out_concepts))
        else:
            unused_out_concepts = [concept_idx for concept_idx in range(self.nb_out_concepts) if not concept_idx in used_out_concepts]
            for unused_out_concept in unused_out_concepts:
                if self.use_unobserved:
                    unused_out_idx_prob_pairs.append(
                        (
                            unused_out_concept,
                            self.unobserved_concepts.data[unused_out_concept].item(),
                        )
                    )
                else:
                    if not unused_out_concept in extra_unused_out_concepts:
                        unused_out_idx_prob_pairs.append((unused_out_concept, self.full_unobserved_value))
            if len(used_out_concepts) > 0:
                self.nb_out_concepts = len(used_out_concepts)
                self.observed_concepts.data = self.observed_concepts.data[used_out_concepts, :]
                if self.use_unobserved:
                    self.unobserved_concepts.data = self.unobserved_concepts.data[used_out_concepts]
                if self.use_missing_values:
                    self.missing_observed_concepts.data = self.missing_observed_concepts.data[used_out_concepts]
            else:
                self.nb_out_concepts = 1
                self.observed_concepts.data = self.observed_concepts.data[[0], :]
                if self.use_unobserved:
                    self.unobserved_concepts.data = (1 - self.full_unobserved_value) * torch.ones((1), device=self.device)
                if self.use_missing_values:
                    self.missing_observed_concepts.data = torch.zeros((1), device=self.device)
        return used_out_concepts, unused_out_idx_prob_pairs

    def remove_duplicates(
        self,
        in_duplicate_lists=[],
        in_tentative_duplicate_lists=-1,
        keep_all_out_concepts=False,
    ):
        if len(in_duplicate_lists) > 0:
            used_in_concepts = list(range(self.nb_in_concepts))
            for duplicate_list in in_duplicate_lists:
                kept_idx = duplicate_list[0]
                for unkept_idx in duplicate_list[1:]:
                    used_in_concepts.remove(unkept_idx)
                    for out_idx in range(self.nb_out_concepts):
                        if self.observed_concepts.data[out_idx, kept_idx] == 0:
                            self.observed_concepts.data[out_idx, kept_idx] = self.observed_concepts.data[out_idx, unkept_idx]
            self.nb_in_concepts = len(used_in_concepts)
            self.observed_concepts.data = self.observed_concepts.data[:, used_in_concepts]
        if isinstance(in_tentative_duplicate_lists, list):
            in_confirmed_duplicate_lists = []
            for in_tentative_duplicate_list in in_tentative_duplicate_lists:
                in_i = in_tentative_duplicate_list[0]
                in_confirmed_duplicate_list = [in_i]
                for in_j_idx, in_j in enumerate(in_tentative_duplicate_list[1:]):
                    if torch.equal(
                        self.observed_concepts.data[:, in_i],
                        self.observed_concepts.data[:, in_j],
                    ):
                        in_confirmed_duplicate_list.append(in_j)
                    elif len(in_tentative_duplicate_list) - in_j_idx >= 2:
                        in_tentative_duplicate_lists.append(in_tentative_duplicate_list[in_j_idx:])
                if len(in_confirmed_duplicate_list) > 1:
                    in_confirmed_duplicate_lists.append(in_confirmed_duplicate_list)
            self._remove_in_confirmed_duplicates(in_confirmed_duplicate_lists)
        if not keep_all_out_concepts:
            if not isinstance(in_tentative_duplicate_lists, list):
                if not self.use_unobserved:
                    out_duplicate_lists = []
                    already_duplicates = []
                    for out_i in range(self.nb_out_concepts):
                        if not out_i in already_duplicates:
                            duplicate_list = [out_i]
                            for out_j in range(out_i + 1, self.nb_out_concepts):
                                if torch.equal(self.observed_concepts.data[out_i, :], self.observed_concepts.data[out_j, :]) and (
                                    not self.use_missing_values or torch.equal(self.missing_observed_concepts.data[out_i], self.missing_observed_concepts.data[out_j])
                                ):
                                    duplicate_list.append(out_j)
                                    already_duplicates.append(out_j)
                            if len(duplicate_list) > 1:
                                out_duplicate_lists.append(duplicate_list)
                    self._remove_out_confirmed_duplicates(out_duplicate_lists)
                    return out_duplicate_lists
            else:
                out_tentative_duplicate_lists = []
                already_duplicates = []
                for out_i in range(self.nb_out_concepts):
                    if not out_i in already_duplicates:
                        duplicate_list = [out_i]
                        for out_j in range(out_i + 1, self.nb_out_concepts):
                            if torch.equal(
                                self.observed_concepts.data[out_i, :],
                                self.observed_concepts.data[out_j, :],
                            ):
                                duplicate_list.append(out_j)
                                already_duplicates.append(out_j)
                        if len(duplicate_list) > 1:
                            out_tentative_duplicate_lists.append(duplicate_list)
                return out_tentative_duplicate_lists, in_confirmed_duplicate_lists
        elif isinstance(in_tentative_duplicate_lists, list):
            return in_confirmed_duplicate_lists

    def _remove_in_confirmed_duplicates(self, in_confirmed_duplicate_lists):
        if len(in_confirmed_duplicate_lists) > 0:
            used_in_concepts = list(range(self.nb_in_concepts))
            for duplicate_list in in_confirmed_duplicate_lists:
                for unkept_idx in duplicate_list[1:]:
                    used_in_concepts.remove(unkept_idx)
            self.nb_in_concepts = len(used_in_concepts)
            self.observed_concepts.data = self.observed_concepts.data[:, used_in_concepts]

    def _remove_out_confirmed_duplicates(self, out_confirmed_duplicate_lists):
        if len(out_confirmed_duplicate_lists) > 0:
            used_out_concepts = list(range(self.nb_out_concepts))
            for duplicate_list in out_confirmed_duplicate_lists:
                kept_idx = duplicate_list[0]
                for unkept_idx in duplicate_list[1:]:
                    used_out_concepts.remove(unkept_idx)
                if self.use_unobserved:
                    self.unobserved_concepts.data[kept_idx] = torch.mean(self.unobserved_concepts.data[duplicate_list])
            self.nb_out_concepts = len(used_out_concepts)
            self.observed_concepts.data = self.observed_concepts.data[used_out_concepts, :]
            if self.use_unobserved:
                self.unobserved_concepts.data = self.unobserved_concepts.data[used_out_concepts]
            if self.use_missing_values:
                self.missing_observed_concepts.data = self.missing_observed_concepts.data[used_out_concepts]

    def duplicate_out_concept_with_new_biases(self, old_rule_idx: int, new_biases: List[float]):
        self.nb_out_concepts = self.nb_out_concepts + (len(new_biases) - 1)

        old_observed_concepts = 1 * self.observed_concepts.data
        self.observed_concepts.data = torch.zeros((self.nb_out_concepts, self.nb_in_concepts)).to(self.device)
        self.observed_concepts.data[:old_rule_idx, :] = old_observed_concepts[:old_rule_idx, :]
        for new_bias_idx in range(len(new_biases)):
            self.observed_concepts.data[old_rule_idx + new_bias_idx, :] = old_observed_concepts[old_rule_idx, :]
        self.observed_concepts.data[old_rule_idx + len(new_biases) :, :] = old_observed_concepts[old_rule_idx + 1 :, :]

        if self.use_unobserved:
            old_unobserved_concepts = 1 * self.unobserved_concepts.data
            self.unobserved_concepts.data = torch.zeros((self.nb_out_concepts)).to(self.device)
            self.unobserved_concepts.data[:old_rule_idx] = old_unobserved_concepts[:old_rule_idx]
            for new_bias_idx, new_bias in enumerate(new_biases):
                self.unobserved_concepts.data[old_rule_idx + new_bias_idx] = new_bias
            self.unobserved_concepts.data[old_rule_idx + len(new_biases) :] = old_unobserved_concepts[old_rule_idx + 1 :]

        if self.use_missing_values:
            old_missing_observed_concepts = 1 * self.missing_observed_concepts.data
            self.missing_observed_concepts.data = torch.zeros((self.nb_out_concepts)).to(self.device)
            self.missing_observed_concepts.data[:old_rule_idx] = old_missing_observed_concepts[:old_rule_idx]
            for new_bias_idx, new_bias in enumerate(new_biases):
                self.missing_observed_concepts.data[old_rule_idx + new_bias_idx] = old_missing_observed_concepts[old_rule_idx]
            self.missing_observed_concepts.data[old_rule_idx + len(new_biases) :] = old_missing_observed_concepts[old_rule_idx + 1 :]

    def duplicate_in_concept_with_new_uses(self, old_rule_idx: int, new_uses: List[List[int]]):
        self.nb_in_concepts = self.nb_in_concepts + (len(new_uses) - 1)

        old_observed_concepts = 1 * self.observed_concepts.data
        self.observed_concepts.data = torch.zeros((self.nb_out_concepts, self.nb_in_concepts)).to(self.device)
        self.observed_concepts.data[:, :old_rule_idx] = old_observed_concepts[:, :old_rule_idx]
        for new_in_copy_uses_idx, new_in_copy_uses in enumerate(new_uses):
            for new_use in new_in_copy_uses:
                self.observed_concepts.data[new_use, old_rule_idx + new_in_copy_uses_idx] = old_observed_concepts[new_use, old_rule_idx]
        self.observed_concepts.data[:, old_rule_idx + len(new_uses) :] = old_observed_concepts[:, old_rule_idx + 1 :]

    def prune(self, eval_model_func, init_loss=-1, log_files=[], progress_bar_hook=lambda weight: None):
        did_prune = False
        if init_loss < 0:
            init_loss = eval_model_func()
        current_weight = 0
        for out_idx in range(self.nb_out_concepts):
            for in_idx in range(self.nb_in_concepts):
                if self.observed_concepts.data[out_idx, in_idx] != 0:
                    old_value = self.observed_concepts.data[out_idx, in_idx].item()
                    self.observed_concepts.data[out_idx, in_idx] = 0
                    new_loss = eval_model_func()
                    if init_loss < new_loss:
                        self.observed_concepts.data[out_idx, in_idx] = old_value
                        print_log(
                            f"Kept {self.junction_display_string}_concepts.observed_concept[{out_idx}, {in_idx}] = {old_value:.3f}",
                            self.verbose,
                            log_files,
                        )
                    else:
                        did_prune = True
                        tmp_extra_string_space = " " if old_value >= 0 else ""
                        print_log(
                            f"Pruned {self.junction_display_string}_concepts.observed_concept[{out_idx}, {in_idx}] = {tmp_extra_string_space}{old_value:.3f} -> 0     (new loss = {new_loss})",
                            self.verbose,
                            log_files,
                        )
                        init_loss = new_loss
                    current_weight += 1
                    progress_bar_hook(current_weight)
        return init_loss, did_prune

    def get_nb_weights(self):
        if not self.is_grouped:
            return torch.count_nonzero(self.observed_concepts.data).item()
        else:
            return torch.count_nonzero(self.observed_concepts.data).item() + torch.count_nonzero(self.observed_grouped_concepts.data).item()

    def discretize(self, discretization_method, eval_model_func=lambda: None, log_files=[], progress_bar_hook=lambda weight: None):
        if self.is_grouped:
            self.ungroup()
        if set(torch.abs(self.observed_concepts.data).view(-1).tolist()).issubset({0, 1}):
            return []
        elif discretization_method == "thresh":
            self._discretize_thresh(progress_bar_hook=progress_bar_hook)
        elif discretization_method[:5] == "stoch":
            return self._discretize_stoch(discretization_method[5:], eval_model_func, log_files=log_files, progress_bar_hook=progress_bar_hook)
        elif discretization_method[:7] == "qthresh":
            return self._discretize_qthresh(discretization_method[7:], eval_model_func, log_files=log_files, progress_bar_hook=progress_bar_hook)
        elif discretization_method == "sub":
            return self._discretize_sub(eval_model_func, log_files=log_files, progress_bar_hook=progress_bar_hook)
        elif discretization_method == "add":
            return self._discretize_add(eval_model_func, log_files=log_files, progress_bar_hook=progress_bar_hook)
        elif discretization_method[:4] == "sel_":
            return self._discretize_sel(discretization_method[4:], eval_model_func, log_files=log_files, progress_bar_hook=progress_bar_hook)
        else:
            raise Exception("discretization_method not legal!")

    def _discretize_thresh(self, progress_bar_hook=lambda weight: None):
        self.observed_concepts.data = self._threshold_weights(self.observed_concepts.data)
        progress_bar_hook(self.get_nb_weights())

    def _discretize_stoch(self, nb_attempts_str, eval_model_func, log_files=[], progress_bar_hook=lambda weight: None):
        losses = []
        current_weight = 0
        weight_incr = self.get_nb_weights() / self.nb_out_concepts
        if nb_attempts_str == "In":
            nb_attempts = 2 * self.nb_in_concepts
        else:
            nb_attempts = int(nb_attempts_str)
        for out_concept in range(self.nb_out_concepts):
            old_observed_concepts = 1 * self.observed_concepts.data[out_concept, :]
            sampled_observed_concepts_dict = dict()
            nb_sampled_observed_concepts = 0
            attempt_idx = 0
            while nb_sampled_observed_concepts < nb_attempts and attempt_idx < 2 * nb_attempts:
                curr_sampled_observed_concepts = self._sample_weights(old_observed_concepts)
                curr_used_idcs = torch.nonzero(curr_sampled_observed_concepts).view(-1).tolist()
                curr_first_used_idx = curr_used_idcs[0] if curr_used_idcs != [] else curr_sampled_observed_concepts.size(0)
                is_duplicate = False
                if curr_first_used_idx in sampled_observed_concepts_dict:
                    for other_sampled_observed_concepts in sampled_observed_concepts_dict[curr_first_used_idx]:
                        if (curr_sampled_observed_concepts == other_sampled_observed_concepts).all():
                            is_duplicate = True
                            break
                if not is_duplicate:
                    if curr_first_used_idx not in sampled_observed_concepts_dict:
                        sampled_observed_concepts_dict[curr_first_used_idx] = [curr_sampled_observed_concepts]
                    else:
                        sampled_observed_concepts_dict[curr_first_used_idx].append(curr_sampled_observed_concepts)
                    nb_sampled_observed_concepts += 1
                attempt_idx += 1
            best_loss = float("inf")
            sampled_observed_concepts = []
            for first_used_idx, sampled_observed_concepts_list in sampled_observed_concepts_dict.items():
                sampled_observed_concepts += sampled_observed_concepts_list
            for curr_sampled_observed_concepts in sampled_observed_concepts:
                self.observed_concepts.data[out_concept, :] = curr_sampled_observed_concepts
                curr_loss = eval_model_func()
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_sampled_observed_concepts = curr_sampled_observed_concepts
            self.observed_concepts.data[out_concept, :] = best_sampled_observed_concepts
            losses.append(best_loss)
            print_log(
                f"Best sampled discrete weights {self.junction_display_string}_concepts.observed_concept[{out_concept}, :] found out of {len(sampled_observed_concepts)} ({attempt_idx} attempts) = {[round(value) for value in best_sampled_observed_concepts.tolist()]}        (new loss = {losses[-1]})",
                self.verbose,
                log_files,
            )
            current_weight += weight_incr
            progress_bar_hook(current_weight)
        return losses

    def _discretize_qthresh(self, nb_attempts_str, eval_model_func, log_files=[], progress_bar_hook=lambda weight: None):
        losses = []
        current_weight = 0
        weight_incr = self.get_nb_weights() / self.nb_out_concepts
        if nb_attempts_str == "In":
            nb_attempts = 2 * self.nb_in_concepts
        else:
            nb_attempts = int(nb_attempts_str)
        for out_concept in range(self.nb_out_concepts):
            old_observed_concepts = 1 * self.observed_concepts.data[out_concept, :]
            old_observed_concepts_abs_values = torch.abs(old_observed_concepts).view(-1).tolist()
            old_observed_concepts_abs_values_set = set(old_observed_concepts_abs_values)
            max_nb_quantiled = len(old_observed_concepts_abs_values_set)
            quantiled_observed_concepts = []
            if max_nb_quantiled < nb_attempts:
                for min_abs_value in old_observed_concepts_abs_values_set:
                    quantiled_observed_concepts.append(self._threshold_weights(old_observed_concepts, threshold=min_abs_value))
            else:
                old_observed_concepts_abs_values.sort()
                grid_size = nb_attempts - 1
                nb_distinct = 0
                while nb_distinct < nb_attempts:
                    grid_size += 1
                    quantiles = (np.linspace(0, len(old_observed_concepts_abs_values), grid_size + 2).tolist())[1:-1]
                    quantile_min_values = set([old_observed_concepts_abs_values[int(floor(quantile))] for quantile in quantiles])
                    nb_distinct = len(quantile_min_values)
                quantile_min_values = list(quantile_min_values)
                quantile_min_values = (
                    quantile_min_values
                    if len(quantile_min_values) == nb_attempts
                    else [quantile_min_values[idx] for idx in random.sample(list(range(len(quantile_min_values))), nb_attempts)]
                )
                quantile_min_values.sort()
                for min_abs_value in quantile_min_values:
                    quantiled_observed_concepts.append(self._threshold_weights(old_observed_concepts, threshold=min_abs_value))
            best_loss = float("inf")
            for curr_quantiled_observed_concepts in quantiled_observed_concepts:
                self.observed_concepts.data[out_concept, :] = curr_quantiled_observed_concepts
                curr_loss = eval_model_func()
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_quantiled_observed_concepts = curr_quantiled_observed_concepts
            self.observed_concepts.data[out_concept, :] = best_quantiled_observed_concepts
            losses.append(best_loss)
            print_log(
                f"Best quantiled discrete weights {self.junction_display_string}_concepts.observed_concept[{out_concept}, :] found out of {len(quantiled_observed_concepts)} = {best_quantiled_observed_concepts.tolist()}        (new loss = {losses[-1]})",
                self.verbose,
                log_files,
            )
            current_weight += weight_incr
            progress_bar_hook(current_weight)
        return losses

    def _discretize_sub(self, eval_model_func, log_files=[], progress_bar_hook=lambda weight: None):
        losses = []
        current_weight = 0
        observed_concepts_data = self.observed_concepts.data
        old_observed_concepts = 1 * observed_concepts_data
        old_observed_concepts_abs_values = list(set(torch.abs(old_observed_concepts).view(-1).tolist()))
        old_observed_concepts_abs_values.sort()
        observed_concepts_data[observed_concepts_data > 0] = 1
        observed_concepts_data[observed_concepts_data < 0] = -1
        losses.append(eval_model_func())
        print_log(
            f"Discretized {self.junction_display_string}_concepts.observed_concepts to +1/-1       (new loss = {losses[-1]})",
            self.verbose,
            log_files,
        )
        for old_observed_concept_abs_value in old_observed_concepts_abs_values:
            if old_observed_concept_abs_value > 0:
                param_out_ins = ((old_observed_concepts == old_observed_concept_abs_value) + (old_observed_concepts == -1 * old_observed_concept_abs_value)).nonzero().tolist()
                for param_out_in in param_out_ins:
                    param_out, param_in = tuple(param_out_in)
                    old_discretized_value = observed_concepts_data[param_out, param_in].item()
                    observed_concepts_data[param_out, param_in] = 0
                    new_loss = eval_model_func()
                    if losses[-1] < new_loss:
                        observed_concepts_data[param_out, param_in] = old_discretized_value
                        losses.append(losses[-1])
                        tmp_extra_string_space = " " if old_discretized_value == 1 else ""
                        print_log(
                            f"Kept discrete {self.junction_display_string}_concepts.observed_concept[{param_out}, {param_in}] = {tmp_extra_string_space}{old_observed_concepts[param_out,param_in].item():.3f} -> {tmp_extra_string_space}{old_discretized_value:.0f}",
                            self.verbose,
                            log_files,
                        )
                    else:
                        losses.append(new_loss)
                        print_log(
                            f"Pruned {self.junction_display_string}_concepts.observed_concept[{param_out}, {param_in}] = {old_observed_concepts[param_out,param_in].item():.3f} -> 0     (new loss = {losses[-1]})",
                            self.verbose,
                            log_files,
                        )
                    current_weight += 1
                    progress_bar_hook(current_weight)
        return losses

    def _discretize_add(self, eval_model_func, log_files=[], progress_bar_hook=lambda weight: None):
        losses = []
        current_weight = 0
        observed_concepts_data = self.observed_concepts.data
        old_observed_concepts = 1 * observed_concepts_data
        old_observed_concepts_abs_values = list(set(torch.abs(old_observed_concepts).view(-1).tolist()))
        old_observed_concepts_abs_values.sort(reverse=True)
        observed_concepts_data[:, :] = torch.zeros_like(observed_concepts_data)
        losses.append(eval_model_func())
        print_log(
            f"Zeroed {self.junction_display_string}_concepts.observed_concepts          (new loss = {losses[-1]})",
            self.verbose,
            log_files,
        )
        for old_observed_concept_abs_value in old_observed_concepts_abs_values:
            if old_observed_concept_abs_value > 0:
                param_out_ins = ((old_observed_concepts == old_observed_concept_abs_value) + (old_observed_concepts == -1 * old_observed_concept_abs_value)).nonzero().tolist()
                for param_out_in in param_out_ins:
                    param_out, param_in = tuple(param_out_in)
                    if old_observed_concepts[param_out, param_in] > 0:
                        observed_concepts_data[param_out, param_in] = 1
                        tmp_extra_string_space = " "
                    else:
                        observed_concepts_data[param_out, param_in] = -1
                        tmp_extra_string_space = ""
                    new_loss = eval_model_func()
                    if losses[-1] < new_loss:
                        observed_concepts_data[param_out, param_in] = 0
                        losses.append(losses[-1])
                        print_log(
                            f"Kept Zeroed {self.junction_display_string}_concepts.observed_concept[{param_out}, {param_in}] = {old_observed_concepts[param_out,param_in].item():.3f} -> 0",
                            self.verbose,
                            log_files,
                        )
                    else:
                        losses.append(new_loss)
                        print_log(
                            f"Added discrete {self.junction_display_string}_concepts.observed_concept[{param_out}, {param_in}] = {tmp_extra_string_space}{old_observed_concepts[param_out,param_in].item():.3f} -> {tmp_extra_string_space}{observed_concepts_data[param_out, param_in].item():.0f}     (new loss = {losses[-1]})",
                            self.verbose,
                            log_files,
                        )
                    current_weight += 1
                    progress_bar_hook(current_weight)
        return losses

    def _discretize_sel(self, sel_type, eval_model_func, log_files=[], progress_bar_hook=lambda weight: None):
        losses = []
        current_weight = 0
        observed_concepts_data = self.observed_concepts.data
        old_observed_concepts = 1 * observed_concepts_data
        old_observed_concepts_abs_values = list(set(torch.abs(old_observed_concepts).view(-1).tolist()))
        if sel_type == "asc":
            old_observed_concepts_abs_values.sort()
        else:  # sel_type == "desc":
            old_observed_concepts_abs_values.sort(reverse=True)
        for old_observed_concept_abs_value in old_observed_concepts_abs_values:
            if old_observed_concept_abs_value > 0:
                param_out_ins = ((old_observed_concepts == old_observed_concept_abs_value) + (old_observed_concepts == -1 * old_observed_concept_abs_value)).nonzero().tolist()
                for param_out_in in param_out_ins:
                    param_out, param_in = tuple(param_out_in)
                    observed_concepts_data[param_out, param_in] = 0
                    loss_without = eval_model_func()
                    if old_observed_concepts[param_out, param_in] > 0:
                        observed_concepts_data[param_out, param_in] = 1
                        tmp_extra_string_space = " "
                    else:
                        observed_concepts_data[param_out, param_in] = -1
                        tmp_extra_string_space = ""
                    loss_with = eval_model_func()
                    if loss_with < loss_without:
                        losses.append(loss_with)
                        print_log(
                            f"Discretized {self.junction_display_string}_concepts.observed_concept[{param_out}, {param_in}] = {tmp_extra_string_space}{old_observed_concepts[param_out,param_in].item():.3f} -> {tmp_extra_string_space}{observed_concepts_data[param_out, param_in].item():.0f}     (new loss = {losses[-1]})",
                            self.verbose,
                            log_files,
                        )
                    else:
                        observed_concepts_data[param_out, param_in] = 0
                        losses.append(loss_without)
                        print_log(
                            f"Pruned {self.junction_display_string}_concepts.observed_concept[{param_out}, {param_in}] = {tmp_extra_string_space}{old_observed_concepts[param_out,param_in].item():.3f} ->  0        (new loss = {losses[-1]})",
                            self.verbose,
                            log_files,
                        )
                    current_weight += 1
                    progress_bar_hook(current_weight)
        return losses


class AndConcepts(CombinationConcepts):
    """AND concepts: present if all observed and unobserved necessary concepts are present"""

    def __init__(
        self,
        nb_in_concepts: int,
        nb_out_concepts: int,
        use_negation: bool = True,
        use_unobserved: bool = True,
        use_missing_values: bool = False,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        empty_reset_in_concepts: bool = EMPTY_RESET_IN_CONCEPTS,
        train_forw_weight_quant: str = TRAIN_FORW_WEIGHT_QUANT,
        approx_AND_OR_params: Union[None, Tuple[float, float, float]] = APPROX_PARAMS,
        in_concepts_group_first_stop_pairs: List[Tuple[int, int]] = [],
        device=DEVICE,
        verbose: bool = VERBOSE,
    ):
        super().__init__(
            True,
            nb_in_concepts,
            nb_out_concepts,
            use_negation=use_negation,
            use_unobserved=use_unobserved,
            use_missing_values=use_missing_values,
            random_init_obs=random_init_obs,
            random_init_unobs=random_init_unobs,
            empty_reset_in_concepts=empty_reset_in_concepts,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            in_concepts_group_first_stop_pairs=in_concepts_group_first_stop_pairs,
            device=device,
            verbose=verbose,
        )

    def combine_observed_concepts(self, x_v, observed_concepts_v):
        if not self.use_negation:
            result = torch.prod(1 - observed_concepts_v * (1 - x_v), dim=-1)
        else:
            if not self.training:
                result = torch.prod(1 - F.relu(observed_concepts_v) * (1 - x_v) - F.relu(-1 * observed_concepts_v) * x_v, dim=-1)
            else:
                with torch.no_grad():
                    one_hot_equals_0 = torch.zeros((len(observed_concepts_v.shape) - 2) * [1] + list(observed_concepts_v.shape[-2:])).to(self.device)
                    one_hot_equals_0[observed_concepts_v == 0] = 1
                result = torch.prod(
                    1 - F.relu(observed_concepts_v) * (1 - x_v) - F.relu(-1 * observed_concepts_v) * x_v + one_hot_equals_0 * observed_concepts_v * 2 * (x_v - 0.5),
                    dim=-1,
                )
        return result

    def combine_observed_concepts_approximately(self, x_v, observed_concepts, combine_grouped=False):
        alpha, beta, gamma = self.approx_AND_OR_params
        g_not_x = 1.0 - 1.0 / (1.0 - (alpha * (1 - x_v)) ** beta)
        if not self.use_negation:
            g_W = 1.0 - 1.0 / (1.0 - (observed_concepts * alpha) ** beta)
            if not combine_grouped:
                return g_not_x @ g_W.T
            else:
                return torch.sum(g_not_x * g_W, dim=-1)
        else:
            g_x = 1.0 - 1.0 / (1.0 - (alpha * x_v) ** beta)
            g_W_pos = 1.0 - 1.0 / (1.0 - (alpha * F.relu(observed_concepts)) ** beta)
            g_W_neg = 1.0 - 1.0 / (1.0 - (alpha * F.relu(-1 * observed_concepts)) ** beta)
            if not combine_grouped:
                print(g_not_x.shape, g_W_pos.T.shape)
                print(g_x.shape, g_W_neg.T.shape)
                return g_not_x @ g_W_pos.T + g_x @ g_W_neg.T
            else:
                return torch.sum(g_not_x * g_W_pos + g_x * g_W_neg, dim=-1)

    def combine_with_unobserved_concepts(self, result):
        if self.use_unobserved:
            unobserved_concepts_v = self.unobserved_concepts.view((len(result.shape) - 1) * [1] + [self.nb_out_concepts])
            result = unobserved_concepts_v * result
        return result

    def update_unobserved_concepts(self, out_concept, unused_in_concept, unused_in_prob):
        if not self.is_grouped or unused_in_concept < self.nb_ungrouped_in_concepts:
            self.unobserved_concepts.data[out_concept] *= (
                1
                - F.relu(self.observed_concepts.data[out_concept, unused_in_concept]) * (1 - unused_in_prob)
                - F.relu(-1 * self.observed_concepts.data[out_concept, unused_in_concept]) * unused_in_prob
            )
        else:
            unused_in_grouped_concept = (unused_in_concept - self.nb_ungrouped_in_concepts) % self.nb_out_concepts
            self.unobserved_concepts.data[out_concept] *= (
                1
                - F.relu(self.observed_grouped_concepts.data[out_concept, unused_in_grouped_concept]) * (1 - unused_in_prob)
                - F.relu(-1 * self.observed_grouped_concepts.data[out_concept, unused_in_grouped_concept]) * unused_in_prob
            )


class OrConcepts(CombinationConcepts):
    """OR concepts: present if any observed or unobserved sufficient concept is present"""

    def __init__(
        self,
        nb_in_concepts: int,
        nb_out_concepts: int,
        use_negation: bool = True,
        use_unobserved: bool = True,
        use_missing_values: bool = False,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        empty_reset_in_concepts: bool = EMPTY_RESET_IN_CONCEPTS,
        train_forw_weight_quant: str = TRAIN_FORW_WEIGHT_QUANT,
        approx_AND_OR_params: Union[None, Tuple[float, float, float]] = APPROX_PARAMS,
        in_concepts_group_first_stop_pairs: List[Tuple[int, int]] = [],
        device=DEVICE,
        verbose: bool = VERBOSE,
    ):
        super().__init__(
            False,
            nb_in_concepts,
            nb_out_concepts,
            use_negation=use_negation,
            use_unobserved=use_unobserved,
            use_missing_values=use_missing_values,
            random_init_obs=random_init_obs,
            random_init_unobs=random_init_unobs,
            empty_reset_in_concepts=empty_reset_in_concepts,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            in_concepts_group_first_stop_pairs=in_concepts_group_first_stop_pairs,
            device=device,
            verbose=verbose,
        )

    def combine_observed_concepts(self, x_v, observed_concepts_v):
        if not self.use_negation:
            result = torch.prod(1 - observed_concepts_v * x_v, dim=-1)
        else:
            if not self.training:
                result = torch.prod(1 - F.relu(observed_concepts_v) * x_v - F.relu(-1 * observed_concepts_v) * (1 - x_v), dim=-1)
            else:
                with torch.no_grad():
                    one_hot_equals_0 = torch.zeros((len(observed_concepts_v.shape) - 2) * [1] + list(observed_concepts_v.shape[-2:])).to(self.device)
                    one_hot_equals_0[observed_concepts_v == 0] = 1
                result = torch.prod(
                    1 - F.relu(observed_concepts_v) * x_v - F.relu(-1 * observed_concepts_v) * (1 - x_v) - one_hot_equals_0 * observed_concepts_v * 2 * (x_v - 0.5),
                    dim=-1,
                )
        return result

    def combine_observed_concepts_approximately(self, x_v, observed_concepts, combine_grouped=False):
        alpha, beta, gamma = self.approx_AND_OR_params
        g_x = 1.0 - 1.0 / (1.0 - (alpha * x_v) ** beta)
        if not self.use_negation:
            g_W = 1.0 - 1.0 / (1.0 - (observed_concepts * alpha) ** beta)
            if not combine_grouped:
                return g_x @ g_W.T
            else:
                return torch.sum(g_x * g_W, dim=-1)
        else:
            g_not_x = 1.0 - 1.0 / (1.0 - (alpha * (1 - x_v)) ** beta)
            g_W_pos = 1.0 - 1.0 / (1.0 - (alpha * F.relu(observed_concepts)) ** beta)
            g_W_neg = 1.0 - 1.0 / (1.0 - (alpha * F.relu(-1 * observed_concepts)) ** beta)
            if not combine_grouped:
                return g_x @ g_W_pos.T + g_not_x @ g_W_neg.T
            else:
                return torch.sum(g_x * g_W_pos + g_not_x * g_W_neg, dim=-1)

    def combine_with_unobserved_concepts(self, result):
        if self.use_unobserved:
            unobserved_concepts_v = self.unobserved_concepts.view((len(result.shape) - 1) * [1] + [self.nb_out_concepts])
            result = 1 - (1 - unobserved_concepts_v) * result
        else:
            result = 1 - result
        return result

    def update_unobserved_concepts(self, out_concept, unused_in_concept, unused_in_prob):
        if not self.is_grouped or unused_in_concept < self.nb_ungrouped_in_concepts:
            self.unobserved_concepts.data[out_concept] = 1 - (1 - self.unobserved_concepts.data[out_concept]) * (
                1
                - F.relu(self.observed_concepts.data[out_concept, unused_in_concept]) * unused_in_prob
                - F.relu(-1 * self.observed_concepts.data[out_concept, unused_in_concept]) * (1 - unused_in_prob)
            )
        else:
            unused_in_grouped_concept = (unused_in_concept - self.nb_ungrouped_in_concepts) % self.nb_out_concepts
            self.unobserved_concepts.data[out_concept] = 1 - (1 - self.unobserved_concepts.data[out_concept]) * (
                1
                - F.relu(self.observed_grouped_concepts.data[out_concept, unused_in_grouped_concept]) * unused_in_prob
                - F.relu(-1 * self.observed_grouped_concepts.data[out_concept, unused_in_grouped_concept]) * (1 - unused_in_prob)
            )


class Dichotomies(nn.Module):
    """Fuzzy dichotomies: present to some degree when associated continuous input feature is greater than the boundary"""

    def __init__(
        self,
        nb_dichotomies: int,
        min_value: float,
        max_value: float,
        device=DEVICE,
    ):
        super().__init__()
        self.device = device
        self.nb_dichotomies = nb_dichotomies
        self.boundaries = nn.Parameter(torch.zeros(nb_dichotomies))
        self.boundaries.data = self.boundaries.data.to(device)
        step = (max_value - min_value) / (nb_dichotomies + 1)
        self.boundaries.data = min_value + step * (1 + torch.arange(nb_dichotomies, device=device))
        self.sharpnesses = nn.Parameter((12 / step) * torch.ones(nb_dichotomies))
        self.sharpnesses.data = self.sharpnesses.data.to(device)
        self.to(device)

    def set(
        self,
        device=None,
    ):
        if device != None and device != self.device:
            self.device = device
            self.boundaries.data = self.boundaries.data.to(device)
            self.sharpnesses.data = self.sharpnesses.data.to(device)
            self.to(device)

    def update_parameters(self):
        self.sharpnesses.data.clamp_(min=1e-5)

    def forward(self, x):
        """Forward pass"""
        x_v = x.view(-1, 1)
        boundaries_v = self.boundaries.view(1, self.nb_dichotomies)
        sharpnesses_v = self.sharpnesses.view(1, self.nb_dichotomies)
        result = F.sigmoid(sharpnesses_v * (x_v - boundaries_v))
        return result

    def __repr__(self):
        string = "["
        for out_idx in range(self.nb_dichotomies):
            if out_idx > 0:
                string += ", "
            string += str(
                (
                    round(
                        1 * self.boundaries.data[out_idx].detach().cpu().numpy().tolist(),
                        3,
                    ),
                    round(
                        1 * self.sharpnesses.data[out_idx].detach().cpu().numpy().tolist(),
                        3,
                    ),
                )
            )
        string += "]"
        return string

    def load_string(self, init_string):
        boundary_sharpness_pairs_strings = init_string[1:-1].split("), ")
        for i, boundary_sharpness_pairs_string in enumerate(boundary_sharpness_pairs_strings):
            if boundary_sharpness_pairs_string[-1] != ")":
                boundary_sharpness_pairs_string += ")"
            boundary, sharpness = eval(boundary_sharpness_pairs_string)
            self.boundaries.data[i] = boundary
            self.sharpnesses.data[i] = sharpness
        self.nb_dichotomies = i + 1
        self.boundaries.data = self.boundaries.data[: self.nb_dichotomies]
        self.sharpnesses.data = self.sharpnesses.data[: self.nb_dichotomies]

    def backward_simplify(self, used_out_concepts):
        self.nb_dichotomies = len(used_out_concepts)
        self.boundaries.data = self.boundaries.data[used_out_concepts]
        self.sharpnesses.data = self.sharpnesses.data[used_out_concepts]

    def reorder(self):
        boundary_out_idx_pairs = [(self.boundaries.data[out_idx].item(), out_idx) for out_idx in range(self.nb_dichotomies)]
        boundary_out_idx_pairs.sort(key=lambda pair: pair[0])
        reordered_out_idcs = [out_idx for boundary, out_idx in boundary_out_idx_pairs]
        self.boundaries.data = self.boundaries.data[reordered_out_idcs]
        self.sharpnesses.data = self.sharpnesses.data[reordered_out_idcs]
        return reordered_out_idcs


class PeriodicDichotomies(nn.Module):
    """Fuzzy dichotomies: present to some degree when associated periodic input feature is in the half-period centered around the center"""

    def __init__(
        self,
        nb_dichotomies: int,
        period: float,
        device=DEVICE,
    ):
        super().__init__()
        self.device = device
        self.nb_dichotomies = nb_dichotomies
        self.period = period
        self.centers = nn.Parameter(torch.zeros(nb_dichotomies))
        self.centers.data = self.centers.data.to(device)
        step = period / nb_dichotomies
        self.centers.data = step * torch.arange(nb_dichotomies, device=device)
        self.sharpnesses = nn.Parameter((12 / step) * torch.ones(nb_dichotomies))
        self.sharpnesses.data = self.sharpnesses.data.to(device)
        self.to(device)

    def set(
        self,
        device=None,
    ):
        if device != None and device != self.device:
            self.device = device
            self.centers.data = self.centers.data.to(device)
            self.sharpnesses.data = self.sharpnesses.data.to(device)
            self.to(device)

    def update_parameters(self):
        self.centers.data[self.centers.data < 0] += self.period
        self.centers.data[self.centers.data > self.period] -= self.period
        self.sharpnesses.data.clamp_(min=1e-5)

    def forward(self, x):
        """Forward pass"""
        x_v = x.view(-1, 1)
        centers_v = self.centers.view(1, self.nb_dichotomies)
        sharpnesses_v = self.sharpnesses.view(1, self.nb_dichotomies)
        ls_v = centers_v - 0.25 * self.period
        ls_v[ls_v < 0] += self.period
        us_v = centers_v + 0.25 * self.period
        us_v[us_v > self.period] -= self.period
        l_sigmoids_v = F.sigmoid(sharpnesses_v * (x_v - ls_v))
        u_sigmoids_v = F.sigmoid(sharpnesses_v * (us_v - x_v))
        result = l_sigmoids_v * u_sigmoids_v
        inverted_mask = us_v < ls_v
        result[inverted_mask] = 1 - result[inverted_mask]
        return result

    def __repr__(self):
        string = "["
        for out_idx in range(self.nb_out_concepts_per_in):
            if out_idx > 0:
                string += ", "
            string += str(
                (
                    round(
                        1 * self.centers.data[out_idx].detach().cpu().numpy().tolist()[0],
                        3,
                    ),
                    round(
                        1 * self.sharpnesses.data[out_idx].detach().cpu().numpy().tolist()[0],
                        3,
                    ),
                )
            )
        string += "]"
        return string

    def load_string(self, init_string):
        center_sharpness_pairs_strings = init_string[1:-1].split("), ")
        for i, center_sharpness_pairs_string in enumerate(center_sharpness_pairs_strings):
            center, sharpness = eval(center_sharpness_pairs_string + ")")
            self.centers.data[i] = center
            self.sharpnesses.data[i] = sharpness
        self.nb_dichotomies = i + 1
        self.centers.data = self.centers.data[: self.nb_dichotomies]
        self.sharpnesses.data = self.sharpnesses.data[: self.nb_dichotomies]

    def backward_simplify(self, used_out_concepts):
        self.nb_dichotomies = len(used_out_concepts)
        self.centers.data = self.centers.data[used_out_concepts]
        self.sharpnesses.data = self.sharpnesses.data[used_out_concepts]

    def reorder(self):
        center_out_idx_pairs = [(self.centers.data[out_idx].item(), out_idx) for out_idx in range(self.nb_dichotomies)]
        center_out_idx_pairs.sort(key=lambda pair: pair[0])
        reordered_out_idcs = [out_idx for center, out_idx in center_out_idx_pairs]
        self.centers.data = self.centers.data[reordered_out_idcs]
        self.sharpnesses.data = self.sharpnesses.data[reordered_out_idcs]
        return reordered_out_idcs


class ContinuousPreProcessingModule(nn.Module):
    """
    Continuous pre-processing module:
        (1) fuzzy dichotomies (normal or periodic)
        (2) fuzzy intersections (layer of AND nodes with negation and with no unobserved necessary concepts)
        (3) collections of fuzzy intersections (layer of OR nodes with no negation and no unobserved sufficient concepts)
    """

    def __init__(
        self,
        nb_dichotomies: int,
        nb_intervals: Union[int, None],
        nb_out_concepts: Union[int, None],
        min_value: float = 0,
        max_value: float = 1,
        period: float = 0,
        has_missing_values: bool = False,
        random_init_obs: bool = RANDOM_INIT_OBS,
        empty_reset_in_concepts: bool = EMPTY_RESET_IN_CONCEPTS,
        train_forw_weight_quant: str = TRAIN_FORW_WEIGHT_QUANT,
        approx_AND_OR_params: Union[None, Tuple[float, float, float]] = APPROX_PARAMS,
        device=DEVICE,
        verbose: bool = VERBOSE,
    ):
        super().__init__()
        self.nb_dichotomies = nb_dichotomies
        if nb_intervals == None:
            nb_intervals = nb_dichotomies + 1
        if nb_out_concepts == None:
            nb_out_concepts = nb_intervals
        self.nb_intervals = nb_intervals
        self.nb_out_concepts = nb_out_concepts
        if period == 0:
            self.is_periodic = False
        else:
            self.is_periodic = True
            self.period = period
        self.has_missing_values = has_missing_values
        self.device = device
        if not self.is_periodic:
            self.dichotomies = Dichotomies(nb_dichotomies, min_value, max_value, device=device)
        else:
            self.dichotomies = PeriodicDichotomies(nb_dichotomies, period, device=device)
        self.intervals = AndConcepts(
            nb_dichotomies,
            nb_intervals,
            use_negation=True,
            use_unobserved=False,
            random_init_obs=random_init_obs,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            device=device,
            verbose=verbose,
        )
        if nb_intervals >= nb_dichotomies + 1:
            minus_eye = torch.cat((-1 * torch.eye(nb_dichotomies, device=device), torch.zeros((1, nb_dichotomies), device=device)), dim=0)
            plus_eye = torch.cat((torch.zeros((1, nb_dichotomies), device=device), torch.eye(nb_dichotomies, device=device)), dim=0)
            self.intervals.observed_concepts.data[: nb_dichotomies + 1, :] = minus_eye + plus_eye
        self.out_concepts = OrConcepts(
            nb_intervals,
            nb_out_concepts,
            use_negation=False,
            use_unobserved=False,
            use_missing_values=has_missing_values,
            random_init_obs=random_init_obs,
            empty_reset_in_concepts=empty_reset_in_concepts,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            device=device,
            verbose=verbose,
        )
        self.to(device)

    def set(
        self,
        empty_reset_in_concepts=None,
        train_forw_weight_quant=None,
        approx_AND_OR_params="",
        device=None,
        verbose=None,
    ):
        self.dichotomies.set(device=device)
        self.intervals.set(
            empty_reset_in_concepts=empty_reset_in_concepts,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            device=device,
            verbose=verbose,
        )
        self.out_concepts.set(
            empty_reset_in_concepts=empty_reset_in_concepts,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            device=device,
            verbose=verbose,
        )

    def reset_out_concept(self, out_concept):
        self.out_concepts.reset_out_concept(out_concept)

    def update_parameters(self):
        self.dichotomies.update_parameters()
        self.intervals.update_parameters()
        self.out_concepts.update_parameters()

    def override(self, new_override):
        self.intervals.override(new_override)
        self.out_concepts.override(new_override)

    def review_unused_concepts(self, used_out_concepts):
        if RESET_UNUSED_CONCEPTS:
            unused_in_concepts = self.out_concepts.new_review_unused_concepts(used_out_concepts, [])
            self.intervals.new_review_unused_concepts(unused_in_concepts, [], do_check_in_concepts=False)

    def forward(self, x):
        """Forward pass"""
        if self.has_missing_values:
            missing_idcs = torch.nonzero(torch.isnan(x)).view(-1).tolist()
            self.out_concepts.missing_idcs = missing_idcs
            x = x.clone()
            x[missing_idcs] = 0
        return self.out_concepts(self.intervals(self.dichotomies(x)))

    def add_regularization(self, loss):
        loss = self.intervals.add_regularization(loss)
        loss = self.out_concepts.add_regularization(loss)
        return loss

    def get_dichotomies_observed_unobserved_parameters(self):
        intervals_observed_concepts, intervals_unobserved_concepts = self.intervals.get_observed_unobserved_parameters()
        out_concepts_observed_concepts, out_concepts_unobserved_concepts = self.out_concepts.get_observed_unobserved_parameters()

        dichotomies_parameters = list(self.dichotomies.parameters())
        observed_concepts = intervals_observed_concepts + out_concepts_observed_concepts
        unobserved_concepts = intervals_unobserved_concepts + out_concepts_unobserved_concepts
        return dichotomies_parameters, observed_concepts, unobserved_concepts

    def __repr__(self):
        string = "[" + str(self.dichotomies) + ", \n" + str(self.intervals) + ", \n" + str(self.out_concepts) + "]"
        return string

    def load_string(self, init_string):
        lines = init_string.split("\n")
        dichotomies_init_string = lines[0][1:-2]
        intervals_init_string = lines[1][:-1]
        out_concepts_init_string = lines[2][:-1]
        self.dichotomies.load_string(dichotomies_init_string)
        self.intervals.load_string(intervals_init_string)
        self.out_concepts.load_string(out_concepts_init_string)
        self.nb_dichotomies = self.dichotomies.nb_dichotomies
        self.nb_intervals = self.intervals.nb_out_concepts
        self.nb_out_concepts = self.out_concepts.nb_out_concepts

    def backward_simplify(self, used_out_concepts):
        self.nb_out_concepts = len(used_out_concepts)
        used_out_concepts = self.out_concepts.backward_simplify(used_out_concepts)
        self.nb_intervals = len(used_out_concepts)
        used_out_concepts = self.intervals.backward_simplify(used_out_concepts)
        self.nb_dichotomies = len(used_out_concepts)
        self.dichotomies.backward_simplify(used_out_concepts)

    def forward_simplify(self):
        used_in_concepts, unused_in_idx_prob_pairs = self.intervals.forward_simplify()
        self.nb_intervals = len(used_in_concepts)
        used_in_concepts, unused_in_idx_prob_pairs = self.out_concepts.forward_simplify(used_in_concepts, unused_in_idx_prob_pairs)
        self.nb_out_concepts = len(used_in_concepts)
        return used_in_concepts, unused_in_idx_prob_pairs

    def remove_duplicates(self):
        duplicate_lists = self.intervals.remove_duplicates()
        self.nb_intervals = self.intervals.nb_out_concepts
        must_retrain = len(duplicate_lists) > 0
        duplicate_lists = self.out_concepts.remove_duplicates(in_duplicate_lists=duplicate_lists)
        self.nb_out_concepts = self.out_concepts.nb_out_concepts
        must_retrain = must_retrain or len(duplicate_lists) > 0
        return duplicate_lists, must_retrain

    def prune(self, eval_model_func, init_loss=-1, log_files=[], progress_bar_hook=lambda weight: None):
        did_prune = False
        if init_loss < 0:
            init_loss = eval_model_func()
        current_weight = 0
        old_module_nb_weights = self.out_concepts.get_nb_weights()
        init_loss, local_did_prune = self.out_concepts.prune(
            eval_model_func, init_loss=init_loss, log_files=log_files, progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight)
        )
        did_prune = did_prune or local_did_prune
        current_weight += old_module_nb_weights
        init_loss, local_did_prune = self.intervals.prune(
            eval_model_func, init_loss=init_loss, log_files=log_files, progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight)
        )
        did_prune = did_prune or local_did_prune
        return init_loss, did_prune

    def get_nb_weights(self):
        return self.intervals.get_nb_weights() + self.out_concepts.get_nb_weights()

    def discretize(self, discretization_method, eval_model_func=lambda: None, log_files=[], progress_bar_hook=lambda weight: None):
        if discretization_method == "thresh":
            self.intervals.discretize("thresh")
            self.out_concepts.discretize("thresh")
        else:
            losses = []
            current_weight = 0
            old_module_nb_weights = self.out_concepts.get_nb_weights()
            losses += self.out_concepts.discretize(
                discretization_method,
                eval_model_func=eval_model_func,
                log_files=log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
            )
            current_weight += old_module_nb_weights
            losses += self.intervals.discretize(
                discretization_method,
                eval_model_func=eval_model_func,
                log_files=log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
            )
            return losses

    def reorder_lexicographically(self):
        reordered_dichotomy_idcs = self.dichotomies.reorder()
        self.intervals.observed_concepts.data = self.intervals.observed_concepts.data[:, reordered_dichotomy_idcs]
        reordered_interval_idcs = find_lexicographical_order(self.intervals.observed_concepts.data)
        self.intervals.observed_concepts.data = self.intervals.observed_concepts.data[reordered_interval_idcs, :]
        self.out_concepts.observed_concepts.data = self.out_concepts.observed_concepts.data[:, reordered_interval_idcs]
        reordered_out_idcs = find_lexicographical_order(self.out_concepts.observed_concepts.data)
        self.out_concepts.observed_concepts.data = self.out_concepts.observed_concepts.data[reordered_out_idcs, :]
        if self.has_missing_values:
            self.out_concepts.missing_observed_concepts.data = self.out_concepts.missing_observed_concepts.data[reordered_out_idcs]
        return reordered_out_idcs


class NLNPreProcessingModules(nn.Module):
    """
    NLN pre-processing modules:
        - binary features
            (unprocessed)
        - categorical features
            (1) one-hot encoding (should already be hardcoded in the dataset --> see NLN_DataGeneration.py)
            (2) equivalency classes (layer of OR nodes with no negation and no unobserved sufficient concepts)
        - continuous/periodic features
            (1) fuzzy dichotomies (normal or periodic)
            (2) fuzzy intersections (layer of AND nodes with negation and with no unobserved necessary concepts)
            (3) collections of fuzzy intersections (layer of OR nodes with no negation and no unobserved sufficient concepts)
    """

    def __init__(
        self,
        nb_in_features: int,
        train_forw_weight_quant: str = TRAIN_FORW_WEIGHT_QUANT,
        approx_AND_OR_params: Union[None, Tuple[float, float, float]] = APPROX_PARAMS,
        category_first_last_has_missing_values_tuples: List[Tuple[int, int, bool]] = [],
        continuous_index_min_max_has_missing_values_tuples: List[Tuple[int, float, float, bool]] = [],
        periodic_index_period_has_missing_values_tuples: List[Tuple[int, float, bool]] = [],
        feature_names: List[str] = [],
        category_nb_out_concepts: int = -1,
        nb_dichotomies_per_continuous: int = NB_DICHOTOMIES_PER_CONTINUOUS,
        nb_intervals_per_continuous: Union[int, None] = None,
        nb_out_concepts_per_continuous: int = -1,
        random_init_obs: bool = RANDOM_INIT_OBS,
        empty_reset_in_concepts: bool = EMPTY_RESET_IN_CONCEPTS,
        device=DEVICE,
        verbose: bool = VERBOSE,
    ):
        super().__init__()
        self.nb_in_features = nb_in_features
        binary_indices = set(range(nb_in_features))
        nb_out_concepts = 0

        category_modules = []
        self.category_first_last_has_missing_values_tuples = []
        for category_first_last_has_missing_values_tuple in category_first_last_has_missing_values_tuples:
            first_idx = category_first_last_has_missing_values_tuple[0]
            last_idx = category_first_last_has_missing_values_tuple[1]
            has_missing_values = category_first_last_has_missing_values_tuple[-1]
            binary_indices = binary_indices - set(range(first_idx, last_idx + 1))
            self.category_first_last_has_missing_values_tuples.append((first_idx, last_idx, has_missing_values))
            category_nb_in_concepts = last_idx - first_idx + 1
            nb_out_concepts += category_nb_out_concepts
            category_modules.append(
                OrConcepts(
                    category_nb_in_concepts,
                    category_nb_out_concepts,
                    use_negation=False,
                    use_unobserved=False,
                    use_missing_values=has_missing_values,
                    random_init_obs=random_init_obs,
                    train_forw_weight_quant=train_forw_weight_quant,
                    approx_AND_OR_params=approx_AND_OR_params,
                    device=device,
                    verbose=verbose,
                )
            )
        self.category_modules = nn.ModuleList(category_modules)

        self.continuous_index_min_max_has_missing_values_tuples = [
            (idx, min_value, max_value, has_missing_values) for idx, min_value, max_value, has_missing_values in continuous_index_min_max_has_missing_values_tuples
        ]
        binary_indices = binary_indices - set(
            [continuous_index_min_max_has_missing_values_tuple[0] for continuous_index_min_max_has_missing_values_tuple in continuous_index_min_max_has_missing_values_tuples]
        )
        continuous_modules = []
        for continuous_index, min_value, max_value, has_missing_values in continuous_index_min_max_has_missing_values_tuples:
            continuous_modules.append(
                ContinuousPreProcessingModule(
                    nb_dichotomies_per_continuous,
                    nb_intervals_per_continuous,
                    nb_out_concepts_per_continuous,
                    min_value=min_value,
                    max_value=max_value,
                    has_missing_values=has_missing_values,
                    random_init_obs=random_init_obs,
                    empty_reset_in_concepts=empty_reset_in_concepts,
                    train_forw_weight_quant=train_forw_weight_quant,
                    approx_AND_OR_params=approx_AND_OR_params,
                    device=device,
                    verbose=verbose,
                )
            )
            nb_out_concepts += continuous_modules[-1].nb_out_concepts
        self.continuous_modules = nn.ModuleList(continuous_modules)

        self.periodic_index_period_has_missing_values_tuples = [
            (idx, period, has_missing_values) for idx, period, has_missing_values in periodic_index_period_has_missing_values_tuples
        ]
        binary_indices = binary_indices - set(
            [periodic_index_period_has_missing_values_tuple[0] for periodic_index_period_has_missing_values_tuple in periodic_index_period_has_missing_values_tuples]
        )
        periodic_modules = []
        for periodic_index, period, has_missing_values in self.periodic_index_period_has_missing_values_tuples:
            periodic_modules.append(
                ContinuousPreProcessingModule(
                    nb_dichotomies_per_continuous,
                    nb_intervals_per_continuous,
                    nb_out_concepts_per_continuous,
                    period=period,
                    has_missing_values=has_missing_values,
                    random_init_obs=random_init_obs,
                    empty_reset_in_concepts=empty_reset_in_concepts,
                    train_forw_weight_quant=train_forw_weight_quant,
                    approx_AND_OR_params=approx_AND_OR_params,
                    device=device,
                    verbose=verbose,
                )
            )
            nb_out_concepts += periodic_modules[-1].nb_out_concepts
        self.periodic_modules = nn.ModuleList(periodic_modules)

        self.binary_indices = sorted(list(binary_indices))
        nb_out_concepts += len(binary_indices)
        self.nb_out_concepts = nb_out_concepts
        self.feature_names = feature_names
        self.device = device
        self.to(device)

    def set(
        self,
        empty_reset_in_concepts=None,
        train_forw_weight_quant=None,
        approx_AND_OR_params="",
        device=None,
        verbose=None,
    ):
        for category_module in self.category_modules:
            category_module.set(
                empty_reset_in_concepts=empty_reset_in_concepts,
                train_forw_weight_quant=train_forw_weight_quant,
                approx_AND_OR_params=approx_AND_OR_params,
                device=device,
                verbose=verbose,
            )
        for continuous_module in self.continuous_modules:
            continuous_module.set(
                empty_reset_in_concepts=empty_reset_in_concepts,
                train_forw_weight_quant=train_forw_weight_quant,
                approx_AND_OR_params=approx_AND_OR_params,
                device=device,
                verbose=verbose,
            )
        for periodic_module in self.periodic_modules:
            periodic_module.set(
                empty_reset_in_concepts=empty_reset_in_concepts,
                train_forw_weight_quant=train_forw_weight_quant,
                approx_AND_OR_params=approx_AND_OR_params,
                device=device,
                verbose=verbose,
            )
        if device != None and device != self.device:
            self.device = device
            self.to(device)

    @staticmethod
    def merge_modules(
        modules,
        nb_in_concepts: int,
        train_forw_weight_quant: str = TRAIN_FORW_WEIGHT_QUANT,
        approx_AND_OR_params: Union[None, Tuple[float, float, float]] = APPROX_PARAMS,
        category_first_last_has_missing_values_tuples: List[Tuple[int, int, bool]] = [],
        continuous_index_min_max_has_missing_values_tuples: List[Tuple[int, float, float, bool]] = [],
        periodic_index_period_has_missing_values_tuples: List[Tuple[int, float, bool]] = [],
        feature_names: List[str] = [],
        category_nb_out_concepts: int = -1,
        nb_dichotomies_per_continuous: int = NB_DICHOTOMIES_PER_CONTINUOUS,
        nb_intervals_per_continuous: Union[int, None] = None,
        nb_out_concepts_per_continuous: int = -1,
        random_init_obs: bool = RANDOM_INIT_OBS,
        empty_reset_in_concepts: bool = EMPTY_RESET_IN_CONCEPTS,
        device=DEVICE,
        verbose: bool = VERBOSE,
    ):
        if len(set([module.nb_in_concepts for module in modules])) > 1:
            raise Exception("The merged modules must have the same number of input concepts.")

        merged_module = NLNPreProcessingModules(
            nb_in_concepts,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            category_first_last_has_missing_values_tuples=category_first_last_has_missing_values_tuples,
            continuous_index_min_max_has_missing_values_tuples=continuous_index_min_max_has_missing_values_tuples,
            periodic_index_period_has_missing_values_tuples=periodic_index_period_has_missing_values_tuples,
            feature_names=feature_names,
            category_nb_out_concepts=category_nb_out_concepts,
            nb_dichotomies_per_continuous=nb_dichotomies_per_continuous,
            nb_intervals_per_continuous=nb_intervals_per_continuous,
            nb_out_concepts_per_continuous=nb_out_concepts_per_continuous,
            random_init_obs=random_init_obs,
            empty_reset_in_concepts=empty_reset_in_concepts,
            device=device,
            verbose=verbose,
        )

        idx_translation_by_module = [[] for module in modules]

        merged_binary_indices = set()
        for module in modules:
            merged_binary_indices |= set(module.binary_indices)
        merged_module.binary_indices = sorted(list(merged_binary_indices))
        for module_idx, module in enumerate(modules):
            for binary_idx in module.binary_indices:
                idx_translation_by_module[module_idx].append(merged_module.binary_indices.index(binary_idx))
        nb_out_concepts = len(merged_binary_indices)

        merged_category_first_last_has_missing_values_tuples = set()
        for module in modules:
            merged_category_first_last_has_missing_values_tuples |= set(module.category_first_last_has_missing_values_tuples)
        merged_module.category_first_last_has_missing_values_tuples = sorted(list(merged_category_first_last_has_missing_values_tuples), key=lambda tup: tup[0])
        categories_to_remove = []
        for i, category_first_last_has_missing_values_tuple in enumerate(category_first_last_has_missing_values_tuples):
            if category_first_last_has_missing_values_tuple not in merged_module.category_first_last_has_missing_values_tuples:
                categories_to_remove.append(i)
        for i in reversed(categories_to_remove):
            del merged_module.category_modules[i]
        for merged_cat_idx, category_first_last_has_missing_values_tuple in enumerate(merged_module.category_first_last_has_missing_values_tuples):
            merged_category_module = merged_module.category_modules[merged_cat_idx]
            is_first = True
            for module_idx, module in enumerate(modules):
                for module_cat_idx, module_category_first_last_has_missing_values_tuple in enumerate(module.category_first_last_has_missing_values_tuples):
                    if module_category_first_last_has_missing_values_tuple == category_first_last_has_missing_values_tuple:
                        module_category_module = module.category_modules[module_cat_idx]
                        if is_first:
                            merged_category_module.observed_concepts.data = 1 * module_category_module.observed_concepts.data
                            if merged_category_module.use_missing_values:
                                merged_category_module.missing_observed_concepts.data = 1 * module_category_module.missing_observed_concepts.data
                            is_first = False
                        else:
                            merged_category_module.observed_concepts.data = torch.concat(
                                (merged_category_module.observed_concepts.data, 1 * module_category_module.observed_concepts.data), dim=0
                            )
                            if merged_category_module.use_missing_values:
                                merged_category_module.missing_observed_concepts.data = torch.concat(
                                    (merged_category_module.missing_observed_concepts.data, 1 * module_category_module.missing_observed_concepts.data), dim=0
                                )
                        merged_category_module.nb_out_concepts = merged_category_module.observed_concepts.data.shape[0]
                        idx_translation_by_module[module_idx] += [
                            nb_out_concepts + module_category_out_idx for module_category_out_idx in range(module_category_module.nb_out_concepts)
                        ]
                        nb_out_concepts += module_category_module.nb_out_concepts
                        break

        merged_continuous_index_min_max_has_missing_values_tuples = set()
        for module in modules:
            merged_continuous_index_min_max_has_missing_values_tuples |= set(module.continuous_index_min_max_has_missing_values_tuples)
        merged_module.continuous_index_min_max_has_missing_values_tuples = sorted(list(merged_continuous_index_min_max_has_missing_values_tuples), key=lambda tup: tup[0])
        continuous_to_remove = []
        for i, continuous_index_min_max_has_missing_values_tuple in enumerate(continuous_index_min_max_has_missing_values_tuples):
            if continuous_index_min_max_has_missing_values_tuple not in merged_module.continuous_index_min_max_has_missing_values_tuples:
                continuous_to_remove.append(i)
        for i in reversed(continuous_to_remove):
            del merged_module.continuous_modules[i]
        for merged_con_idx, continuous_index_min_max_has_missing_values_tuple in enumerate(merged_module.continuous_index_min_max_has_missing_values_tuples):
            merged_continuous_module = merged_module.continuous_modules[merged_con_idx]
            is_first = True
            for module_idx, module in enumerate(modules):
                for module_con_idx, module_continuous_index_min_max_has_missing_values_tuple in enumerate(module.continuous_index_min_max_has_missing_values_tuples):
                    if module_continuous_index_min_max_has_missing_values_tuple == continuous_index_min_max_has_missing_values_tuple:
                        module_continuous_module = module.continuous_modules[module_con_idx]
                        if is_first:
                            merged_continuous_module.dichotomies.boundaries.data = 1 * module_continuous_module.dichotomies.boundaries.data
                            merged_continuous_module.dichotomies.sharpnesses.data = 1 * module_continuous_module.dichotomies.sharpnesses.data
                            merged_continuous_module.intervals.observed_concepts.data = 1 * module_continuous_module.intervals.observed_concepts.data
                            merged_continuous_module.out_concepts.observed_concepts.data = 1 * module_continuous_module.out_concepts.observed_concepts.data
                            if merged_continuous_module.out_concepts.use_missing_values:
                                merged_continuous_module.out_concepts.missing_observed_concepts.data = 1 * module_continuous_module.out_concepts.missing_observed_concepts.data
                            is_first = False
                        else:
                            merged_continuous_module.dichotomies.boundaries.data = torch.concat(
                                (merged_continuous_module.dichotomies.boundaries.data, 1 * module_continuous_module.dichotomies.boundaries.data), dim=0
                            )
                            merged_continuous_module.dichotomies.sharpnesses.data = torch.concat(
                                (merged_continuous_module.dichotomies.sharpnesses.data, 1 * module_continuous_module.dichotomies.sharpnesses.data), dim=0
                            )
                            merged_continuous_module.intervals.observed_concepts.data = torch.concat(
                                (
                                    torch.concat(
                                        (
                                            merged_continuous_module.intervals.observed_concepts.data,
                                            torch.zeros((merged_continuous_module.intervals.nb_out_concepts, module_continuous_module.intervals.nb_in_concepts), device=device),
                                        ),
                                        dim=1,
                                    ),
                                    torch.concat(
                                        (
                                            torch.zeros((module_continuous_module.intervals.nb_out_concepts, merged_continuous_module.intervals.nb_in_concepts), device=device),
                                            1 * module_continuous_module.intervals.observed_concepts.data,
                                        ),
                                        dim=1,
                                    ),
                                ),
                                dim=0,
                            )
                            merged_continuous_module.out_concepts.observed_concepts.data = torch.concat(
                                (
                                    torch.concat(
                                        (
                                            merged_continuous_module.out_concepts.observed_concepts.data,
                                            torch.zeros(
                                                (merged_continuous_module.out_concepts.nb_out_concepts, module_continuous_module.out_concepts.nb_in_concepts), device=device
                                            ),
                                        ),
                                        dim=1,
                                    ),
                                    torch.concat(
                                        (
                                            torch.zeros(
                                                (module_continuous_module.out_concepts.nb_out_concepts, merged_continuous_module.out_concepts.nb_in_concepts), device=device
                                            ),
                                            1 * module_continuous_module.out_concepts.observed_concepts.data,
                                        ),
                                        dim=1,
                                    ),
                                ),
                                dim=0,
                            )
                            if merged_continuous_module.out_concepts.use_missing_values:
                                merged_continuous_module.out_concepts.missing_observed_concepts.data = torch.concat(
                                    (
                                        merged_continuous_module.out_concepts.missing_observed_concepts.data,
                                        1 * module_continuous_module.out_concepts.missing_observed_concepts.data,
                                    ),
                                    dim=0,
                                )
                        merged_continuous_module.dichotomies.nb_dichotomies = merged_continuous_module.dichotomies.boundaries.data.shape[0]
                        merged_continuous_module.nb_dichotomies = merged_continuous_module.dichotomies.nb_dichotomies
                        merged_continuous_module.intervals.nb_in_concepts = merged_continuous_module.intervals.observed_concepts.data.shape[1]
                        merged_continuous_module.intervals.nb_out_concepts = merged_continuous_module.intervals.observed_concepts.data.shape[0]
                        merged_continuous_module.nb_intervals = merged_continuous_module.intervals.nb_out_concepts
                        merged_continuous_module.out_concepts.nb_in_concepts = merged_continuous_module.out_concepts.observed_concepts.data.shape[1]
                        merged_continuous_module.out_concepts.nb_out_concepts = merged_continuous_module.out_concepts.observed_concepts.data.shape[0]
                        merged_continuous_module.nb_out_concepts = merged_continuous_module.out_concepts.nb_out_concepts
                        idx_translation_by_module[module_idx] += [
                            nb_out_concepts + module_continuous_out_idx for module_continuous_out_idx in range(module_continuous_module.nb_out_concepts)
                        ]
                        nb_out_concepts += module_continuous_module.nb_out_concepts
                        break

        merged_periodic_index_period_has_missing_values_tuples = set()
        for module in modules:
            merged_periodic_index_period_has_missing_values_tuples |= set(module.periodic_index_period_has_missing_values_tuples)
        merged_module.periodic_index_period_has_missing_values_tuples = sorted(list(merged_periodic_index_period_has_missing_values_tuples), key=lambda tup: tup[0])
        periodic_to_remove = []
        for i, periodic_index_period_has_missing_values_tuple in enumerate(periodic_index_period_has_missing_values_tuples):
            if periodic_index_period_has_missing_values_tuple not in merged_module.periodic_index_period_has_missing_values_tuples:
                periodic_to_remove.append(i)
        for i in reversed(periodic_to_remove):
            del merged_module.periodic_modules[i]
        for merged_per_idx, periodic_index_period_has_missing_values_tuple in enumerate(merged_module.periodic_index_period_has_missing_values_tuples):
            merged_periodic_module = merged_module.periodic_modules[merged_per_idx]
            is_first = True
            for module_idx, module in enumerate(modules):
                for module_per_idx, module_periodic_index_period_has_missing_values_tuple in enumerate(module.periodic_index_period_has_missing_values_tuples):
                    if module_periodic_index_period_has_missing_values_tuple == periodic_index_period_has_missing_values_tuple:
                        module_periodic_module = module.periodic_modules[module_per_idx]
                        if is_first:
                            merged_periodic_module.dichotomies.centers.data = 1 * module_periodic_module.dichotomies.centers.data
                            merged_periodic_module.dichotomies.sharpnesses.data = 1 * module_periodic_module.dichotomies.sharpnesses.data
                            merged_periodic_module.intervals.observed_concepts.data = 1 * module_periodic_module.intervals.observed_concepts.data
                            merged_periodic_module.out_concepts.observed_concepts.data = 1 * module_periodic_module.out_concepts.observed_concepts.data
                            if merged_periodic_module.out_concepts.use_missing_values:
                                merged_periodic_module.out_concepts.missing_observed_concepts.data = 1 * module_periodic_module.out_concepts.missing_observed_concepts.data
                            is_first = False
                        else:
                            merged_periodic_module.dichotomies.centers.data = torch.concat(
                                (merged_periodic_module.dichotomies.centers.data, 1 * module_periodic_module.dichotomies.centers.data), dim=0
                            )
                            merged_periodic_module.dichotomies.sharpnesses.data = torch.concat(
                                (merged_periodic_module.dichotomies.sharpnesses.data, 1 * module_periodic_module.dichotomies.sharpnesses.data), dim=0
                            )
                            merged_periodic_module.intervals.observed_concepts.data = torch.concat(
                                (
                                    torch.concat(
                                        (
                                            merged_periodic_module.intervals.observed_concepts.data,
                                            torch.zeros((merged_periodic_module.intervals.nb_out_concepts, module_periodic_module.intervals.nb_in_concepts), device=device),
                                        ),
                                        dim=1,
                                    ),
                                    torch.concat(
                                        (
                                            torch.zeros((module_periodic_module.intervals.nb_out_concepts, merged_periodic_module.intervals.nb_in_concepts), device=device),
                                            1 * module_periodic_module.intervals.observed_concepts.data,
                                        ),
                                        dim=1,
                                    ),
                                ),
                                dim=0,
                            )
                            merged_periodic_module.out_concepts.observed_concepts.data = torch.concat(
                                (
                                    torch.concat(
                                        (
                                            merged_periodic_module.out_concepts.observed_concepts.data,
                                            torch.zeros((merged_periodic_module.out_concepts.nb_out_concepts, module_periodic_module.out_concepts.nb_in_concepts), device=device),
                                        ),
                                        dim=1,
                                    ),
                                    torch.concat(
                                        (
                                            torch.zeros((module_periodic_module.out_concepts.nb_out_concepts, merged_periodic_module.out_concepts.nb_in_concepts), device=device),
                                            1 * module_periodic_module.out_concepts.observed_concepts.data,
                                        ),
                                        dim=1,
                                    ),
                                ),
                                dim=0,
                            )
                            if merged_periodic_module.out_concepts.use_missing_values:
                                merged_periodic_module.out_concepts.missing_observed_concepts.data = torch.concat(
                                    (merged_periodic_module.out_concepts.missing_observed_concepts.data, 1 * module_periodic_module.out_concepts.missing_observed_concepts.data),
                                    dim=0,
                                )
                        merged_periodic_module.dichotomies.nb_dichotomies = merged_periodic_module.dichotomies.centers.data.shape[0]
                        merged_periodic_module.nb_dichotomies = merged_periodic_module.dichotomies.nb_dichotomies
                        merged_periodic_module.intervals.nb_in_concepts = merged_periodic_module.intervals.observed_concepts.data.shape[1]
                        merged_periodic_module.intervals.nb_out_concepts = merged_periodic_module.intervals.observed_concepts.data.shape[0]
                        merged_periodic_module.nb_intervals = merged_periodic_module.intervals.nb_out_concepts
                        merged_periodic_module.out_concepts.nb_in_concepts = merged_periodic_module.out_concepts.observed_concepts.data.shape[1]
                        merged_periodic_module.out_concepts.nb_out_concepts = merged_periodic_module.out_concepts.observed_concepts.data.shape[0]
                        merged_periodic_module.nb_out_concepts = merged_periodic_module.out_concepts.nb_out_concepts
                        idx_translation_by_module[module_idx] += [
                            nb_out_concepts + module_periodic_out_idx for module_periodic_out_idx in range(module_periodic_module.nb_out_concepts)
                        ]
                        nb_out_concepts += module_periodic_module.nb_out_concepts
                        break

        merged_module.nb_out_concepts = nb_out_concepts

        return merged_module, idx_translation_by_module

    def reset_out_concepts(self, out_concepts):
        first_out_concept_idx = len(self.binary_indices)

        for category_module in self.category_modules:
            next_first_out_concept_idx = first_out_concept_idx + category_module.nb_out_concepts
            module_out_concepts = [
                out_concept - first_out_concept_idx for out_concept in out_concepts if out_concept >= first_out_concept_idx and out_concept < next_first_out_concept_idx
            ]
            for module_out_concept in module_out_concepts:
                category_module.reset_out_concept(module_out_concept)
            first_out_concept_idx = next_first_out_concept_idx

        for continuous_module in self.continuous_modules:
            next_first_out_concept_idx = first_out_concept_idx + continuous_module.nb_out_concepts
            module_out_concepts = [
                out_concept - first_out_concept_idx for out_concept in out_concepts if out_concept >= first_out_concept_idx and out_concept < next_first_out_concept_idx
            ]
            for module_out_concept in module_out_concepts:
                continuous_module.reset_out_concept(module_out_concept)
            first_out_concept_idx = next_first_out_concept_idx

        for periodic_module in self.periodic_modules:
            next_first_out_concept_idx = first_out_concept_idx + periodic_module.nb_out_concepts
            module_out_concepts = [
                out_concept - first_out_concept_idx for out_concept in out_concepts if out_concept >= first_out_concept_idx and out_concept < next_first_out_concept_idx
            ]
            for module_out_concept in module_out_concepts:
                periodic_module.reset_out_concept(module_out_concept)
            first_out_concept_idx = next_first_out_concept_idx

    def update_parameters(self):
        for category_module in self.category_modules:
            category_module.update_parameters()
        for continuous_module in self.continuous_modules:
            continuous_module.update_parameters()
        for periodic_module in self.periodic_modules:
            periodic_module.update_parameters()

    def override(self, new_override):
        for category_module in self.category_modules:
            category_module.override(new_override)
        for continuous_module in self.continuous_modules:
            continuous_module.override(new_override)
        for periodic_module in self.periodic_modules:
            periodic_module.override(new_override)

    def review_unused_concepts(self, used_out_concepts):
        if RESET_UNUSED_CONCEPTS:
            first_out_concept_idx = len(self.binary_indices)

            for category_module in self.category_modules:
                module_used_out_concepts = [
                    used_out_concept - first_out_concept_idx
                    for used_out_concept in used_out_concepts
                    if used_out_concept >= first_out_concept_idx and used_out_concept < first_out_concept_idx + category_module.nb_out_concepts
                ]
                category_module.review_unused_concepts(module_used_out_concepts, [], do_check_in_concepts=False)
                first_out_concept_idx += category_module.nb_out_concepts

            for continuous_module in self.continuous_modules:
                module_used_out_concepts = [
                    used_out_concept - first_out_concept_idx
                    for used_out_concept in used_out_concepts
                    if used_out_concept >= first_out_concept_idx and used_out_concept < first_out_concept_idx + continuous_module.nb_out_concepts
                ]
                continuous_module.review_unused_concepts(module_used_out_concepts)
                first_out_concept_idx += continuous_module.nb_out_concepts

            for periodic_module in self.periodic_modules:
                module_used_out_concepts = [
                    used_out_concept - first_out_concept_idx
                    for used_out_concept in used_out_concepts
                    if used_out_concept >= first_out_concept_idx and used_out_concept < first_out_concept_idx + periodic_module.nb_out_concepts
                ]
                periodic_module.review_unused_concepts(module_used_out_concepts)
                first_out_concept_idx += periodic_module.nb_out_concepts

    def forward(self, x):
        """Forward pass"""
        results = []

        if len(self.binary_indices) > 0:
            results.append(x[:, self.binary_indices])

        for i, category_module in enumerate(self.category_modules):
            first_idx, last_idx, has_missing_values = self.category_first_last_has_missing_values_tuples[i]
            if has_missing_values:
                category_module.missing_idcs = torch.nonzero((x[:, first_idx : last_idx + 1] == 0).all(dim=-1)).view(-1).tolist()
            results.append(category_module(x[:, first_idx : last_idx + 1]))

        for i, continuous_module in enumerate(self.continuous_modules):
            idx, min_value, max_value, has_missing_values = self.continuous_index_min_max_has_missing_values_tuples[i]
            results.append(continuous_module(x[:, idx]))

        for i, periodic_module in enumerate(self.periodic_modules):
            idx, period, has_missing_values = self.periodic_index_period_has_missing_values_tuples[i]
            results.append(periodic_module(x[:, idx]))

        if len(results) == 1:
            return results[0]
        elif len(results) > 0:
            return torch.cat(results, dim=-1)
        else:
            return torch.zeros((x.size(0)), device=self.device)

    def add_regularization(self, loss):
        for category_module in self.category_modules:
            loss = category_module.add_regularization(loss)
        for continuous_module in self.continuous_modules:
            loss = continuous_module.add_regularization(loss)
        for periodic_module in self.periodic_modules:
            loss = periodic_module.add_regularization(loss)
        return loss

    def get_out_concepts_group_first_stop_pairs(self):
        out_concepts_group_first_stop_pairs = []

        first_out_concept_idx = len(self.binary_indices)

        for category_module in self.category_modules:
            next_first_out_concept_idx = first_out_concept_idx + category_module.nb_out_concepts
            out_concepts_group_first_stop_pairs.append((first_out_concept_idx, next_first_out_concept_idx))
            first_out_concept_idx = next_first_out_concept_idx

        for continuous_module in self.continuous_modules:
            next_first_out_concept_idx = first_out_concept_idx + continuous_module.nb_out_concepts
            out_concepts_group_first_stop_pairs.append((first_out_concept_idx, next_first_out_concept_idx))
            first_out_concept_idx = next_first_out_concept_idx

        for periodic_module in self.periodic_modules:
            next_first_out_concept_idx = first_out_concept_idx + periodic_module.nb_out_concepts
            out_concepts_group_first_stop_pairs.append((first_out_concept_idx, next_first_out_concept_idx))
            first_out_concept_idx = next_first_out_concept_idx

        return out_concepts_group_first_stop_pairs

    def get_category_values_names(self, category_idx):
        first_idx, last_idx, has_missing_values = self.category_first_last_has_missing_values_tuples[category_idx]
        common_prefix_words = self.feature_names[first_idx].split("_")
        for idx in range(first_idx + 1, last_idx + 1):
            current_idx_words = self.feature_names[idx].split("_")
            for common_prefix_word_idx in range(len(common_prefix_words)):
                if common_prefix_words[common_prefix_word_idx] != current_idx_words[common_prefix_word_idx]:
                    common_prefix_words = common_prefix_words[:common_prefix_word_idx]
                    break
        category_name = "_".join(common_prefix_words)
        values_names = [self.feature_names[idx][len(category_name) + 1 :] for idx in range(first_idx, last_idx + 1)]
        return category_name, values_names

    def get_dichotomies_observed_unobserved_parameters(self):
        dichotomies_parameters = []
        observed_concepts = []
        unobserved_concepts = []
        for category_module in self.category_modules:
            module_observed_concepts, module_unobserved_concepts = category_module.get_observed_unobserved_parameters()
            observed_concepts += module_observed_concepts
            unobserved_concepts += module_unobserved_concepts
        for continuous_module in self.continuous_modules:
            module_dichotomies_parameters, module_observed_concepts, module_unobserved_concepts = continuous_module.get_dichotomies_observed_unobserved_parameters()
            dichotomies_parameters += module_dichotomies_parameters
            observed_concepts += module_observed_concepts
            unobserved_concepts += module_unobserved_concepts
        for periodic_module in self.periodic_modules:
            module_dichotomies_parameters, module_observed_concepts, module_unobserved_concepts = periodic_module.get_dichotomies_observed_unobserved_parameters()
            dichotomies_parameters += module_dichotomies_parameters
            observed_concepts += module_observed_concepts
            unobserved_concepts += module_unobserved_concepts
        return dichotomies_parameters, observed_concepts, unobserved_concepts

    def __repr__(self):
        string = ""
        for in_concept_idx in range(self.nb_in_features):
            found = False
            if not found:
                if in_concept_idx in self.binary_indices:
                    if len(string) > 0:
                        string += "\n"
                    string += "Binary " + str(in_concept_idx)
                    found = True

            if not found:
                for i, first_last_has_missing_values_tuple in enumerate(self.category_first_last_has_missing_values_tuples):
                    first, last, has_missing_values = first_last_has_missing_values_tuple
                    if in_concept_idx == first:
                        if len(string) > 0:
                            string += "\n"
                        has_missing_string = "!" if not has_missing_values else "?"
                        string += "Category " + str(first) + "-" + str(last) + " " + has_missing_string + " " + str(self.category_modules[i])
                        found = True
                        break
                    elif in_concept_idx > first and in_concept_idx <= last:
                        found = True
                        break

            if not found:
                for i, continuous_index_min_max_has_missing_values_tuple in enumerate(self.continuous_index_min_max_has_missing_values_tuples):
                    index, min_value, max_value, has_missing_values = continuous_index_min_max_has_missing_values_tuple
                    if in_concept_idx == index:
                        if len(string) > 0:
                            string += "\n"
                        has_missing_string = "!" if not has_missing_values else "?"
                        string += (
                            "Continuous "
                            + str(index)
                            + " in ["
                            + str(round(min_value, 3))
                            + ", "
                            + str(round(max_value, 3))
                            + "] "
                            + has_missing_string
                            + " "
                            + str(self.continuous_modules[i])
                        )
                        found = True
                        break

            if not found:
                for i, periodic_index_period_has_missing_values_tuple in enumerate(self.periodic_index_period_has_missing_values_tuples):
                    index, period, has_missing_values = periodic_index_period_has_missing_values_tuple
                    if in_concept_idx == index:
                        if len(string) > 0:
                            string += "\n"
                        has_missing_string = "!" if not has_missing_values else "?"
                        string += "Periodic " + str(index) + " of period " + str(round(period, 3)) + " " + has_missing_string + " " + str(self.periodic_modules[i])
                        found = True
                        break

        return string

    def to_simplified_string(self, feature_names):
        string = ""
        for idx in self.binary_indices:
            if len(string) > 0:
                string += "\n"
            string += "Binary " + feature_names[idx]

        for i, category_module in enumerate(self.category_modules):
            first_idx, last_idx, has_missing_values = self.category_first_last_has_missing_values_tuples[i]
            if len(string) > 0:
                string += "\n"
            category_name = feature_names[first_idx].split("_")[0]
            string += "Category " + category_name + " "
            string += (
                "("
                + str(
                    [
                        [feature_names[first_idx + j].split("_")[1] for j, val in enumerate(sublist) if val > 0]
                        for sublist in (1 * category_module.observed_concepts.data).detach().cpu().numpy().tolist()
                    ]
                )
                + ")"
            )

        for i, continuous_module in enumerate(self.continuous_modules):
            idx, min_value, max_value, has_missing_values = self.continuous_index_min_max_has_missing_values_tuples[i]
            if len(string) > 0:
                string += "\n"
            string += "Continuous " + feature_names[idx] + " in [" + str(round(min_value, 3)) + ", " + str(round(max_value, 3)) + "] " + str(continuous_module)

        for i, periodic_module in enumerate(self.periodic_modules):
            idx, period = self.periodic_index_period_has_missing_values_tuples[i]
            if len(string) > 0:
                string += "\n"
            string += "Periodic " + feature_names[idx] + " of period " + str(round(period, 3)) + " " + str(periodic_module)

        return string

    def load_string(self, init_string):
        lines = init_string.split("\n")
        in_continuous_periodic_counter = 0
        binaries_to_remove = list(range(len(self.binary_indices)))
        categories_to_remove = list(range(len(self.category_first_last_has_missing_values_tuples)))
        continuous_to_remove = list(range(len(self.continuous_index_min_max_has_missing_values_tuples)))
        periodic_to_remove = list(range(len(self.periodic_index_period_has_missing_values_tuples)))
        for i, line in enumerate(lines):
            if len(line) > 0:
                if in_continuous_periodic_counter > 0:
                    module_init_string += line + "\n"
                    in_continuous_periodic_counter -= 1
                    if in_continuous_periodic_counter == 0:
                        found = False
                        if not found:
                            for j, continuous_index_min_max_has_missing_values_tuple in enumerate(self.continuous_index_min_max_has_missing_values_tuples):
                                (
                                    continuous_index,
                                    continuous_min_value,
                                    continuous_max_value,
                                    has_missing_values,
                                ) = continuous_index_min_max_has_missing_values_tuple
                                if index == continuous_index:
                                    continuous_to_remove.remove(j)
                                    self.continuous_modules[j].load_string(module_init_string)
                                    found = True
                                    break
                        if not found:
                            for j, periodic_index_period_has_missing_values_tuple in enumerate(self.periodic_index_period_has_missing_values_tuples):
                                periodic_index, periodic_period, has_missing_values = periodic_index_period_has_missing_values_tuple
                                if index == periodic_index and period_string == str(round(periodic_period, 3)):
                                    periodic_to_remove.remove(j)
                                    self.periodic_modules[j].load_string(module_init_string)
                                    found = True
                                    break
                        if not found:
                            raise Exception(
                                "LNNInputModule input string is not formatted correctly.\n Continuous/Periodic not found at line " + str(i - 2) + " --> " + str(lines[i - 2])
                            )
                else:
                    if line[:6] == "Binary":
                        words = line.split(" ")
                        index = int(words[1])
                        for j, binary_index in enumerate(self.binary_indices):
                            if index == binary_index:
                                binaries_to_remove.remove(j)
                                break
                            if j == len(self.binary_indices) - 1:
                                raise Exception("LNNInputModule input string is not formatted correctly.\n Binary not found at line " + str(i) + " --> " + str(line))
                        continue
                    elif line[:8] == "Category":
                        words = line.split(" ")
                        first_last_string = words[1]
                        first_last_strings = first_last_string.split("-")
                        first_last_has_missing_values_tuple = (
                            int(first_last_strings[0]),
                            int(first_last_strings[1]),
                            words[2] == "?",
                        )
                        module_init_string = line[len("Category " + first_last_string + " " + words[2] + " ") :]
                        for j, category_first_last_has_missing_values_tuple in enumerate(self.category_first_last_has_missing_values_tuples):
                            if first_last_has_missing_values_tuple == category_first_last_has_missing_values_tuple:
                                categories_to_remove.remove(j)
                                self.category_modules[j].load_string(module_init_string)
                                break
                            if j == len(self.category_first_last_has_missing_values_tuples) - 1:
                                raise Exception("LNNInputModule input string is not formatted correctly.\n Category not found at line " + str(i) + " --> " + str(line))
                    elif line[:10] == "Continuous":
                        words = line.split(" ")
                        index = int(words[1])
                        min_value_string = words[3]
                        max_value_string = words[4]
                        has_missing_values_string = words[5]
                        module_init_string = (
                            line[len("Continuous " + str(index) + " in " + min_value_string + " " + max_value_string + " " + has_missing_values_string + " ") :] + "\n"
                        )
                        in_continuous_periodic_counter = 2
                    elif line[:8] == "Periodic":
                        words = line.split(" ")
                        index = int(words[1])
                        period_string = words[4]
                        has_missing_values_string = words[5]
                        module_init_string = line[len("Periodic " + str(index) + " of period " + period_string + " " + has_missing_values_string + " ") :] + "\n"
                        in_continuous_periodic_counter = 2
                    else:
                        raise Exception("LNNInputModule input string is not formatted correctly.\n Input type not recognized at line " + str(i) + " --> " + str(line))

        for i in reversed(binaries_to_remove):
            del self.binary_indices[i]

        for i in reversed(categories_to_remove):
            del self.category_first_last_has_missing_values_tuples[i]
            del self.category_modules[i]

        for i in reversed(continuous_to_remove):
            del self.continuous_index_min_max_has_missing_values_tuples[i]
            del self.continuous_modules[i]

        for i in reversed(periodic_to_remove):
            del self.periodic_index_period_has_missing_values_tuples[i]
            del self.periodic_modules[i]

    def backward_simplify(self, used_out_concepts):
        self.nb_out_concepts = len(used_out_concepts)

        new_binary_indices_indices = list(set(range(len(self.binary_indices))).intersection(set(used_out_concepts)))
        new_binary_indices_indices.sort()
        old_out_concept_idx = len(self.binary_indices)
        self.binary_indices = [self.binary_indices[new_binary_indices_idx] for new_binary_indices_idx in new_binary_indices_indices]

        categories_to_remove = []
        for i, category_module in enumerate(self.category_modules):
            first_out_concept_idx = old_out_concept_idx
            last_out_concept_idx = first_out_concept_idx + category_module.nb_out_concepts - 1
            category_used_out_concepts = [
                used_out_concept_idx - first_out_concept_idx
                for used_out_concept_idx in used_out_concepts
                if used_out_concept_idx >= first_out_concept_idx and used_out_concept_idx <= last_out_concept_idx
            ]
            if len(category_used_out_concepts) == 0:
                categories_to_remove.append(i)
            else:
                category_module.backward_simplify(category_used_out_concepts, keep_all_in_concepts=True)
            old_out_concept_idx = last_out_concept_idx + 1
        for i in reversed(categories_to_remove):
            del self.category_first_last_has_missing_values_tuples[i]
            del self.category_modules[i]

        continuous_to_remove = []
        for i, continuous_module in enumerate(self.continuous_modules):
            first_out_concept_idx = old_out_concept_idx
            last_out_concept_idx = first_out_concept_idx + continuous_module.nb_out_concepts - 1
            continuous_used_out_concepts = [
                used_out_concept_idx - first_out_concept_idx
                for used_out_concept_idx in used_out_concepts
                if used_out_concept_idx >= first_out_concept_idx and used_out_concept_idx <= last_out_concept_idx
            ]
            if len(continuous_used_out_concepts) == 0:
                continuous_to_remove.append(i)
            else:
                continuous_module.backward_simplify(continuous_used_out_concepts)
            old_out_concept_idx = last_out_concept_idx + 1
        for i in reversed(continuous_to_remove):
            del self.continuous_index_min_max_has_missing_values_tuples[i]
            del self.continuous_modules[i]

        periodic_to_remove = []
        for i, periodic_module in enumerate(self.periodic_modules):
            first_out_concept_idx = old_out_concept_idx
            last_out_concept_idx = first_out_concept_idx + periodic_module.nb_out_concepts - 1
            periodic_used_out_concepts = [
                used_out_concept_idx - first_out_concept_idx
                for used_out_concept_idx in used_out_concepts
                if used_out_concept_idx >= first_out_concept_idx and used_out_concept_idx <= last_out_concept_idx
            ]
            if len(periodic_used_out_concepts) == 0:
                periodic_to_remove.append(i)
            else:
                periodic_module.backward_simplify(periodic_used_out_concepts)
            old_out_concept_idx = last_out_concept_idx + 1
        for i in reversed(periodic_to_remove):
            del self.periodic_index_period_has_missing_values_tuples[i]
            del self.periodic_modules[i]

    def forward_simplify(self):
        used_out_concepts = []
        unused_out_idx_prob_pairs = []
        first_out_concept_idx = 0

        if len(self.binary_indices) > 0:
            used_out_concepts += list(range(len(self.binary_indices)))
            first_out_concept_idx += len(self.binary_indices)

        categories_to_remove = []
        for i, category_module in enumerate(self.category_modules):
            next_first_out_concept_idx = first_out_concept_idx + category_module.nb_out_concepts
            module_used_out_concepts, module_unused_out_idx_prob_pairs = category_module.forward_simplify()
            if len(module_used_out_concepts) == 0:
                categories_to_remove.append(i)
            else:
                used_out_concepts += [first_out_concept_idx + module_used_out_concept_idx for module_used_out_concept_idx in module_used_out_concepts]
                unused_out_idx_prob_pairs += [
                    (
                        first_out_concept_idx + module_unused_out_idx,
                        module_unused_out_idx_prob,
                    )
                    for module_unused_out_idx, module_unused_out_idx_prob in module_unused_out_idx_prob_pairs
                ]
            first_out_concept_idx = next_first_out_concept_idx
        for i in reversed(categories_to_remove):
            del self.category_first_last_has_missing_values_tuples[i]
            del self.category_modules[i]

        continuous_to_remove = []
        for i, continuous_module in enumerate(self.continuous_modules):
            next_first_out_concept_idx = first_out_concept_idx + continuous_module.nb_out_concepts
            module_used_out_concepts, module_unused_out_idx_prob_pairs = continuous_module.forward_simplify()
            if len(module_used_out_concepts) == 0:
                continuous_to_remove.append(i)
            else:
                used_out_concepts += [first_out_concept_idx + module_used_out_concept_idx for module_used_out_concept_idx in module_used_out_concepts]
                unused_out_idx_prob_pairs += [
                    (
                        first_out_concept_idx + module_unused_out_idx,
                        module_unused_out_idx_prob,
                    )
                    for module_unused_out_idx, module_unused_out_idx_prob in module_unused_out_idx_prob_pairs
                ]
            first_out_concept_idx = next_first_out_concept_idx
        for i in reversed(continuous_to_remove):
            del self.continuous_index_min_max_has_missing_values_tuples[i]
            del self.continuous_modules[i]

        periodic_to_remove = []
        for i, periodic_module in enumerate(self.periodic_modules):
            next_first_out_concept_idx = first_out_concept_idx + periodic_module.nb_out_concepts
            module_used_out_concepts, module_unused_out_idx_prob_pairs = periodic_module.forward_simplify()
            if len(module_used_out_concepts) == 0:
                periodic_to_remove.append(i)
            else:
                used_out_concepts += [first_out_concept_idx + module_used_out_concept_idx for module_used_out_concept_idx in module_used_out_concepts]
                unused_out_idx_prob_pairs += [
                    (
                        first_out_concept_idx + module_unused_out_idx,
                        module_unused_out_idx_prob,
                    )
                    for module_unused_out_idx, module_unused_out_idx_prob in module_unused_out_idx_prob_pairs
                ]
            first_out_concept_idx = next_first_out_concept_idx
        for i in reversed(periodic_to_remove):
            del self.periodic_index_period_has_missing_values_tuples[i]
            del self.periodic_modules[i]

        return used_out_concepts, unused_out_idx_prob_pairs

    def remove_duplicates(self, can_retrain):
        duplicate_lists = []
        first_out_concept_idx = len(self.binary_indices)

        for category_module in self.category_modules:
            next_first_out_concept_idx = first_out_concept_idx + category_module.nb_out_concepts
            module_duplicate_lists = category_module.remove_duplicates()
            if len(module_duplicate_lists) > 0:
                duplicate_lists += [
                    [first_out_concept_idx + module_out_concept_idx for module_out_concept_idx in module_duplicate_list] for module_duplicate_list in module_duplicate_lists
                ]
            first_out_concept_idx = next_first_out_concept_idx

        if can_retrain:
            must_retrain = False

            for continuous_module in self.continuous_modules:
                next_first_out_concept_idx = first_out_concept_idx + continuous_module.nb_out_concepts
                module_duplicate_lists, module_must_retrain = continuous_module.remove_duplicates()
                if len(module_duplicate_lists) > 0:
                    duplicate_lists += [
                        [first_out_concept_idx + module_out_concept_idx for module_out_concept_idx in module_duplicate_list] for module_duplicate_list in module_duplicate_lists
                    ]
                must_retrain = must_retrain or module_must_retrain
                first_out_concept_idx = next_first_out_concept_idx

            for periodic_module in self.periodic_modules:
                next_first_out_concept_idx = first_out_concept_idx + periodic_module.nb_out_concepts
                module_duplicate_lists, module_must_retrain = periodic_module.remove_duplicates()
                if len(module_duplicate_lists) > 0:
                    duplicate_lists += [
                        [first_out_concept_idx + module_out_concept_idx for module_out_concept_idx in module_duplicate_list] for module_duplicate_list in module_duplicate_lists
                    ]
                must_retrain = must_retrain or module_must_retrain
                first_out_concept_idx = next_first_out_concept_idx

            return duplicate_lists, must_retrain
        else:
            return duplicate_lists

    def prune(self, eval_model_func, init_loss=-1, log_files=[], progress_bar_hook=lambda weight: None):
        did_prune = False
        if init_loss < 0:
            init_loss = eval_model_func()
        current_weight = 0
        for category_module in self.category_modules:
            old_module_nb_weights = category_module.get_nb_weights()
            init_loss, local_did_prune = category_module.prune(
                eval_model_func, init_loss=init_loss, log_files=log_files, progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight)
            )
            did_prune = did_prune or local_did_prune
            current_weight += old_module_nb_weights
        for continuous_module in self.continuous_modules:
            old_module_nb_weights = continuous_module.get_nb_weights()
            init_loss, local_did_prune = continuous_module.prune(
                eval_model_func, init_loss=init_loss, log_files=log_files, progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight)
            )
            did_prune = did_prune or local_did_prune
            current_weight += old_module_nb_weights
        for periodic_module in self.periodic_modules:
            old_module_nb_weights = periodic_module.get_nb_weights()
            init_loss, local_did_prune = periodic_module.prune(
                eval_model_func, init_loss=init_loss, log_files=log_files, progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight)
            )
            did_prune = did_prune or local_did_prune
            current_weight += old_module_nb_weights
        return init_loss, did_prune

    def get_nb_weights(self):
        nb_weights = 0

        for category_module in self.category_modules:
            nb_weights += category_module.get_nb_weights()

        for continuous_module in self.continuous_modules:
            nb_weights += continuous_module.get_nb_weights()

        for periodic_module in self.periodic_modules:
            nb_weights += periodic_module.get_nb_weights()

        return nb_weights

    def discretize(self, discretization_method, eval_model_func=lambda: None, save_model_func=lambda: None, log_files=[], progress_bar_hook=lambda weight: None):
        if discretization_method == "thresh":
            for category_module in self.category_modules:
                category_module.discretize("thresh")
            for continuous_module in self.continuous_modules:
                continuous_module.discretize("thresh")
            for periodic_module in self.periodic_modules:
                periodic_module.discretize("thresh")
            return []
        else:
            losses = []
            current_weight = 0
            for category_module in self.category_modules:
                old_module_nb_weights = category_module.get_nb_weights()
                losses += category_module.discretize(
                    discretization_method,
                    eval_model_func=eval_model_func,
                    log_files=log_files,
                    progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
                )
                save_model_func()
                current_weight += old_module_nb_weights
            for continuous_module in self.continuous_modules:
                old_module_nb_weights = continuous_module.get_nb_weights()
                losses += continuous_module.discretize(
                    discretization_method,
                    eval_model_func=eval_model_func,
                    log_files=log_files,
                    progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
                )
                save_model_func()
                current_weight += old_module_nb_weights
            for periodic_module in self.periodic_modules:
                old_module_nb_weights = periodic_module.get_nb_weights()
                losses += periodic_module.discretize(
                    discretization_method,
                    eval_model_func=eval_model_func,
                    log_files=log_files,
                    progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
                )
                save_model_func()
                current_weight += old_module_nb_weights
            return losses

    def reorder_lexicographically(self):
        reordered_out_idcs = list(range(len(self.binary_indices)))
        first_out_concept_idx = len(self.binary_indices)

        for i, category_module in enumerate(self.category_modules):
            reordered_module_out_idcs = find_lexicographical_order(category_module.observed_concepts.data)
            category_module.observed_concepts.data = category_module.observed_concepts.data[reordered_module_out_idcs, :]
            if category_module.use_missing_values:
                category_module.missing_observed_concepts.data = category_module.missing_observed_concepts.data[reordered_module_out_idcs]
            reordered_out_idcs += [first_out_concept_idx + reordered_module_out_idx for reordered_module_out_idx in reordered_module_out_idcs]
            first_out_concept_idx += category_module.nb_out_concepts

        for i, continuous_module in enumerate(self.continuous_modules):
            reordered_module_out_idcs = continuous_module.reorder_lexicographically()
            reordered_out_idcs += [first_out_concept_idx + reordered_module_out_idx for reordered_module_out_idx in reordered_module_out_idcs]
            first_out_concept_idx += continuous_module.nb_out_concepts

        for i, periodic_module in enumerate(self.periodic_modules):
            reordered_module_out_idcs = periodic_module.reorder_lexicographically()
            reordered_out_idcs += [first_out_concept_idx + reordered_module_out_idx for reordered_module_out_idx in reordered_module_out_idcs]
            first_out_concept_idx += periodic_module.nb_out_concepts

        return reordered_out_idcs


class NeuralLogicNetwork(nn.Module):
    """
    Neural Logic Network (NLN):
        (1) NLN pre-processing modules
        (2) DNF
            (1) rules (layer of AND concepts with negation and unobserved necessary concepts)
            (2) logic programs (layer of OR concepts with no negation and with unobserved sufficient concepts)
    """

    def __init__(
        self,
        nb_in_features: int,
        nb_out_concepts: int,
        nb_concepts_per_hidden_layer: int = NB_RULES,
        nb_hidden_layers: int = 1,
        last_layer_is_OR_no_neg: bool = True,
        train_forw_weight_quant: str = TRAIN_FORW_WEIGHT_QUANT,
        approx_AND_OR_params: Union[None, Tuple[float, float, float]] = APPROX_PARAMS,
        category_first_last_has_missing_values_tuples: List[Tuple[int, int, bool]] = [],
        continuous_index_min_max_has_missing_values_tuples: List[Tuple[int, float, float, bool]] = [],
        periodic_index_period_has_missing_values_tuples: List[Tuple[int, float, bool]] = [],
        column_names: List[str] = [],
        nb_dichotomies_per_continuous: int = NB_DICHOTOMIES_PER_CONTINUOUS,
        nb_intervals_per_continuous: Union[int, None] = None,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        empty_init_targets: bool = EMPTY_INIT_TARGETS,
        empty_reset_in_concepts: bool = EMPTY_RESET_IN_CONCEPTS,
        device=DEVICE,
        verbose: bool = VERBOSE,
        init_string="",
    ):
        super().__init__()
        self.nb_in_features = nb_in_features
        self.nb_out_concepts = nb_out_concepts
        if nb_hidden_layers < 1:
            raise Exception("NLN must contain at least one hidden layer.")
        self.nb_hidden_layers = nb_hidden_layers
        self.nb_concepts_per_hidden_layer = nb_concepts_per_hidden_layer
        self.last_layer_is_OR_no_neg = last_layer_is_OR_no_neg
        self.train_forw_weight_quant = train_forw_weight_quant
        self.approx_AND_OR_params = approx_AND_OR_params
        self.feature_names = column_names
        self.nb_dichotomies_per_continuous = nb_dichotomies_per_continuous
        self.nb_intervals_per_continuous = nb_intervals_per_continuous
        self.random_init_obs = random_init_obs
        self.random_init_unobs = random_init_unobs
        self.empty_init_targets = empty_init_targets
        self.empty_reset_in_concepts = empty_reset_in_concepts
        self.device = device
        self.verbose = verbose

        self.category_first_last_has_missing_values_tuples = [
            (first_idx, last_idx, has_missing_values) for first_idx, last_idx, has_missing_values in category_first_last_has_missing_values_tuples
        ]
        self.continuous_index_min_max_has_missing_values_tuples = [
            (idx, min_value, max_value, has_missing_values) for idx, min_value, max_value, has_missing_values in continuous_index_min_max_has_missing_values_tuples
        ]
        self.periodic_index_period_has_missing_values_tuples = [
            (idx, period, has_missing_values) for idx, period, has_missing_values in periodic_index_period_has_missing_values_tuples
        ]
        self.input_module = NLNPreProcessingModules(
            nb_in_features,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            category_first_last_has_missing_values_tuples=category_first_last_has_missing_values_tuples,
            continuous_index_min_max_has_missing_values_tuples=continuous_index_min_max_has_missing_values_tuples,
            periodic_index_period_has_missing_values_tuples=periodic_index_period_has_missing_values_tuples,
            feature_names=column_names,
            category_nb_out_concepts=nb_concepts_per_hidden_layer,
            nb_dichotomies_per_continuous=nb_dichotomies_per_continuous,
            nb_intervals_per_continuous=nb_intervals_per_continuous,
            nb_out_concepts_per_continuous=nb_concepts_per_hidden_layer,
            random_init_obs=random_init_obs,
            empty_reset_in_concepts=empty_reset_in_concepts,
            device=device,
            verbose=verbose,
        )
        self.binary_indices = self.input_module.binary_indices.copy()
        in_concepts_group_first_stop_pairs = self.input_module.get_out_concepts_group_first_stop_pairs()

        layers = []
        for layer_idx in range(nb_hidden_layers):
            if layer_idx == 0:
                layers.append(
                    AndConcepts(
                        self.input_module.nb_out_concepts,
                        nb_concepts_per_hidden_layer,
                        use_negation=True,
                        use_unobserved=True,
                        random_init_obs=random_init_obs,
                        random_init_unobs=random_init_unobs,
                        empty_reset_in_concepts=empty_reset_in_concepts,
                        train_forw_weight_quant=train_forw_weight_quant,
                        approx_AND_OR_params=approx_AND_OR_params,
                        in_concepts_group_first_stop_pairs=in_concepts_group_first_stop_pairs,
                        device=device,
                        verbose=verbose,
                    )
                )
            else:
                layers.append(
                    AndConcepts(
                        nb_concepts_per_hidden_layer,
                        nb_concepts_per_hidden_layer,
                        use_negation=True,
                        use_unobserved=True,
                        random_init_obs=random_init_obs,
                        random_init_unobs=random_init_unobs,
                        empty_reset_in_concepts=empty_reset_in_concepts,
                        train_forw_weight_quant=train_forw_weight_quant,
                        approx_AND_OR_params=approx_AND_OR_params,
                        device=device,
                        verbose=verbose,
                    )
                )
        if last_layer_is_OR_no_neg:
            layers.append(
                OrConcepts(
                    nb_concepts_per_hidden_layer,
                    nb_out_concepts,
                    use_negation=False,
                    use_unobserved=True,
                    random_init_obs=random_init_obs,
                    random_init_unobs=random_init_unobs,
                    empty_reset_in_concepts=empty_reset_in_concepts,
                    train_forw_weight_quant=train_forw_weight_quant,
                    approx_AND_OR_params=approx_AND_OR_params,
                    device=device,
                    verbose=verbose,
                )
            )
        else:
            layers.append(
                AndConcepts(
                    nb_concepts_per_hidden_layer,
                    nb_out_concepts,
                    use_negation=True,
                    use_unobserved=True,
                    random_init_obs=random_init_obs,
                    random_init_unobs=random_init_unobs,
                    empty_reset_in_concepts=empty_reset_in_concepts,
                    train_forw_weight_quant=train_forw_weight_quant,
                    approx_AND_OR_params=approx_AND_OR_params,
                    device=device,
                    verbose=verbose,
                )
            )
        if empty_init_targets:
            layers[-1].observed_concepts.data = 0 * torch.ones((nb_out_concepts, nb_concepts_per_hidden_layer), device=device)
            layers[-1].unobserved_concepts.data = 1 * torch.ones(nb_out_concepts, device=device)
        self.layers = nn.Sequential(*layers)

        if len(init_string) > 0:
            self.load_string(init_string)

        self.to(device)

    def set(
        self,
        empty_reset_in_concepts=None,
        train_forw_weight_quant=None,
        approx_AND_OR_params="",
        device=None,
        verbose=None,
    ):
        if empty_reset_in_concepts != None and empty_reset_in_concepts != self.empty_reset_in_concepts:
            self.empty_reset_in_concepts = empty_reset_in_concepts
        if train_forw_weight_quant != None and train_forw_weight_quant != self.train_forw_weight_quant:
            self.train_forw_weight_quant = train_forw_weight_quant
        if approx_AND_OR_params != "" and approx_AND_OR_params != self.approx_AND_OR_params:
            self.approx_AND_OR_params = approx_AND_OR_params
        if verbose != None and verbose != self.verbose:
            self.verbose = verbose
        self.input_module.set(
            empty_reset_in_concepts=empty_reset_in_concepts,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            device=device,
            verbose=verbose,
        )
        for layer in self.layers:
            layer.set(
                empty_reset_in_concepts=empty_reset_in_concepts,
                train_forw_weight_quant=train_forw_weight_quant,
                approx_AND_OR_params=approx_AND_OR_params,
                device=device,
                verbose=verbose,
            )
        if device != None and device != self.device:
            self.device = device
            self.to(device)

    @staticmethod
    def merge_modules(modules):
        if len(set([module.nb_in_concepts for module in modules])) > 1:
            raise Exception("The merged modules must have the same number of input concepts.")
        if len(set([module.nb_out_concepts for module in modules])) > 1:
            raise Exception("The merged modules must have the same number of output concepts.")
        if len(set([module.nb_hidden_layers for module in modules])) > 1:
            raise Exception("The merged modules must have the same number of hidden layers.")
        if len(set([module.last_layer_is_OR_no_neg for module in modules])) > 1:
            raise Exception("The merged modules must have the same type of output layer.")
        for module_idx in range(2, len(modules)):
            if (
                modules[0].category_first_last_has_missing_values_tuples != modules[module_idx].category_first_last_has_missing_values_tuples
                or modules[0].continuous_index_min_max_has_missing_values_tuples != modules[module_idx].continuous_index_min_max_has_missing_values_tuples
                or modules[0].periodic_index_period_has_missing_values_tuples != modules[module_idx].periodic_index_period_has_missing_values_tuples
                or modules[0].feature_names != modules[module_idx].feature_names
            ):
                raise Exception("The merged modules must work with the same input format.")

        def get_most_used_value(values):
            return sorted([(value, values.count(value)) for value in set(values)], key=lambda pair: pair[1], reverse=True)[0][0]

        merged_module = NeuralLogicNetwork(
            modules[0].nb_in_concepts,
            modules[0].nb_out_concepts,
            nb_concepts_per_hidden_layer=sum([module.nb_concepts_per_hidden_layer for module in modules]),
            nb_hidden_layers=modules[0].nb_hidden_layers,
            train_forw_weight_quant=get_most_used_value([module.train_forw_weight_quant for module in modules]),
            approx_AND_OR_params=get_most_used_value([module.approx_AND_OR_params for module in modules]),
            category_first_last_has_missing_values_tuples=modules[0].category_first_last_has_missing_values_tuples,
            continuous_index_min_max_has_missing_values_tuples=modules[0].continuous_index_min_max_has_missing_values_tuples,
            periodic_index_period_has_missing_values_tuples=modules[0].periodic_index_period_has_missing_values_tuples,
            column_names=modules[0].feature_names,
            nb_dichotomies_per_continuous=sum([module.nb_dichotomies_per_continuous for module in modules]),
            nb_intervals_per_continuous=(
                None if None in [module.nb_intervals_per_continuous for module in modules] else sum([module.nb_intervals_per_continuous for module in modules])
            ),
            random_init_obs=get_most_used_value([module.random_init_obs for module in modules]),
            random_init_unobs=get_most_used_value([module.random_init_unobs for module in modules]),
            empty_init_targets=get_most_used_value([module.empty_init_targets for module in modules]),
            empty_reset_in_concepts=get_most_used_value([module.empty_reset_in_concepts for module in modules]),
            device=get_most_used_value([module.device for module in modules]),
            verbose=bool(round(mean([module.verbose for module in modules]))),
        )

        for i, merged_layer in enumerate(reversed(merged_module.layers)):
            i = len(merged_module.layers) - i - 1
            if i > 0:
                merged_layer.nb_in_concepts = sum([module.layers[i].nb_in_concepts for module in modules])
                if i == len(merged_module.layers) - 1:
                    merged_layer.observed_concepts.data = torch.concat([module.layers[i].observed_concepts.data for module in modules], dim=1)
                    if merged_layer.use_unobserved:
                        merged_layer.unobserved_concepts.data = torch.zeros_like(merged_layer.unobserved_concepts.data)
                        for module in modules:
                            merged_layer.unobserved_concepts.data += (1 / len(modules)) * module.layers[i].unobserved_concepts.data
                else:
                    merged_layer.nb_out_concepts = sum([module.layers[i].nb_out_concepts for module in modules])
                    merged_layer.observed_concepts.data = torch.zeros((merged_layer.nb_out_concepts, merged_layer.nb_in_concepts), device=merged_layer.device)
                    curr_out_count = 0
                    curr_in_count = 0
                    for module_idx, module in enumerate(modules):
                        merged_layer.observed_concepts.data[
                            curr_out_count : curr_out_count + module.layers[i].nb_out_concepts, curr_in_count : curr_in_count + module.layers[i].nb_in_concepts
                        ] = module.layers[i].observed_concepts.data
                        curr_out_count += module.layers[i].nb_out_concepts
                        curr_in_count += module.layers[i].nb_in_concepts

        merged_module.input_module, idx_translation_by_module = NLNPreProcessingModules.merge_modules(
            [module.input_module for module in modules],
            merged_module.nb_in_features,
            train_forw_weight_quant=merged_module.train_forw_weight_quant,
            approx_AND_OR_params=merged_module.approx_AND_OR_params,
            category_first_last_has_missing_values_tuples=merged_module.category_first_last_has_missing_values_tuples,
            continuous_index_min_max_has_missing_values_tuples=merged_module.continuous_index_min_max_has_missing_values_tuples,
            periodic_index_period_has_missing_values_tuples=merged_module.periodic_index_period_has_missing_values_tuples,
            feature_names=merged_module.feature_names,
            category_nb_out_concepts=merged_module.nb_concepts_per_hidden_layer,
            nb_dichotomies_per_continuous=merged_module.nb_dichotomies_per_continuous,
            nb_intervals_per_continuous=merged_module.nb_intervals_per_continuous,
            nb_out_concepts_per_continuous=merged_module.nb_concepts_per_hidden_layer,
            random_init_obs=merged_module.random_init_obs,
            empty_reset_in_concepts=merged_module.empty_reset_in_concepts,
            device=merged_module.device,
            verbose=merged_module.verbose,
        )

        merged_first_layer = merged_module.layers[0]
        merged_first_layer.ungroup()
        merged_first_layer.in_concepts_group_first_stop_pairs = merged_module.input_module.get_out_concepts_group_first_stop_pairs()
        merged_first_layer.nb_in_concepts = merged_module.input_module.nb_out_concepts
        merged_first_layer.nb_out_concepts = merged_module.layers[1].nb_in_concepts
        merged_first_layer.observed_concepts.data = torch.zeros((merged_first_layer.nb_out_concepts, merged_first_layer.nb_in_concepts), device=merged_module.device)
        if merged_first_layer.use_unobserved:
            merged_first_layer.unobserved_concepts.data = torch.zeros((merged_first_layer.nb_out_concepts), device=merged_module.device)
        nb_out_concepts = 0
        for module_idx, module in enumerate(modules):
            module_first_layer = module.layers[0]
            if module_first_layer.is_grouped:
                module_first_layer.ungroup()
            for module_in_idx in range(module_first_layer.nb_in_concepts):
                merged_in_idx = idx_translation_by_module[module_idx][module_in_idx]
                merged_first_layer.observed_concepts.data[nb_out_concepts : nb_out_concepts + module_first_layer.nb_out_concepts, merged_in_idx] = (
                    module_first_layer.observed_concepts.data[:, module_in_idx]
                )
            if merged_first_layer.use_unobserved:
                merged_first_layer.unobserved_concepts.data[nb_out_concepts : nb_out_concepts + module_first_layer.nb_out_concepts] = module_first_layer.unobserved_concepts.data
            nb_out_concepts += module_first_layer.nb_out_concepts

        return merged_module

    def update_parameters(self):
        self.input_module.update_parameters()
        # if USE_RULE_MODULE:
        #     for first_in_concept_idx, stop_in_concept_idx in self.layers[0].in_concepts_group_first_stop_pairs:
        #         if stop_in_concept_idx - first_in_concept_idx == self.layers[0].nb_out_concepts:
        #             self.layers[0].observed_concepts.data[:, first_in_concept_idx:stop_in_concept_idx][
        #                 torch.arange(0, self.layers[0].nb_out_concepts).unsqueeze(1) != torch.arange(0, self.layers[0].nb_out_concepts).unsqueeze(0)
        #             ] = 0
        for layer in self.layers:
            layer.update_parameters()

    def override(self, new_override):
        self.input_module.override(new_override)
        for layer in self.layers:
            layer.override(new_override)

    def review_unused_concepts(self):
        if RESET_UNUSED_CONCEPTS:
            for i, layer in enumerate(reversed(self.layers)):
                i = len(self.layers) - i - 1
                if i > 0 and self.layers[i - 1].use_unobserved:
                    also_unused_in_concepts = [
                        used_out_concept
                        for used_out_concept in range(self.layers[i - 1].nb_out_concepts)
                        if self.layers[i - 1].unobserved_concepts.data[used_out_concept].item() == 1 - self.layers[i - 1].full_unobserved_value
                    ]
                else:
                    also_unused_in_concepts = []
                if i == len(self.layers) - 1:
                    used_in_concepts = layer.review_unused_concepts(list(range(layer.nb_out_concepts)), also_unused_in_concepts)
                else:
                    if not USE_RULE_MODULE or i > 0:
                        used_in_concepts = layer.review_unused_concepts(used_in_concepts, also_unused_in_concepts)
                    else:
                        used_rules = used_in_concepts
                        layer.review_unused_concepts(used_rules, [], do_check_in_concepts=False)
                        if len(layer.in_concepts_group_first_stop_pairs) > 0:
                            unused_rules = [rule_idx for rule_idx in range(layer.nb_out_concepts) if rule_idx not in used_rules]
                            input_module_out_concepts_to_reset = []
                            for (
                                first_in_concept_idx,
                                stop_in_concept_idx,
                            ) in layer.in_concepts_group_first_stop_pairs:
                                if stop_in_concept_idx - first_in_concept_idx == layer.nb_out_concepts:
                                    input_module_out_concepts_to_reset += [first_in_concept_idx + unused_rule_idx for unused_rule_idx in unused_rules]
                            self.input_module.reset_out_concepts(input_module_out_concepts_to_reset)
            if not USE_RULE_MODULE:
                self.input_module.review_unused_concepts(used_in_concepts)

    def forward(self, x):
        """Forward pass"""
        if len(x.shape) > 1 or self.nb_in_features > 1:
            x_shape_prefix = list(x.shape[0:-1])
        else:
            x_shape_prefix = [x.shape[0]]
        x_v = x.view(-1, self.nb_in_features)
        result = self.input_module(x_v)
        result = self.layers(result)
        return result.view(x_shape_prefix + [self.nb_out_concepts])

    def add_regularization(self, loss):
        loss = self.input_module.add_regularization(loss)
        loss = self.layers[0].add_regularization(loss)
        for layer in self.layers[1:]:
            loss = layer.add_regularization(loss)
        return loss

    def get_dichotomies_observed_unobserved_parameters(self):
        dichotomies_parameters, observed_concepts, unobserved_concepts = self.input_module.get_dichotomies_observed_unobserved_parameters()
        for layer in self.layers:
            layer_observed_concepts, layer_unobserved_concepts = layer.get_observed_unobserved_parameters()
            observed_concepts += layer_observed_concepts
            unobserved_concepts += layer_unobserved_concepts
        return dichotomies_parameters, observed_concepts, unobserved_concepts

    def __repr__(self):
        string = str(self.input_module)
        if len(string) > 0:
            string += "\n"
        string += "["
        for i, layer in enumerate(self.layers):
            if i > 0:
                string += ", \n"
            string += str(layer)
        string += "]"
        return string

    def to_simplified_string(self):
        string = self.input_module.to_simplified_string(self.feature_names)
        if len(string) > 0:
            string += "\n"
        string += "["
        for i, layer in enumerate(self.layers):
            if i > 0:
                string += ", \n"
            string += str(layer)
        string += "]"
        return string

    def show(
        self,
        fig_ax=-1,
        filename="",
        train_no_reg_loss=-1,
        train_loss=-1,
        valid_raw_loss=-1,
        valid_stoch_loss=-1,
        valid_thresh_loss=-1,
        save_dpi=100,
        one_rule_at_a_time=False,
        is_single_rule=False,
    ):
        if one_rule_at_a_time:
            if torch.max(self.layers[-1].observed_concepts).item() > 0:
                for out_OR_concept_idx in range(self.nb_out_concepts):
                    single_OR_copy = copy.deepcopy(self)
                    single_OR_copy.feature_names = single_OR_copy.feature_names[: single_OR_copy.nb_in_features] + [
                        single_OR_copy.feature_names[-1 * (single_OR_copy.nb_out_concepts - out_OR_concept_idx)]
                    ]
                    single_OR_copy.nb_out_concepts = 1
                    single_OR_copy.layers[-1].observed_concepts.data = single_OR_copy.layers[-1].observed_concepts.data[out_OR_concept_idx : out_OR_concept_idx + 1, :]
                    single_OR_copy.layers[-1].unobserved_concepts.data = single_OR_copy.layers[-1].unobserved_concepts.data[out_OR_concept_idx : out_OR_concept_idx + 1]
                    single_OR_copy.layers[-1].nb_out_concepts = single_OR_copy.layers[-1].observed_concepts.data.size(0)
                    single_OR_copy.simplify()
                    if torch.max(single_OR_copy.layers[-1].observed_concepts.data) > 0:
                        for mid_AND_concept_idx in range(single_OR_copy.layers[-1].nb_in_concepts):
                            single_AND_copy = copy.deepcopy(single_OR_copy)
                            single_AND_copy.layers[-1].observed_concepts.data = single_AND_copy.layers[-1].observed_concepts.data[:, mid_AND_concept_idx : mid_AND_concept_idx + 1]
                            single_AND_copy.layers[-1].nb_in_concepts = single_AND_copy.layers[-1].observed_concepts.data.size(1)
                            single_AND_copy.layers[-2].observed_concepts.data = single_AND_copy.layers[-2].observed_concepts.data[mid_AND_concept_idx : mid_AND_concept_idx + 1, :]
                            single_AND_copy.layers[-2].unobserved_concepts.data = single_AND_copy.layers[-2].unobserved_concepts.data[mid_AND_concept_idx : mid_AND_concept_idx + 1]
                            single_AND_copy.layers[-2].nb_out_concepts = single_AND_copy.layers[-2].observed_concepts.data.size(0)
                            single_AND_copy.simplify()
                            single_AND_copy.layers = single_AND_copy.layers[:-1]
                            if filename == "":
                                single_AND_copy.show(
                                    train_no_reg_loss=train_no_reg_loss,
                                    train_loss=train_loss,
                                    valid_raw_loss=valid_raw_loss,
                                    valid_stoch_loss=valid_stoch_loss,
                                    valid_thresh_loss=valid_thresh_loss,
                                    save_dpi=save_dpi,
                                    is_single_rule=True,
                                )
                            else:
                                single_AND_copy.show(
                                    filename=filename + "_" + str(out_OR_concept_idx) + "-" + str(mid_AND_concept_idx),
                                    train_no_reg_loss=train_no_reg_loss,
                                    train_loss=train_loss,
                                    valid_raw_loss=valid_raw_loss,
                                    valid_stoch_loss=valid_stoch_loss,
                                    valid_thresh_loss=valid_thresh_loss,
                                    save_dpi=save_dpi,
                                    is_single_rule=True,
                                )
        else:
            # Helper Functions
            def text_fixed_size(
                ax,
                x,
                y,
                text,
                h=12,
                color="k",
                flip_vert=False,
                bold=False,
                align_left=False,
            ):
                if bold:
                    tp = TextPath((0.0, 0.0), text, size=1, prop=FontProperties(weight="bold"))
                else:
                    tp = TextPath((0.0, 0.0), text, size=1)
                x0, y0 = np.amin(np.array(tp.vertices), axis=0)
                x1, y1 = np.amax(np.array(tp.vertices), axis=0)
                verts = []
                for vert in tp.vertices:
                    vert_0 = vert[0] - 0.5 * (x1 - x0) if not align_left else vert[0]
                    vx = vert_0 * h + x
                    vert_1 = vert[1] if not flip_vert else y1 - (vert[1] - y0)
                    vy = (vert_1 - 0.4) * h + y
                    verts += [[vx, vy]]
                verts = np.array(verts)
                tp = Path(verts, tp.codes)
                ax.add_patch(PathPatch(tp, facecolor=color, lw=0))

            def combination_node(ax, x, y, flip_vert, color, unobserved_concept=-1):
                ax.add_patch(Circle((x, y), radius=10, edgecolor="k", facecolor=color, linewidth=0.5))
                if not flip_vert:
                    # text_fixed_size(ax, x - 1, y + 1.5, r"$ \vee $", h=16)
                    text_fixed_size(ax, x - 1, y + 1.5, r"$ \cup $", h=16)
                else:
                    # text_fixed_size(ax, x - 1, y + 1.5, r"$ \wedge $", h=16)
                    text_fixed_size(ax, x - 1, y + 1.5, r"$ \cap $", h=16)
                if unobserved_concept >= 0:
                    text_fixed_size(
                        ax,
                        x + 0.85 * 10,
                        y + 0.95 * 10,
                        str(round(unobserved_concept, 3)),
                        h=9,
                        align_left=True,
                    )

            def AND_node(ax, x, y, unobserved_concept=-1):
                combination_node(
                    ax,
                    x,
                    y,
                    True,
                    (71 / 255, 141 / 255, 203 / 255),
                    unobserved_concept=unobserved_concept,
                )

            def OR_node(ax, x, y, unobserved_concept=-1):
                combination_node(
                    ax,
                    x,
                    y,
                    False,
                    (113 / 255, 191 / 255, 68 / 255),
                    unobserved_concept=unobserved_concept,
                )

            def arrow(
                ax,
                start_x,
                start_y,
                end_x,
                end_y,
                color="k",
                alpha=1,
                use_arrow_head=False,
            ):
                if use_arrow_head:
                    ax.arrow(
                        start_x,
                        start_y,
                        end_x - start_x,
                        end_y - start_y,
                        length_includes_head=True,
                        width=0.75,
                        head_width=7.5,
                        head_length=7.5,
                        shape="full",
                        color=color,
                        alpha=alpha,
                    )
                else:
                    ax.arrow(
                        start_x,
                        start_y,
                        end_x - start_x,
                        end_y - start_y,
                        length_includes_head=True,
                        width=0.75,
                        head_width=0.75,
                        head_length=7.5,
                        shape="full",
                        color=color,
                        alpha=alpha,
                    )
                    # ax.plot([start_x, end_x], [start_y, end_y], lw=1.45, color=color, alpha=alpha)

            # Constants
            x_step = 120
            y_step = 26
            x_node_buffer = 15
            x_indir_buffer = 10
            y_text_line = 12 * 1.25
            boundary_rel_x_min_spacing = 0.17

            if torch.max(torch.abs(self.layers[-1].observed_concepts)).item() > 0:

                # Find maximum layer width
                nb_variables = (
                    len(self.input_module.binary_indices)
                    + len(self.input_module.category_modules)
                    + len(self.input_module.continuous_modules)
                    + len(self.input_module.periodic_modules)
                )
                input_module_group_widths = []
                input_categories_used_in_values = []
                for category_module in self.input_module.category_modules:
                    if not is_single_rule:
                        extra_missing_values = 1 if category_module.use_missing_values else 0
                        input_module_group_widths.append(
                            [
                                category_module.nb_in_concepts + extra_missing_values,
                                category_module.nb_out_concepts,
                            ]
                        )
                    else:
                        input_categories_used_in_values.append(torch.nonzero(category_module.observed_concepts.data[0, :]).view(-1).tolist())
                        if category_module.use_missing_values:
                            input_categories_used_in_values[-1].append(-1)
                        input_module_group_widths.append([len(input_categories_used_in_values[-1]), 1])
                first_out_concept_with_same_graphs_list = []
                for continuous_module in self.input_module.continuous_modules:
                    if not is_single_rule:
                        # input_module_group_widths.append(
                        #     [
                        #         continuous_module.dichotomies.nb_dichotomies,
                        #         continuous_module.intervals.nb_out_concepts,
                        #         continuous_module.out_concepts.nb_out_concepts,
                        #     ]
                        # )
                        first_out_concept_with_same_graphs = []
                        for out_i in range(continuous_module.out_concepts.nb_out_concepts):
                            is_duplicate = False
                            for out_j in first_out_concept_with_same_graphs:
                                if torch.equal(continuous_module.out_concepts.observed_concepts.data[out_i, :], continuous_module.out_concepts.observed_concepts.data[out_j, :]):
                                    is_duplicate = True
                                    first_out_concept_with_same_graphs.append(out_j)
                                    break
                            if not is_duplicate:
                                first_out_concept_with_same_graphs.append(out_i)
                        first_out_concept_with_same_graphs_list.append(first_out_concept_with_same_graphs)
                        input_module_group_widths.append(
                            [
                                6
                                * sum(
                                    [
                                        1
                                        for out_concept, first_out_concept_with_same_graphs in enumerate(first_out_concept_with_same_graphs)
                                        if out_concept == first_out_concept_with_same_graphs
                                    ]
                                )
                                - 1
                            ]
                        )
                        # input_module_group_widths.append([6 * continuous_module.out_concepts.nb_out_concepts - 1])
                    else:
                        input_module_group_widths.append([5])

                for periodic_module in self.input_module.periodic_modules:
                    input_module_group_widths.append(
                        [
                            periodic_module.dichotomies.nb_dichotomies,
                            periodic_module.intervals.nb_out_concepts,
                            periodic_module.out_concepts.nb_out_concepts,
                        ]
                    )
                if len(input_module_group_widths) > 0:
                    layer_width = max(
                        len(self.input_module.binary_indices) + sum([max(group_widths) for group_widths in input_module_group_widths]),
                        nb_variables,
                    )
                    if not is_single_rule:
                        layer_widths_with_spacing = [layer_width + nb_variables - 1 + len(self.input_module.continuous_modules)]
                    else:
                        layer_widths_with_spacing = [layer_width + nb_variables - 1]
                else:
                    layer_widths_with_spacing = [nb_variables]
                for layer in self.layers:
                    layer_widths_with_spacing.append(layer.nb_out_concepts)
                max_layer_width_with_spacing = max(layer_widths_with_spacing)

                # Find output indices
                output_target_idcs = self._get_target_indices()

                # Define figure
                if train_loss >= 0 or train_no_reg_loss >= 0 or valid_raw_loss >= 0 or valid_stoch_loss >= 0 or valid_thresh_loss >= 0:
                    y_heading = 1.5 * y_step
                else:
                    y_heading = 0
                if not is_single_rule or len(self.input_module.continuous_modules) + len(self.input_module.periodic_modules) == 0:
                    height = y_step * max_layer_width_with_spacing + y_heading
                else:
                    height = y_step * max_layer_width_with_spacing + y_heading + 1.05 * x_node_buffer
                if len(input_module_group_widths) == 0:
                    width = x_step * (0.5 + 0.5 + len(self.layers) + 0.5 + 0.5) + x_node_buffer
                else:
                    width = x_step * (0.5 + 0.5 + 2 + len(self.layers) + 0.5 + 0.5) + x_node_buffer
                if is_single_rule:
                    width += 0.25 * x_step
                if self.layers[-1].use_unobserved:
                    width += x_indir_buffer
                if filename != "":
                    plt.ion()
                name = filename if filename != "" else "GNLN"
                if fig_ax == -1:
                    fig, ax = plt.subplots(
                        1,
                        1,
                        num=name,
                        gridspec_kw={"left": 0.0, "right": 1.0, "bottom": 0.0, "top": 1.0},
                    )
                else:
                    fig, ax = fig_ax
                fig.set_frameon(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                dpi = fig.get_dpi()
                fig.set_size_inches(width / dpi, height / dpi)
                ax.set_ylim(0, height)
                ax.set_xlim(0, width)
                ax.set_aspect("equal")

                # Losses
                current_y = height - 0.5 * y_step
                if train_loss >= 0:
                    current_x = width * (0.2 / 5)
                    text_fixed_size(ax, current_x, current_y, f"{train_loss:.3e}", h=12, align_left=True)
                if train_no_reg_loss >= 0:
                    current_x = width * (1.2 / 5)
                    text_fixed_size(
                        ax,
                        current_x,
                        current_y,
                        f"{train_no_reg_loss:.3e}",
                        h=12,
                        align_left=True,
                    )
                if valid_raw_loss >= 0:
                    current_x = width * (2.2 / 5)
                    text_fixed_size(
                        ax,
                        current_x,
                        current_y,
                        f"{valid_raw_loss:.3e}",
                        h=12,
                        align_left=True,
                    )
                if valid_stoch_loss >= 0:
                    current_x = width * (3.2 / 5)
                    text_fixed_size(
                        ax,
                        current_x,
                        current_y,
                        f"{valid_stoch_loss:.3e}",
                        h=12,
                        align_left=True,
                    )
                if valid_thresh_loss >= 0:
                    current_x = width * (4.2 / 5)
                    text_fixed_size(
                        ax,
                        current_x,
                        current_y,
                        f"{valid_thresh_loss:.3e}",
                        h=12,
                        align_left=True,
                    )

                # Binary Inputs
                coordinates = [[]]
                if layer_widths_with_spacing[0] == max_layer_width_with_spacing:
                    input_y_step = y_step
                else:
                    input_y_step = (max_layer_width_with_spacing - 1) / layer_widths_with_spacing[0] * y_step
                if len(input_module_group_widths) > 0 or layer_widths_with_spacing[0] == max_layer_width_with_spacing:
                    current_y = height - y_heading - 0.5 * y_step
                else:
                    current_y = height - y_heading - 0.5 * y_step - input_y_step / 2
                for idx in self.input_module.binary_indices:
                    current_x = 0.5 * x_step
                    text_fixed_size(ax, current_x, current_y, self.feature_names[idx])
                    if len(input_module_group_widths) > 0:
                        arrow(
                            ax,
                            current_x + 0.5 * x_step,
                            current_y,
                            current_x + 2.5 * x_step + x_node_buffer,
                            current_y,
                        )
                        coordinates[-1].append((current_x + 2.5 * x_step, current_y))
                    else:
                        coordinates[-1].append((current_x + 0.5 * x_step, current_y))
                    if len(input_module_group_widths) > 0:
                        current_y -= 2 * input_y_step
                    else:
                        current_y -= input_y_step
                if len(input_module_group_widths) == 0:
                    current_x += 0.5 * x_step

                # Category Inputs
                group_idx = 0
                for i, category_module in enumerate(self.input_module.category_modules):
                    first_idx, last_idx, has_missing_values = self.input_module.category_first_last_has_missing_values_tuples[i]
                    category_name, values_names = self.input_module.get_category_values_names(i)
                    current_x = 0.5 * x_step
                    loc_current_y = current_y - input_y_step * (max(input_module_group_widths[group_idx]) - 1) / 2
                    text_fixed_size(
                        ax,
                        current_x,
                        loc_current_y,
                        category_name,
                    )
                    group_coordinates = [[]]
                    current_x += x_step
                    if input_module_group_widths[group_idx][0] == max(input_module_group_widths[group_idx]):
                        loc_current_y = current_y
                    else:
                        loc_current_y = current_y - input_y_step * ((max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][0]) / 2
                    for idx in range(first_idx, last_idx + 1):
                        if not is_single_rule or idx - first_idx in input_categories_used_in_values[i]:
                            text_fixed_size(
                                ax,
                                current_x,
                                loc_current_y,
                                values_names[idx - first_idx],
                            )
                            group_coordinates[-1].append((current_x + 0.5 * x_step, loc_current_y))
                            if input_module_group_widths[group_idx][0] == max(input_module_group_widths[group_idx]):
                                loc_current_y -= input_y_step
                            else:
                                loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][0]
                    if has_missing_values:
                        text_fixed_size(
                            ax,
                            current_x,
                            loc_current_y,
                            "?",
                        )
                        group_coordinates[-1].append((current_x + 0.5 * x_step, loc_current_y))
                        if input_module_group_widths[group_idx][0] == max(input_module_group_widths[group_idx]):
                            loc_current_y -= input_y_step
                        else:
                            loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][0]
                    group_coordinates.append([])
                    current_x += 1.5 * x_step
                    if input_module_group_widths[group_idx][1] == max(input_module_group_widths[group_idx]):
                        loc_current_y = current_y
                    else:
                        loc_current_y = current_y - input_y_step * ((max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][1]) / 2
                    for out_concept in range(category_module.nb_out_concepts):
                        OR_node(ax, current_x, loc_current_y)
                        group_coordinates[-1].append((current_x, loc_current_y))
                        coordinates[-1].append((current_x, loc_current_y))
                        if input_module_group_widths[group_idx][1] == max(input_module_group_widths[group_idx]):
                            loc_current_y -= input_y_step
                        else:
                            loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][1]
                    for in_concept in range(category_module.nb_in_concepts):
                        for out_concept in range(category_module.nb_out_concepts):
                            weight = category_module.observed_concepts.data[out_concept, in_concept].item()
                            if weight > 0:
                                if is_single_rule:
                                    in_concept = input_categories_used_in_values[i].index(in_concept)
                                arrow(
                                    ax,
                                    group_coordinates[-2][in_concept][0],
                                    group_coordinates[-2][in_concept][1],
                                    group_coordinates[-1][out_concept][0] - x_node_buffer,
                                    group_coordinates[-1][out_concept][1],
                                    color="k",
                                    alpha=weight,
                                )
                    if has_missing_values:
                        in_concept = -1
                        for out_concept in range(category_module.nb_out_concepts):
                            weight = category_module.missing_observed_concepts.data[out_concept].item()
                            if weight > 0:
                                if is_single_rule:
                                    in_concept = input_categories_used_in_values[i].index(in_concept)
                                arrow(
                                    ax,
                                    group_coordinates[-2][in_concept][0],
                                    group_coordinates[-2][in_concept][1],
                                    group_coordinates[-1][out_concept][0] - x_node_buffer,
                                    group_coordinates[-1][out_concept][1],
                                    color="k",
                                    alpha=weight,
                                )
                    current_y -= input_y_step * (max(input_module_group_widths[group_idx]) + 1)
                    group_idx += 1

                # Continuous Inputs
                for i, continuous_module in enumerate(self.input_module.continuous_modules):
                    idx, min_value, max_value, has_missing_values = self.input_module.continuous_index_min_max_has_missing_values_tuples[i]
                    current_x = 0.5 * x_step
                    loc_current_y = current_y - input_y_step * (max(input_module_group_widths[group_idx]) - 1) / 2
                    text_fixed_size(
                        ax,
                        current_x,
                        loc_current_y + y_text_line / 2,
                        self.feature_names[idx],
                    )
                    text_fixed_size(
                        ax,
                        current_x,
                        loc_current_y - y_text_line / 2,
                        "[" + str(min_value) + ", " + str(max_value) + "]",
                    )
                    if is_single_rule:
                        axins3 = inset_axes(
                            ax,
                            width="100%",
                            height="100%",
                            bbox_to_anchor=(
                                (current_x + 0.5 * x_step + x_node_buffer) / width,
                                (loc_current_y - input_y_step * (max(input_module_group_widths[group_idx]) - 1) / 2) / height,
                                (2 * x_step) / width,
                                (y_step * (max(input_module_group_widths[group_idx]) - 1) + x_node_buffer) / height,
                            ),
                            bbox_transform=ax.transAxes,
                        )
                        plot_x_coordinates = [min_value + point_idx * (max_value - min_value) / 255 for point_idx in range(256)]
                        torch_plot_x_coordinates = torch.tensor(plot_x_coordinates, device=self.device).view(-1, 1)

                        continuous_copy = copy.deepcopy(self)
                        (continuous_module_first_idx, continuous_module_stop_idx) = continuous_copy.layers[-1].in_concepts_group_first_stop_pairs[
                            len(self.input_module.category_modules) + i
                        ]
                        continuous_copy.layers[-1].observed_concepts.data[:, :continuous_module_first_idx] = 0
                        continuous_copy.layers[-1].observed_concepts.data[:, continuous_module_stop_idx:] = 0
                        continuous_copy.layers[-1].unobserved_concepts.data[:] = 1
                        continuous_copy.simplify()
                        continuous_copy.input_module.continuous_index_min_max_has_missing_values_tuples = [(0, min_value, max_value, has_missing_values)]
                        continuous_copy.eval()
                        torch_plot_y_coordinates = continuous_copy.forward(torch_plot_x_coordinates)
                        plot_y_coordinates = torch_plot_y_coordinates.view(-1).cpu().tolist()

                        axins3.plot(
                            plot_x_coordinates,
                            plot_y_coordinates,
                        )
                        axins3.set_xlim([min_value, max_value])
                        boundaries = continuous_module.dichotomies.boundaries.data.view(-1).tolist()
                        boundary_idcs = list(range(len(boundaries)))
                        boundary_rel_x_positions = ((continuous_module.dichotomies.boundaries.data.view(-1) - min_value) / (max_value - min_value)).tolist()
                        boundary_abs_derivatives = []
                        for boundary in boundaries:
                            x_idx = 0
                            while plot_x_coordinates[x_idx] < boundary and x_idx < len(plot_x_coordinates) - 1:
                                x_idx += 1
                            boundary_abs_derivatives.append(abs(plot_y_coordinates[x_idx] - plot_y_coordinates[x_idx - 1]))
                        chosen_boundary_idcs = []
                        while len(boundary_idcs) > 0:
                            remaining_max_abs_deriv_idx = boundary_abs_derivatives.index(max(boundary_abs_derivatives))
                            is_too_close = False
                            for chosen_boundary_idx in chosen_boundary_idcs:
                                if (
                                    abs(boundary_rel_x_positions[chosen_boundary_idx] - boundary_rel_x_positions[boundary_idcs[remaining_max_abs_deriv_idx]])
                                    < boundary_rel_x_min_spacing
                                ):
                                    is_too_close = True
                                    break
                            if not is_too_close:
                                chosen_boundary_idcs.append(boundary_idcs[remaining_max_abs_deriv_idx])
                            boundary_idcs.remove(boundary_idcs[remaining_max_abs_deriv_idx])
                            boundary_abs_derivatives.remove(boundary_abs_derivatives[remaining_max_abs_deriv_idx])
                        chosen_boundary_idcs.sort()
                        # print([boundary_rel_x_positions[chosen_boundary_idx] for chosen_boundary_idx in chosen_boundary_idcs])
                        axins3.set_xticks([boundaries[chosen_boundary_idx] for chosen_boundary_idx in chosen_boundary_idcs])
                        axins3.set_ylim([-0.01, 1.02])
                        axins3.set_yticks([0, 1])
                        current_x += 2.5 * x_step
                        coordinates[-1].append((current_x, loc_current_y))
                        current_y -= input_y_step * 6
                    else:
                        plot_x_coordinates = [min_value + point_idx * (max_value - min_value) / 255 for point_idx in range(256)]
                        torch_plot_x_coordinates = torch.tensor(plot_x_coordinates, device=self.device).view(-1)
                        continuous_module_copy = copy.deepcopy(continuous_module)
                        torch_plot_y_coordinates = continuous_module_copy.forward(torch_plot_x_coordinates)

                        loc_current_y = current_y - 2 * input_y_step
                        loc_coordinates = []
                        for out_concept in range(continuous_module.out_concepts.nb_out_concepts):
                            if out_concept == first_out_concept_with_same_graphs_list[i][out_concept]:
                                plot_y_coordinates = torch_plot_y_coordinates[:, out_concept].view(-1).cpu().tolist()

                                axins3 = inset_axes(
                                    ax,
                                    width="100%",
                                    height="100%",
                                    bbox_to_anchor=(
                                        (current_x + 0.5 * x_step + x_node_buffer) / width,
                                        (loc_current_y - input_y_step * 2) / height,
                                        (2 * x_step) / width,
                                        (y_step * 4 + x_node_buffer) / height,
                                    ),
                                    bbox_transform=ax.transAxes,
                                )
                                axins3.plot(plot_x_coordinates, plot_y_coordinates)
                                axins3.set_xlim([min_value, max_value])
                                boundaries = continuous_module.dichotomies.boundaries.data.view(-1).tolist()
                                boundary_idcs = list(range(len(boundaries)))
                                boundary_rel_x_positions = ((continuous_module.dichotomies.boundaries.data.view(-1) - min_value) / (max_value - min_value)).tolist()
                                boundary_abs_derivatives = []
                                for boundary in boundaries:
                                    x_idx = 0
                                    while plot_x_coordinates[x_idx] < boundary and x_idx < len(plot_x_coordinates) - 1:
                                        x_idx += 1
                                    boundary_abs_derivatives.append(abs(plot_y_coordinates[x_idx] - plot_y_coordinates[x_idx - 1]))
                                chosen_boundary_idcs = []
                                while len(boundary_idcs) > 0:
                                    remaining_max_abs_deriv_idx = boundary_abs_derivatives.index(max(boundary_abs_derivatives))
                                    is_too_close = False
                                    for chosen_boundary_idx in chosen_boundary_idcs:
                                        if (
                                            abs(boundary_rel_x_positions[chosen_boundary_idx] - boundary_rel_x_positions[boundary_idcs[remaining_max_abs_deriv_idx]])
                                            < boundary_rel_x_min_spacing
                                        ):
                                            is_too_close = True
                                            break
                                    if not is_too_close:
                                        chosen_boundary_idcs.append(boundary_idcs[remaining_max_abs_deriv_idx])
                                    boundary_idcs.remove(boundary_idcs[remaining_max_abs_deriv_idx])
                                    boundary_abs_derivatives.remove(boundary_abs_derivatives[remaining_max_abs_deriv_idx])
                                chosen_boundary_idcs.sort()
                                # print([boundary_rel_x_positions[chosen_boundary_idx] for chosen_boundary_idx in chosen_boundary_idcs])
                                axins3.set_xticks([boundaries[chosen_boundary_idx] for chosen_boundary_idx in chosen_boundary_idcs])
                                axins3.set_ylim([-0.01, 1.02])
                                axins3.set_yticks([0, 1])
                                coordinates[-1].append((current_x + 2.5 * x_step, loc_current_y))
                                loc_coordinates.append((current_x + 2.5 * x_step, loc_current_y))

                                loc_current_y -= input_y_step * 6
                                current_y -= input_y_step * 6
                            else:
                                first_out_concept_with_same_graph = first_out_concept_with_same_graphs_list[i][out_concept]
                                coordinates[-1].append((loc_coordinates[first_out_concept_with_same_graph][0], loc_coordinates[first_out_concept_with_same_graph][1]))
                                loc_coordinates.append((loc_coordinates[first_out_concept_with_same_graph][0], loc_coordinates[first_out_concept_with_same_graph][1]))
                        current_y -= input_y_step
                        current_x += 2.5 * x_step
                        # group_coordinates = [[]]
                        # current_x += x_step
                        # if input_module_group_widths[group_idx][0] == max(input_module_group_widths[group_idx]):
                        #     loc_current_y = current_y
                        # else:
                        #     loc_current_y = current_y - input_y_step * ((max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][0]) / 2
                        # for dichotomy in range(continuous_module.dichotomies.nb_dichotomies):
                        #     dichotomy_string = "approx. > " if continuous_module.dichotomies.sharpnesses.data[dichotomy].item() < 2 else "exactly > "
                        #     dichotomy_string += str(
                        #         round(
                        #             continuous_module.dichotomies.boundaries.data[dichotomy].item(),
                        #             1,
                        #         )
                        #     )
                        #     text_fixed_size(ax, current_x, loc_current_y, dichotomy_string)
                        #     group_coordinates[-1].append((current_x + 0.5 * x_step, loc_current_y))
                        #     if input_module_group_widths[group_idx][0] == max(input_module_group_widths[group_idx]):
                        #         loc_current_y -= input_y_step
                        #     else:
                        #         loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][0]
                        # group_coordinates.append([])
                        # current_x += x_step
                        # if input_module_group_widths[group_idx][1] == max(input_module_group_widths[group_idx]):
                        #     loc_current_y = current_y
                        # else:
                        #     loc_current_y = current_y - input_y_step * ((max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][1]) / 2
                        # for interval in range(continuous_module.intervals.nb_out_concepts):
                        #     AND_node(ax, current_x, loc_current_y)
                        #     group_coordinates[-1].append((current_x, loc_current_y))
                        #     if input_module_group_widths[group_idx][1] == max(input_module_group_widths[group_idx]):
                        #         loc_current_y -= input_y_step
                        #     else:
                        #         loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][1]
                        # for in_concept in range(continuous_module.intervals.nb_in_concepts):
                        #     for out_concept in range(continuous_module.intervals.nb_out_concepts):
                        #         weight = continuous_module.intervals.observed_concepts.data[out_concept, in_concept].item()
                        #         if weight < 0:
                        #             arrow(
                        #                 ax,
                        #                 group_coordinates[-2][in_concept][0],
                        #                 group_coordinates[-2][in_concept][1],
                        #                 group_coordinates[-1][out_concept][0] - x_node_buffer,
                        #                 group_coordinates[-1][out_concept][1],
                        #                 color="r",
                        #                 alpha=abs(weight),
                        #             )
                        # for in_concept in range(continuous_module.intervals.nb_in_concepts):
                        #     for out_concept in range(continuous_module.intervals.nb_out_concepts):
                        #         weight = continuous_module.intervals.observed_concepts.data[out_concept, in_concept].item()
                        #         if weight > 0:
                        #             arrow(
                        #                 ax,
                        #                 group_coordinates[-2][in_concept][0],
                        #                 group_coordinates[-2][in_concept][1],
                        #                 group_coordinates[-1][out_concept][0] - x_node_buffer,
                        #                 group_coordinates[-1][out_concept][1],
                        #                 color="k",
                        #                 alpha=weight,
                        #             )
                        # group_coordinates.append([])
                        # current_x += 0.5 * x_step
                        # if input_module_group_widths[group_idx][2] == max(input_module_group_widths[group_idx]):
                        #     loc_current_y = current_y
                        # else:
                        #     loc_current_y = current_y - input_y_step * ((max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][2]) / 2
                        # for out_concept in range(continuous_module.out_concepts.nb_out_concepts):
                        #     OR_node(ax, current_x, loc_current_y)
                        #     group_coordinates[-1].append((current_x, loc_current_y))
                        #     coordinates[-1].append((current_x, loc_current_y))
                        #     if input_module_group_widths[group_idx][2] == max(input_module_group_widths[group_idx]):
                        #         loc_current_y -= input_y_step
                        #     else:
                        #         loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][2]
                        # for in_concept in range(continuous_module.out_concepts.nb_in_concepts):
                        #     for out_concept in range(continuous_module.out_concepts.nb_out_concepts):
                        #         weight = continuous_module.out_concepts.observed_concepts.data[out_concept, in_concept].item()
                        #         if weight > 0:
                        #             arrow(
                        #                 ax,
                        #                 group_coordinates[-2][in_concept][0] + x_node_buffer,
                        #                 group_coordinates[-2][in_concept][1],
                        #                 group_coordinates[-1][out_concept][0] - x_node_buffer,
                        #                 group_coordinates[-1][out_concept][1],
                        #                 color="k",
                        #                 alpha=weight,
                        #             )
                    group_idx += 1

                # Periodic Inputs
                for idx, period, has_missing_values in self.input_module.periodic_index_period_has_missing_values_tuples:
                    current_x = 0.5 * x_step
                    loc_current_y = current_y - input_y_step * (max(input_module_group_widths[group_idx]) - 1) / 2
                    text_fixed_size(ax, current_x, loc_current_y, self.feature_names[idx])
                    raise Exception("Case with periodic inputs not coded yet.")
                    # TODO do dichotomies
                    # TODO do intervals
                    # TODO do collections
                    current_y -= y_step * (max(input_module_group_widths[group_idx]) + 1)
                    group_idx += 1

                # Fully-Connected Layers
                layer_width_idx = 1
                for i, layer in enumerate(self.layers):
                    coordinates.append([])
                    current_x += x_step
                    if layer_widths_with_spacing[layer_width_idx] == max_layer_width_with_spacing:
                        current_y = height - y_heading - 0.5 * y_step
                    else:
                        current_y = height - y_heading - 0.5 * y_step - y_step * ((max_layer_width_with_spacing - 1) / layer_widths_with_spacing[layer_width_idx]) / 2
                    for out_concept in range(layer.nb_out_concepts):
                        if layer.is_AND:
                            if layer.use_unobserved:
                                AND_node(
                                    ax,
                                    current_x,
                                    current_y,
                                    unobserved_concept=layer.unobserved_concepts.data[out_concept].item(),
                                )
                            else:
                                AND_node(ax, current_x, current_y)
                        else:
                            if layer.use_unobserved:
                                OR_node(
                                    ax,
                                    current_x,
                                    current_y,
                                    unobserved_concept=layer.unobserved_concepts.data[out_concept].item(),
                                )
                            else:
                                OR_node(ax, current_x, current_y)
                        coordinates[-1].append((current_x, current_y))
                        if layer_widths_with_spacing[layer_width_idx] == max_layer_width_with_spacing:
                            current_y -= y_step
                        else:
                            current_y -= y_step * (max_layer_width_with_spacing - 1) / layer_widths_with_spacing[layer_width_idx]
                    in_has_indir = i > 0 and self.layers[i - 1].use_unobserved
                    if is_single_rule:
                        max_bin_cat_idx = len(self.input_module.binary_indices) + sum([category_module.nb_out_concepts for category_module in self.input_module.category_modules])
                    for in_concept in range(layer.nb_in_concepts):
                        if not is_single_rule or in_concept < max_bin_cat_idx:
                            for out_concept in range(layer.nb_out_concepts):
                                weight = layer.get_weight_value(out_concept, in_concept)
                                if weight < 0:
                                    if in_has_indir:
                                        arrow(
                                            ax,
                                            coordinates[-2][in_concept][0] + x_node_buffer + x_indir_buffer,
                                            coordinates[-2][in_concept][1],
                                            coordinates[-1][out_concept][0] - x_node_buffer,
                                            coordinates[-1][out_concept][1],
                                            color="r",
                                            alpha=abs(weight),
                                        )
                                    else:
                                        arrow(
                                            ax,
                                            coordinates[-2][in_concept][0] + x_node_buffer,
                                            coordinates[-2][in_concept][1],
                                            coordinates[-1][out_concept][0] - x_node_buffer,
                                            coordinates[-1][out_concept][1],
                                            color="r",
                                            alpha=abs(weight),
                                        )
                    for in_concept in range(layer.nb_in_concepts):
                        if not is_single_rule or in_concept < max_bin_cat_idx:
                            for out_concept in range(layer.nb_out_concepts):
                                weight = layer.get_weight_value(out_concept, in_concept)
                                if weight > 0:
                                    if in_has_indir:
                                        arrow(
                                            ax,
                                            coordinates[-2][in_concept][0] + x_node_buffer + x_indir_buffer,
                                            coordinates[-2][in_concept][1],
                                            coordinates[-1][out_concept][0] - x_node_buffer,
                                            coordinates[-1][out_concept][1],
                                            color="k",
                                            alpha=weight,
                                        )
                                    else:
                                        arrow(
                                            ax,
                                            coordinates[-2][in_concept][0] + x_node_buffer,
                                            coordinates[-2][in_concept][1],
                                            coordinates[-1][out_concept][0] - x_node_buffer,
                                            coordinates[-1][out_concept][1],
                                            color="k",
                                            alpha=weight,
                                        )
                    if is_single_rule:
                        in_concept = max_bin_cat_idx
                        for continuous_module in self.input_module.continuous_modules:
                            arrow(
                                ax,
                                coordinates[-2][in_concept][0] + x_node_buffer,
                                coordinates[-2][in_concept][1],
                                coordinates[-1][out_concept][0] - x_node_buffer,
                                coordinates[-1][out_concept][1],
                                color="k",
                                alpha=1,
                            )
                            in_concept += 1
                        for periodic_module in self.input_module.periodic_modules:
                            arrow(
                                ax,
                                coordinates[-2][in_concept][0] + x_node_buffer,
                                coordinates[-2][in_concept][1],
                                coordinates[-1][out_concept][0] - x_node_buffer,
                                coordinates[-1][out_concept][1],
                                color="k",
                                alpha=1,
                            )
                            in_concept += 1
                    layer_width_idx += 1

                # Target Outputs
                current_x += 0.5 * x_step + x_node_buffer
                if self.layers[-1].use_unobserved:
                    current_x += x_indir_buffer
                if layer_widths_with_spacing[-1] == max_layer_width_with_spacing:
                    current_y = height - y_heading - 0.5 * y_step
                else:
                    current_y = height - y_heading - 0.5 * y_step - y_step * ((max_layer_width_with_spacing - 1) / layer_widths_with_spacing[-1]) / 2
                for idx in output_target_idcs:
                    if is_single_rule:
                        arrow(
                            ax,
                            coordinates[-1][0][0] + x_node_buffer + x_indir_buffer,
                            coordinates[-1][0][1],
                            current_x - 0.25 * x_step,
                            current_y,
                            color="k",
                            alpha=1,
                            use_arrow_head=True,
                        )
                        current_x += 0.25 * x_step
                    text_fixed_size(ax, current_x, current_y, self.feature_names[idx])
                    if layer_widths_with_spacing[-1] == max_layer_width_with_spacing:
                        current_y -= y_step
                    else:
                        current_y -= y_step * (max_layer_width_with_spacing - 1) / layer_widths_with_spacing[-1]
            else:

                # Find maximum layer width
                max_layer_width_with_spacing = self.layers[-1].nb_out_concepts

                # Find output indices
                output_target_idcs = list(range(len(self.feature_names)))
                for idx in self.binary_indices:
                    output_target_idcs.remove(idx)
                for first_idx, last_idx, has_missing_values in self.category_first_last_has_missing_values_tuples:
                    for idx in range(first_idx, last_idx + 1):
                        output_target_idcs.remove(idx)
                for idx, min_value, max_value, has_missing_values in self.continuous_index_min_max_has_missing_values_tuples:
                    output_target_idcs.remove(idx)
                for idx, period, has_missing_values in self.periodic_index_period_has_missing_values_tuples:
                    output_target_idcs.remove(idx)

                # Define figure
                if valid_raw_loss >= 0:
                    y_heading = 1.5 * y_step
                else:
                    y_heading = 0
                height = y_step * max_layer_width_with_spacing + y_heading + 1.05 * x_node_buffer
                width = x_step * (0.5 + 0.5 + 0.5) + x_node_buffer
                width += 0.25 * x_step
                if self.layers[-1].use_unobserved:
                    width += x_indir_buffer
                if filename != "":
                    plt.ion()
                name = filename if filename != "" else "GNLN"
                if fig_ax == -1:
                    fig, ax = plt.subplots(
                        1,
                        1,
                        num=name,
                        gridspec_kw={"left": 0.0, "right": 1.0, "bottom": 0.0, "top": 1.0},
                    )
                else:
                    fig, ax = fig_ax
                fig.set_frameon(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                dpi = fig.get_dpi()
                fig.set_size_inches(width / dpi, height / dpi)
                ax.set_ylim(0, height)
                ax.set_xlim(0, width)
                ax.set_aspect("equal")

                # Losses
                current_y = height - 0.5 * y_step
                if valid_raw_loss >= 0:
                    current_x = width * (2.2 / 5)
                    text_fixed_size(
                        ax,
                        current_x,
                        current_y,
                        f"{valid_raw_loss:.3e}",
                        h=12,
                        align_left=True,
                    )

                current_x = 0.5 * x_step

                # Fully-Connected Layers
                i, layer = -1, self.layers[-1]

                current_y = height - y_heading - 0.5 * y_step
                for out_concept in range(layer.nb_out_concepts):
                    if layer.is_AND:
                        if layer.use_unobserved:
                            AND_node(
                                ax,
                                current_x,
                                current_y,
                                unobserved_concept=layer.unobserved_concepts.data[out_concept].item(),
                            )
                        else:
                            AND_node(ax, current_x, current_y)
                    else:
                        if layer.use_unobserved:
                            OR_node(
                                ax,
                                current_x,
                                current_y,
                                unobserved_concept=layer.unobserved_concepts.data[out_concept].item(),
                            )
                        else:
                            OR_node(ax, current_x, current_y)
                    current_y -= y_step

                # Target Outputs
                current_x += 0.5 * x_step + x_node_buffer
                if self.layers[-1].use_unobserved:
                    current_x += x_indir_buffer
                current_y = height - y_heading - 0.5 * y_step
                for idx in output_target_idcs:
                    text_fixed_size(ax, current_x, current_y, self.feature_names[idx])
                    current_y -= y_step

            if fig_ax == -1:
                if filename == "":
                    plt.show()
                else:
                    plt.pause(1e-8)
                    plt.ioff()
                    plt.savefig(filename + ".png", dpi=save_dpi)
                    plt.close("all")

    def _get_target_indices(self):
        output_target_idcs = list(range(len(self.feature_names)))
        for idx in self.binary_indices:
            output_target_idcs.remove(idx)
        for first_idx, last_idx, has_missing_values in self.category_first_last_has_missing_values_tuples:
            for idx in range(first_idx, last_idx + 1):
                output_target_idcs.remove(idx)
        for idx, min_value, max_value, has_missing_values in self.continuous_index_min_max_has_missing_values_tuples:
            output_target_idcs.remove(idx)
        for idx, period, has_missing_values in self.periodic_index_period_has_missing_values_tuples:
            output_target_idcs.remove(idx)
        return output_target_idcs

    def get_target_feature_names(self):
        output_target_idcs = self._get_target_indices()
        return [self.feature_names[output_target_idx] for output_target_idx in output_target_idcs]

    def load_string(self, init_string):
        lines = init_string.split("\n")
        input_module_init_string = ""
        layers_strings = []
        in_continuous_periodic_counter = 0
        for i, line in enumerate(lines):
            if in_continuous_periodic_counter > 0:
                input_module_init_string += line + "\n"
                in_continuous_periodic_counter -= 1
            else:
                if line[:6] == "Binary" or line[:8] == "Category":
                    input_module_init_string += line + "\n"
                elif line[:10] == "Continuous" or line[:8] == "Periodic":
                    input_module_init_string += line + "\n"
                    in_continuous_periodic_counter = 2
                else:
                    if line[0] == "[":
                        if line[-2:] == ", ":
                            layers_strings.append(line[1:-2])
                        elif line[-1:] == ",":
                            layers_strings.append(line[1:-1])
                        elif line[-1] == "]":
                            layers_strings.append(line[1:-1])
                        else:
                            raise Exception("LNN input string is not formatted correctly.\n Error at line " + str(i) + " --> " + str(line))
                    else:
                        if line[-2:] == ", ":
                            layers_strings.append(line[:-2])
                        elif line[-1:] == ",":
                            layers_strings.append(line[:-1])
                        elif line[-1] == "]":
                            layers_strings.append(line[:-1])
                        else:
                            raise Exception("LNN input string is not formatted correctly.\n Error at line " + str(i) + " --> " + str(line))
        self.input_module.load_string(input_module_init_string)
        for i, layer in enumerate(self.layers):
            layer.load_string(layers_strings[i])
        self.layers[0].in_concepts_group_first_stop_pairs = self.input_module.get_out_concepts_group_first_stop_pairs()
        if self.layers[0].is_grouped:
            self.layers[0].nb_ungrouped_in_concepts = self.layers[0].in_concepts_group_first_stop_pairs[0][0]

    def simplify(self, can_retrain=False):
        self._backward_simplify()
        self._forward_simplify()
        must_retrain = self.remove_duplicates(can_retrain)
        self.layers[0].in_concepts_group_first_stop_pairs = self.input_module.get_out_concepts_group_first_stop_pairs()
        if can_retrain:
            if must_retrain:
                self = NeuralLogicNetwork(
                    nb_in_features=self.nb_in_features,
                    nb_out_concepts=self.nb_out_concepts,
                    nb_hidden_layers=len(self.layers) - 1,
                    nb_concepts_per_hidden_layer=self.nb_concepts_per_hidden_layer,
                    last_layer_is_OR_no_neg=(not self.layers[-1].is_AND and not self.layers[-1].use_negation),
                    category_first_last_has_missing_values_tuples=self.category_first_last_has_missing_values_tuples,
                    continuous_index_min_max_has_missing_values_tuples=self.continuous_index_min_max_has_missing_values_tuples,
                    periodic_index_period_has_missing_values_tuples=self.periodic_index_period_has_missing_values_tuples,
                    device=self.device,
                    init_string=str(self),
                )
            return must_retrain

    def _backward_simplify(self):
        used_out_concepts = [-1]
        for layer in reversed(self.layers):
            used_out_concepts = layer.backward_simplify(used_out_concepts)
        self.input_module.backward_simplify(used_out_concepts)

    def _forward_simplify(self):
        used_out_concepts, unused_out_idx_prob_pairs = self.input_module.forward_simplify()
        in_concepts_group_first_stop_pairs = self.input_module.get_out_concepts_group_first_stop_pairs()
        for i, layer in enumerate(self.layers[:-1]):
            if i == 0:
                used_out_concepts, unused_out_idx_prob_pairs = layer.forward_simplify(
                    used_in_concepts=used_out_concepts,
                    unused_in_idx_prob_pairs=unused_out_idx_prob_pairs,
                    in_concepts_group_first_stop_pairs=in_concepts_group_first_stop_pairs,
                )
            else:
                used_out_concepts, unused_out_idx_prob_pairs = layer.forward_simplify(
                    used_in_concepts=used_out_concepts,
                    unused_in_idx_prob_pairs=unused_out_idx_prob_pairs,
                )
        self.layers[-1].forward_simplify(
            used_in_concepts=used_out_concepts,
            unused_in_idx_prob_pairs=unused_out_idx_prob_pairs,
            keep_all_out_concepts=True,
        )

    def remove_duplicates(self, can_retrain):
        if not can_retrain:
            duplicate_lists = self.input_module.remove_duplicates(can_retrain)
            self.layers[0].remove_duplicates(in_duplicate_lists=duplicate_lists)
        if can_retrain:
            duplicate_lists, must_retrain = self.input_module.remove_duplicates(can_retrain)
            if len(self.layers) > 1:
                tentative_duplicate_lists, _ = self.layers[0].remove_duplicates(in_duplicate_lists=duplicate_lists, in_tentative_duplicate_lists=[])
                for j, layer in enumerate(self.layers[1:-1]):
                    i = j + 1
                    tentative_duplicate_lists, confirmed_duplicate_lists = layer.remove_duplicates(in_tentative_duplicate_lists=tentative_duplicate_lists)
                    if len(confirmed_duplicate_lists) > 0:
                        self.layers[i - 1].remove_out_confirmed_duplicates(confirmed_duplicate_lists)
                        must_retrain = True
                confirmed_duplicate_lists = self.layers[-1].remove_duplicates(
                    in_tentative_duplicate_lists=tentative_duplicate_lists,
                    keep_all_out_concepts=True,
                )
                if len(confirmed_duplicate_lists) > 0:
                    self.layers[-2].remove_out_confirmed_duplicates(confirmed_duplicate_lists)
                    must_retrain = True
            else:
                self.layers[-1].remove_duplicates(in_duplicate_lists=duplicate_lists, keep_all_out_concepts=True)
            return must_retrain

    def duplicate_rule_with_new_biases_and_uses(self, old_rule_idx: int, new_biases_uses: List[Tuple[float, List[int]]]):
        if self.layers[-2].use_unobserved:
            self.layers[-2].duplicate_out_concept_with_new_biases(old_rule_idx, [new_bias for new_bias, new_uses in new_biases_uses])
            self.layers[-1].duplicate_in_concept_with_new_uses(old_rule_idx, [new_uses for new_bias, new_uses in new_biases_uses])
            self.simplify()

    def create_simplifed_copy(self):
        simplified_model = copy.deepcopy(self)
        simplified_model.simplify()
        return simplified_model

    def prune(
        self, eval_model_func, save_model_func, init_loss=-1, filename="", do_log=True, prune_log_files=[], progress_bar_hook=lambda iteration, nb_iterations: None, init_weight=0
    ):
        if filename != "" and prune_log_files == []:
            prune_log_files = get_log_files(filename, do_log)
        print_log("Pruning model...", self.verbose, prune_log_files)
        did_prune = False
        if init_loss < 0:
            init_loss = eval_model_func()
        total_nb_weights = self._get_nb_weights()
        current_weight = 0
        for i, layer in enumerate(reversed(self.layers)):
            init_loss, local_did_prune = layer.prune(
                eval_model_func,
                init_loss=init_loss,
                log_files=prune_log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(init_weight + current_weight + loc_weight, init_weight + total_nb_weights + 1),
            )
            did_prune = did_prune or local_did_prune
            self.simplify()
            total_nb_weights = self._get_nb_weights()
            current_weight = sum([layer.get_nb_weights() for j, layer in enumerate(reversed(self.layers)) if j <= i])
        init_loss, local_did_prune = self.input_module.prune(
            eval_model_func,
            init_loss=init_loss,
            log_files=prune_log_files,
            progress_bar_hook=lambda loc_weight: progress_bar_hook(init_weight + current_weight + loc_weight, init_weight + total_nb_weights + 1),
        )
        did_prune = did_prune or local_did_prune
        self.simplify()
        total_nb_weights = self._get_nb_weights()
        if did_prune:
            save_model_func()
            init_loss = self.prune(
                eval_model_func, save_model_func, init_loss=init_loss, prune_log_files=prune_log_files, progress_bar_hook=progress_bar_hook, init_weight=total_nb_weights
            )
        else:
            progress_bar_hook(1, 1)
            if filename != "":
                print_log("\n" + str(self), False, prune_log_files)
                close_log_files(prune_log_files)
        return init_loss

    def _get_nb_weights(self):
        nb_weights = self.input_module.get_nb_weights()
        for layer in self.layers:
            nb_weights += layer.get_nb_weights()
        return nb_weights

    def discretize(
        self, discretization_method, eval_model_func=lambda: None, save_model_func=lambda: None, filename="", do_log=True, progress_bar_hook=lambda iteration, nb_iterations: None
    ):
        if discretization_method == "thresh":
            self.input_module.discretize("thresh")
            for layer in self.layers:
                layer.discretize("thresh")
            self.simplify()
            return []
        else:
            discretize_log_files = get_log_files(filename, do_log)
            losses = [eval_model_func()]
            total_nb_weights = self._get_nb_weights()
            current_weight = 0
            for i, layer in enumerate(reversed(self.layers)):
                losses += layer.discretize(
                    discretization_method,
                    eval_model_func=eval_model_func,
                    log_files=discretize_log_files,
                    progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight, total_nb_weights + 1),
                )
                self.simplify()
                save_model_func()
                total_nb_weights = self._get_nb_weights()
                current_weight = sum([layer.get_nb_weights() for j, layer in enumerate(reversed(self.layers)) if j <= i])
            losses += self.input_module.discretize(
                discretization_method,
                eval_model_func=eval_model_func,
                save_model_func=save_model_func,
                log_files=discretize_log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight, total_nb_weights + 1),
            )
            self.simplify()
            total_nb_weights = max(1, self._get_nb_weights())
            progress_bar_hook(total_nb_weights, total_nb_weights)
            print_log("\n" + str(self), False, discretize_log_files)
            close_log_files(discretize_log_files)
            return losses

    def rearrange_for_visibility(self):
        self._simplify_category_modules()
        self.reorder_lexicographically_and_by_unobserved()

    def _simplify_category_modules(self):
        for category_module_idx, category_module in enumerate(self.input_module.category_modules):
            new_category_OR_nodes_used_out_concepts = []
            for layer0_AND_concept_idx in range(self.layers[0].nb_out_concepts):
                used_positively_in_concepts = (
                    torch.nonzero(
                        self.layers[0].observed_concepts.data[
                            layer0_AND_concept_idx,
                            self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][0] : self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][1],
                        ]
                        > 0
                    )
                    .view(-1)
                    .tolist()
                )
                used_negatively_in_concepts = (
                    torch.nonzero(
                        self.layers[0].observed_concepts.data[
                            layer0_AND_concept_idx,
                            self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][0] : self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][1],
                        ]
                        < 0
                    )
                    .view(-1)
                    .tolist()
                )
                if len(used_positively_in_concepts) + len(used_negatively_in_concepts) > 0:
                    is_duplicate = False
                    for (
                        other_applicable_layer0_AND_concepts,
                        other_used_positively_in_concepts,
                        other_used_negatively_in_concepts,
                    ) in new_category_OR_nodes_used_out_concepts:
                        if used_positively_in_concepts == other_used_positively_in_concepts and used_negatively_in_concepts == other_used_negatively_in_concepts:
                            other_applicable_layer0_AND_concepts.append(layer0_AND_concept_idx)
                            is_duplicate = True
                    if not is_duplicate:
                        applicable_layer0_AND_concepts = [layer0_AND_concept_idx]
                        new_category_OR_nodes_used_out_concepts.append(([layer0_AND_concept_idx], used_positively_in_concepts, used_negatively_in_concepts))
            new_category_module_observed_concepts = torch.zeros(
                (len(new_category_OR_nodes_used_out_concepts), category_module.nb_in_concepts),
                device=category_module.device,
            )
            if category_module.use_missing_values:
                new_category_module_missing_observed_concepts = torch.zeros((len(new_category_OR_nodes_used_out_concepts)), device=category_module.device)
            new_layer0_AND_missing_columns = torch.zeros(
                (self.layers[0].nb_out_concepts, len(new_category_OR_nodes_used_out_concepts)), device=self.layers[0].observed_concepts.data.device
            )
            for (
                new_category_OR_node_idx,
                new_category_OR_node_used_out_concepts,
            ) in enumerate(new_category_OR_nodes_used_out_concepts):
                (
                    applicable_layer0_AND_concepts,
                    used_positively_in_concepts,
                    used_negatively_in_concepts,
                ) = new_category_OR_node_used_out_concepts
                new_category_OR_node_in_concepts = set(range(category_module.nb_in_concepts))
                if category_module.use_missing_values:
                    new_category_OR_node_missing_observed_concept = 1
                for used_positively_in_concept in used_positively_in_concepts:
                    new_category_OR_node_in_concepts = new_category_OR_node_in_concepts & set(
                        torch.nonzero(category_module.observed_concepts.data[used_positively_in_concept, :]).view(-1).tolist()
                    )
                    if category_module.use_missing_values:
                        new_category_OR_node_missing_observed_concept *= category_module.missing_observed_concepts.data[used_positively_in_concept]
                for used_negatively_in_concept in used_negatively_in_concepts:
                    new_category_OR_node_in_concepts = new_category_OR_node_in_concepts - set(
                        torch.nonzero(category_module.observed_concepts.data[used_negatively_in_concept, :]).view(-1).tolist()
                    )
                    if category_module.use_missing_values:
                        new_category_OR_node_missing_observed_concept *= 1 - category_module.missing_observed_concepts.data[used_negatively_in_concept]
                new_category_OR_node_is_used_positively = len(new_category_OR_node_in_concepts) <= category_module.nb_in_concepts - len(new_category_OR_node_in_concepts)
                if new_category_OR_node_is_used_positively:
                    for new_category_OR_node_in_concept in new_category_OR_node_in_concepts:
                        new_category_module_observed_concepts[new_category_OR_node_idx, new_category_OR_node_in_concept] = 1
                    if category_module.use_missing_values:
                        new_category_module_missing_observed_concepts[new_category_OR_node_idx] = new_category_OR_node_missing_observed_concept
                else:
                    for new_category_OR_node_in_concept in set(range(category_module.nb_in_concepts)) - new_category_OR_node_in_concepts:
                        new_category_module_observed_concepts[new_category_OR_node_idx, new_category_OR_node_in_concept] = 1
                    if category_module.use_missing_values:
                        new_category_module_missing_observed_concepts[new_category_OR_node_idx] = 1 - new_category_OR_node_missing_observed_concept
                duplicate_category_OR_node_idx = None
                for possible_duplicate_category_OR_node_idx in range(new_category_OR_node_idx):
                    if (
                        new_category_module_observed_concepts[new_category_OR_node_idx, :] == new_category_module_observed_concepts[possible_duplicate_category_OR_node_idx, :]
                    ).all() and (
                        not category_module.use_missing_values
                        or (
                            new_category_module_missing_observed_concepts[new_category_OR_node_idx]
                            == new_category_module_missing_observed_concepts[possible_duplicate_category_OR_node_idx]
                        ).all()
                    ):
                        duplicate_category_OR_node_idx = possible_duplicate_category_OR_node_idx
                        duplicate_category_sign_matches = 1
                        break
                    elif (
                        new_category_module_observed_concepts[new_category_OR_node_idx, :] != new_category_module_observed_concepts[possible_duplicate_category_OR_node_idx, :]
                    ).all() and (
                        not category_module.use_missing_values
                        or (
                            new_category_module_missing_observed_concepts[new_category_OR_node_idx]
                            == 1 - new_category_module_missing_observed_concepts[possible_duplicate_category_OR_node_idx]
                        ).all()
                    ):
                        duplicate_category_OR_node_idx = possible_duplicate_category_OR_node_idx
                        duplicate_category_sign_matches = -1
                        break
                if duplicate_category_OR_node_idx == None:
                    for layer0_AND_concept_idx in applicable_layer0_AND_concepts:
                        new_layer0_AND_missing_columns[layer0_AND_concept_idx, new_category_OR_node_idx] = 1 if new_category_OR_node_is_used_positively else -1
                else:
                    new_category_module_observed_concepts[new_category_OR_node_idx, :] = 0
                    if category_module.use_missing_values:
                        new_category_module_missing_observed_concepts[new_category_OR_node_idx] = 0
                    for layer0_AND_concept_idx in applicable_layer0_AND_concepts:
                        new_layer0_AND_missing_columns[layer0_AND_concept_idx, duplicate_category_OR_node_idx] = (
                            duplicate_category_sign_matches if new_category_OR_node_is_used_positively else -1 * duplicate_category_sign_matches
                        )
            if self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][1] < self.layers[0].nb_in_concepts:
                self.layers[0].observed_concepts.data = torch.cat(
                    (
                        self.layers[0].observed_concepts.data[:, : self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][0]],
                        new_layer0_AND_missing_columns,
                        self.layers[0].observed_concepts.data[:, self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][1] :],
                    ),
                    dim=1,
                )
            else:
                self.layers[0].observed_concepts.data = torch.cat(
                    (
                        self.layers[0].observed_concepts.data[:, : self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][0]],
                        new_layer0_AND_missing_columns,
                    ),
                    dim=1,
                )
            category_module.observed_concepts.data = new_category_module_observed_concepts
            if category_module.use_missing_values:
                category_module.missing_observed_concepts.data = new_category_module_missing_observed_concepts
            category_module.nb_out_concepts = category_module.observed_concepts.data.size(0)
            self.layers[0].nb_in_concepts = self.layers[0].observed_concepts.data.size(1)
            self.layers[0].in_concepts_group_first_stop_pairs = self.input_module.get_out_concepts_group_first_stop_pairs()
        self.simplify()

    def reorder_lexicographically_and_by_unobserved(self, rule_order=None):
        if rule_order != None and not (
            isinstance(rule_order, list) and len(rule_order) == self.layers[-1].nb_in_concepts and set(rule_order) == set(list(range(self.layers[-1].nb_in_concepts)))
        ):
            raise Exception("Invalid rule order for NeuralLogicNetwork reordering.")
        reordered_input_module_out_idcs = self.input_module.reorder_lexicographically()
        self.layers[0].observed_concepts.data = self.layers[0].observed_concepts.data[:, reordered_input_module_out_idcs]
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < len(self.layers) - 1:
                if layer_idx < len(self.layers) - 2 or rule_order == None:
                    reordered_layer_out_idcs = find_lexicographical_order(layer.observed_concepts.data)
                else:
                    reordered_layer_out_idcs = rule_order
                layer.observed_concepts.data = layer.observed_concepts.data[reordered_layer_out_idcs, :]
                self.layers[layer_idx + 1].observed_concepts.data = self.layers[layer_idx + 1].observed_concepts.data[:, reordered_layer_out_idcs]
                if layer.use_unobserved:
                    layer.unobserved_concepts.data = layer.unobserved_concepts.data[reordered_layer_out_idcs]

                    if layer_idx < len(self.layers) - 2 or rule_order == None:
                        unobserved_out_idx_pairs = [(layer.unobserved_concepts.data[out_idx].item(), out_idx) for out_idx in range(layer.nb_out_concepts)]
                        unobserved_out_idx_pairs.sort(key=lambda pair: pair[0], reverse=True)
                        reordered_layer_out_idcs = [out_idx for unobserved, out_idx in unobserved_out_idx_pairs]
                        layer.unobserved_concepts.data = layer.unobserved_concepts.data[reordered_layer_out_idcs]
                        layer.observed_concepts.data = layer.observed_concepts.data[reordered_layer_out_idcs, :]
                        self.layers[layer_idx + 1].observed_concepts.data = self.layers[layer_idx + 1].observed_concepts.data[:, reordered_layer_out_idcs]
        reordered_last_layer_in_idcs = find_lexicographical_order(self.layers[-1].observed_concepts.data, order_outs=False)
        self.layers[-1].observed_concepts.data = self.layers[-1].observed_concepts.data[:, reordered_last_layer_in_idcs]
        self.layers[-2].observed_concepts.data = self.layers[-2].observed_concepts.data[reordered_last_layer_in_idcs, :]
        if self.layers[-2].use_unobserved:
            self.layers[-2].unobserved_concepts.data = self.layers[-2].unobserved_concepts.data[reordered_last_layer_in_idcs]

    def _reorder_layers_by_target_and_unobserved(self):
        for i, layer in enumerate(reversed(self.layers)):
            i = len(self.layers) - 1 - i
            if i > 0:
                layer_observed_concepts = layer.observed_concepts.data
                if self.layers[i - 1].use_unobserved:
                    prev_layer_unobserved_concepts = self.layers[i - 1].unobserved_concepts.data
                ins_are_used_in_which_outs = [[] for in_concept in range(layer.nb_in_concepts)]
                for out_concept in range(layer.nb_out_concepts):
                    for in_concept in range(layer.nb_in_concepts):
                        abs_observed_concept = torch.abs(layer_observed_concepts[out_concept, in_concept]).item()
                        if abs_observed_concept > 0:
                            ins_are_used_in_which_outs[in_concept].append((abs_observed_concept, out_concept))
                ins_rel_positions = []
                for in_concept in range(layer.nb_in_concepts):
                    in_rel_position = 0
                    sum_abs_observed_concept = 0
                    for abs_observed_concept, out_concept in ins_are_used_in_which_outs[in_concept]:
                        in_rel_position += abs_observed_concept * out_concept
                        sum_abs_observed_concept += abs_observed_concept
                    in_rel_position /= sum_abs_observed_concept
                    if self.layers[i - 1].use_unobserved:
                        prev_layer_unobserved_concepts
                        in_rel_position += 0.1 * (1 - prev_layer_unobserved_concepts[in_concept].item())
                    ins_rel_positions.append((in_concept, in_rel_position))
                ins_rel_positions.sort(key=lambda couple: couple[1])
                ins_rel_positions = [in_concept for in_concept, in_rel_position in ins_rel_positions]
                layer.observed_concepts.data = layer.observed_concepts.data[:, ins_rel_positions]
                self.layers[i - 1].observed_concepts.data = self.layers[i - 1].observed_concepts.data[ins_rel_positions, :]
                if self.layers[i - 1].use_unobserved:
                    self.layers[i - 1].unobserved_concepts.data = self.layers[i - 1].unobserved_concepts.data[ins_rel_positions]

    def to_rule_string(self):
        if len(self.layers) != 2:
            raise Exception("Case with other than 2 fully-connected layers is not coded yet.")

        if len(self.input_module.category_modules) > 0 or len(self.input_module.continuous_modules) > 0 or len(self.input_module.periodic_modules) > 0:
            raise Exception("Case with input module is not coded yet.")

        output_target_idcs = list(range(len(self.feature_names)))
        for idx in self.binary_indices:
            output_target_idcs.remove(idx)
        for first_idx, last_idx, has_missing_values in self.category_first_last_has_missing_values_tuples:
            for idx in range(first_idx, last_idx + 1):
                output_target_idcs.remove(idx)
        for idx, min_value, max_value, has_missing_values in self.continuous_index_min_max_has_missing_values_tuples:
            output_target_idcs.remove(idx)
        for idx, period, has_missing_values in self.periodic_index_period_has_missing_values_tuples:
            output_target_idcs.remove(idx)

        string = ""
        is_first_line = True
        for out_concept in range(self.nb_out_concepts):
            for mid_concept in range(self.layers[-1].nb_in_concepts):
                observed_concept = self.layers[-1].observed_concepts.data[out_concept, mid_concept].item()
                if observed_concept == 1:
                    if not is_first_line:
                        string += "\n"
                    string += self.feature_names[output_target_idcs[out_concept]] + " <- "
                    is_first_litteral = True
                    for in_concept in range(self.layers[0].nb_in_concepts):
                        observed_concept = self.layers[0].observed_concepts.data[mid_concept, in_concept].item()
                        if observed_concept == 1:
                            if not is_first_litteral:
                                string += ", "
                            string += self.feature_names[self.input_module.binary_indices[in_concept]]
                            if is_first_litteral:
                                is_first_litteral = False
                        elif observed_concept == -1:
                            if not is_first_litteral:
                                string += ", "
                            string += "¬" + self.feature_names[self.input_module.binary_indices[in_concept]]
                            if is_first_litteral:
                                is_first_litteral = False
                        elif observed_concept != 0:
                            raise Exception("Case with first layer not taking values in {-1, 0, 1} is not coded yet.")
                    if is_first_line:
                        is_first_line = False
                elif observed_concept != 0:
                    raise Exception("Case with last layer not taking values in {0, 1} is not coded yet.")
        return string
