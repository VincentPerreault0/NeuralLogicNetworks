import random
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

USE_RULE_MODULE = True

NB_RULES = 128

RANDOM_INIT_OBS = True
RANDOM_INIT_UNOBS = True
MONITOR_RESETS_REVIVES = True
USE_RESETS = MONITOR_RESETS_REVIVES and True
UNUSED_CONCEPT_THRESHOLD = 1e-5

CATEGORY_CONCEPT_MULTIPLIER = 1
NB_DICHOTOMIES_PER_CONTINUOUS = 32


class CombinationConcepts(nn.Module):
    """
    Parent class of AND concepts and OR concepts.
    """

    def __init__(
        self,
        is_AND: bool,
        nb_in_concepts: int,
        nb_out_concepts: int,
        use_negation: bool = True,
        use_unobserved: bool = True,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        in_concepts_group_first_stop_pairs: List[Tuple[int, int]] = [],
        device="cuda" if torch.cuda.is_available() else "cpu",
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
        if not random_init_unobs:
            self.init_unobserved_value = self.full_unobserved_value
        self.nb_in_concepts = nb_in_concepts
        self.nb_out_concepts = nb_out_concepts
        self.use_negation = use_negation
        self.use_unobserved = use_unobserved
        self.random_init_obs = random_init_obs
        self.random_init_unobs = random_init_unobs
        self.in_concepts_group_first_stop_pairs = in_concepts_group_first_stop_pairs
        self.device = device
        self.verbose = verbose
        if random_init_obs:
            if use_negation:
                self.observed_concepts = nn.Parameter(torch.Tensor(nb_out_concepts, nb_in_concepts).uniform_(-1, 1))
            else:
                self.observed_concepts = nn.Parameter(torch.Tensor(nb_out_concepts, nb_in_concepts).uniform_(0, 1))
        else:
            self._reinitialize_next_resets(first_time=True)
            self.observed_concepts = nn.Parameter(torch.zeros(nb_out_concepts, nb_in_concepts))
            max_init_with_reset = 2 * nb_in_concepts if use_negation else nb_in_concepts
            for out_concept in range(nb_out_concepts):
                if out_concept < max_init_with_reset:
                    in_concept, sign = self._get_next_reset()
                    self.observed_concepts.data[out_concept, in_concept] = sign
        self.observed_concepts.data = self.observed_concepts.data.to(device)
        if USE_RULE_MODULE and len(self.in_concepts_group_first_stop_pairs) > 0:
            for (
                first_in_concept_idx,
                stop_in_concept_idx,
            ) in self.in_concepts_group_first_stop_pairs:
                if stop_in_concept_idx - first_in_concept_idx == self.nb_out_concepts:
                    self.observed_concepts.data[:, first_in_concept_idx:stop_in_concept_idx] = torch.eye(self.nb_out_concepts, device=self.device)
        if use_unobserved:
            if random_init_unobs:
                self.unobserved_concepts = nn.Parameter(torch.Tensor(nb_out_concepts).uniform_(0, 1))
            else:
                max_init_with_reset = 2 * nb_in_concepts if use_negation else nb_in_concepts
                if nb_out_concepts <= max_init_with_reset:
                    self.unobserved_concepts = nn.Parameter(self.init_unobserved_value * torch.ones(nb_out_concepts))
                else:
                    self.unobserved_concepts = nn.Parameter(torch.Tensor(nb_out_concepts).uniform_(0, 1))
                    self.unobserved_concepts.data[:max_init_with_reset] = self.init_unobserved_value * torch.ones(max_init_with_reset)
            self.unobserved_concepts.data = self.unobserved_concepts.data.to(device)
        if MONITOR_RESETS_REVIVES:
            self.memorizing = False
        self.overridden = ""
        self.to(device)

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
        if self.random_init_obs:
            if self.use_negation:
                self.observed_concepts.data[out_concept, :] = torch.Tensor(1, self.nb_in_concepts).uniform_(-1, 1)
            else:
                self.observed_concepts.data[out_concept, :] = torch.Tensor(1, self.nb_in_concepts).uniform_(0, 1)
        else:
            self.observed_concepts.data[out_concept, :] = torch.zeros(1, self.nb_in_concepts)
            in_concept, sign = self._get_next_reset()
            self.observed_concepts.data[out_concept, in_concept] = sign
        if USE_RULE_MODULE and len(self.in_concepts_group_first_stop_pairs) > 0:
            for (
                first_in_concept_idx,
                stop_in_concept_idx,
            ) in self.in_concepts_group_first_stop_pairs:
                if stop_in_concept_idx - first_in_concept_idx == self.nb_out_concepts:
                    self.observed_concepts.data[:, first_in_concept_idx:stop_in_concept_idx] = torch.eye(self.nb_out_concepts, device=self.device)
        if self.use_unobserved:
            if self.random_init_unobs:
                self.unobserved_concepts.data[out_concept] = torch.Tensor(1).uniform_(0, 1)
            else:
                self.unobserved_concepts.data[out_concept] = self.init_unobserved_value

    def update_parameters(self):
        if self.use_negation:
            self.observed_concepts.data.clamp_(-1, 1)
        else:
            self.observed_concepts.data.clamp_(0, 1)
        if self.use_unobserved:
            self.unobserved_concepts.data.clamp_(0, 1)

    def override(self, new_override):
        self.overridden = new_override

    def sample_weights(self):
        weight_prob_thresholds = torch.abs(self.observed_concepts.data)
        weight_signs = torch.sign(self.observed_concepts.data)
        probs = torch.FloatTensor(self.observed_concepts.data.shape).to(self.device).uniform_()
        mask = probs <= weight_prob_thresholds
        self.observed_concepts.data = mask * weight_signs

    def threshold_weights(self):
        mask_pos = self.observed_concepts.data >= 0.5
        mask_neg = self.observed_concepts.data <= -0.5
        self.observed_concepts.data[mask_pos] = 1
        self.observed_concepts.data[mask_neg] = -1
        self.observed_concepts.data[~(mask_pos + mask_neg)] = 0

    def quantize(self):
        self.threshold_weights()

    def activate_memory(self):
        if MONITOR_RESETS_REVIVES:
            self.memorizing = True
            self.out_concepts_are_all_alive = False
            self.dead_out_concepts = torch.ones(self.nb_out_concepts, device=self.device).bool()

    def review_and_shut_memory(self, used_out_concepts, also_unused_in_concepts, do_check_in_concepts=True):
        if MONITOR_RESETS_REVIVES:
            if do_check_in_concepts:
                if len(used_out_concepts) > 0:
                    in_concepts_max_observed = torch.max(
                        torch.abs(self.observed_concepts[used_out_concepts, :].detach()),
                        dim=0,
                    )[0]
                    used_in_concepts = torch.nonzero(in_concepts_max_observed > UNUSED_CONCEPT_THRESHOLD).view(-1).tolist()
                    used_in_concepts = [used_in_concept for used_in_concept in used_in_concepts if used_in_concept not in also_unused_in_concepts]
                    used_in_concepts.sort()
                else:
                    used_in_concepts = []
                # if len(used_in_concepts) < self.nb_in_concepts:
                #     if len(used_in_concepts) == self.nb_in_concepts - 1:
                #         display_string = "Unused " + self.junction_display_string + " in concept: "
                #     else:
                #         display_string = "Unused " + self.junction_display_string + " in concepts: "
                #     for i, unused_in_concept in enumerate([unused_in_concept for unused_in_concept in range(self.nb_in_concepts) if unused_in_concept not in used_in_concepts]):
                #         if i == 0:
                #             display_string += str(unused_in_concept)
                #         else:
                #             display_string += ", " + str(unused_in_concept)
                #     print(display_string)
            if USE_RESETS:
                if do_check_in_concepts:
                    for in_concept in range(self.nb_in_concepts):
                        if in_concept not in used_in_concepts:
                            self.observed_concepts.data[:, in_concept] = 0
                for out_concept in range(self.nb_out_concepts):
                    if out_concept not in used_out_concepts:
                        self.reset_out_concept(out_concept)
            if do_check_in_concepts:
                return used_in_concepts

    def add_regularization(self, loss):
        if self.use_negation:
            less_than_one_observed_concepts_sums = torch.sum(torch.abs(self.observed_concepts), dim=1)
        else:
            less_than_one_observed_concepts_sums = torch.sum(self.observed_concepts, dim=1)
        less_than_one_observed_concepts_sums = less_than_one_observed_concepts_sums[less_than_one_observed_concepts_sums < 1]
        if less_than_one_observed_concepts_sums.size(0) > 0:
            loss += 1e-1 * torch.nn.functional.mse_loss(
                less_than_one_observed_concepts_sums,
                torch.ones_like(less_than_one_observed_concepts_sums),
            )
        loss += 1e-3 * torch.nn.functional.l1_loss(self.observed_concepts, torch.zeros_like(self.observed_concepts))
        return loss

    def get_discrete_continuous_parameters(self):
        discrete_parameters = []
        continuous_parameters = []
        for name, parameter in self.named_parameters():
            if "unobserved" in name:
                continuous_parameters.append(parameter)
            else:
                discrete_parameters.append(parameter)
        return discrete_parameters, continuous_parameters

    def __repr__(self):
        observed_string = str([[round(val, 3) for val in sublist] for sublist in (1 * self.observed_concepts.data).detach().cpu().numpy().tolist()])
        if self.use_unobserved:
            unobserved_string = str([round(val, 3) for val in (1 * self.unobserved_concepts.data).detach().cpu().numpy().tolist()])
            return "(" + unobserved_string + ", " + observed_string + ")"
        else:
            return "(" + observed_string + ")"

    def load_string(self, init_string):
        unobserved_observed_tuple = eval(init_string)
        if isinstance(unobserved_observed_tuple, list):
            observed_concepts_lists = unobserved_observed_tuple
        elif len(unobserved_observed_tuple) == 1:  # is tuple
            observed_concepts_lists = unobserved_observed_tuple[0]
        else:
            unobserved_concepts_lists = unobserved_observed_tuple[0]
            observed_concepts_lists = unobserved_observed_tuple[1]
            self.unobserved_concepts.data = torch.tensor(unobserved_concepts_lists, device=self.device)
        self.observed_concepts.data = torch.tensor(observed_concepts_lists, device=self.device).float()
        self.nb_in_concepts = self.observed_concepts.data.size(1)
        self.nb_out_concepts = self.observed_concepts.data.size(0)

    def backward_simplify(self, used_out_concepts=[-1], keep_all_in_concepts=False):
        if used_out_concepts == [-1]:
            used_out_concepts = list(range(self.nb_out_concepts))
        if len(used_out_concepts) > 0:
            self.nb_out_concepts = len(used_out_concepts)
            self.observed_concepts.data = self.observed_concepts.data[used_out_concepts, :]
            if self.use_unobserved:
                self.unobserved_concepts.data = self.unobserved_concepts.data[used_out_concepts]
            used_in_concepts = torch.nonzero(torch.max(torch.abs(self.observed_concepts.data), dim=0)[0]).view(-1).tolist()
        else:
            self.nb_out_concepts = 1
            self.observed_concepts.data = torch.zeros((self.nb_in_concepts, 1), device=self.device)
            if self.use_unobserved:
                self.unobserved_concepts.data = (1 - self.full_unobserved_value) * torch.ones((1), device=self.device)
            used_in_concepts = []
        if keep_all_in_concepts:
            pass
        elif len(used_in_concepts) > 0:
            self.nb_in_concepts = len(used_in_concepts)
            self.observed_concepts.data = self.observed_concepts.data[:, used_in_concepts]
        else:
            self.nb_in_concepts = 1
            self.observed_concepts.data = self.observed_concepts.data[:, [0]]
        if MONITOR_RESETS_REVIVES:
            self.memorizing = False
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
            else:
                self.nb_out_concepts = 1
                self.observed_concepts.data = self.observed_concepts.data[[0], :]
                if self.use_unobserved:
                    self.unobserved_concepts.data = (1 - self.full_unobserved_value) * torch.ones((1), device=self.device)
        if MONITOR_RESETS_REVIVES:
            self.memorizing = False
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
                                if torch.equal(
                                    self.observed_concepts.data[out_i, :],
                                    self.observed_concepts.data[out_j, :],
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
                            "Kept " + self.junction_display_string + "_concepts.observed_concept[" + str(out_idx) + ", " + str(in_idx) + f"] = {old_value:.3f}",
                            self.verbose,
                            log_files,
                        )
                    else:
                        did_prune = True
                        tmp_extra_string_space = " " if old_value >= 0 else ""
                        print_log(
                            "Pruned "
                            + self.junction_display_string
                            + "_concepts.observed_concept["
                            + str(out_idx)
                            + ", "
                            + str(in_idx)
                            + f"] = "
                            + tmp_extra_string_space
                            + f"{old_value:.3f} -> 0     (new loss = "
                            + str(new_loss)
                            + ")",
                            self.verbose,
                            log_files,
                        )
                        init_loss = new_loss
                    current_weight += 1
                    progress_bar_hook(current_weight)
        return init_loss, did_prune

    def get_nb_weights(self):
        return torch.count_nonzero(self.observed_concepts.data).item()

    def new_quantize(self, eval_model_func, quantize_type="sub", goes_by_layer=False, log_files=[], progress_bar_hook=lambda weight: None):
        losses = []
        if goes_by_layer:
            self._new_quantize_inner(eval_model_func, losses, self.observed_concepts.data, quantize_type=quantize_type, log_files=log_files, progress_bar_hook=progress_bar_hook)
        else:
            current_weight = 0
            for out_concept in range(self.nb_out_concepts):
                self._new_quantize_inner(
                    eval_model_func,
                    losses,
                    self.observed_concepts.data[out_concept : out_concept + 1, :],
                    quantize_type=quantize_type,
                    log_files=log_files,
                    progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
                )
            current_weight += torch.count_nonzero(self.observed_concepts.data[out_concept : out_concept + 1, :]).item()
        return losses

    def _new_quantize_inner(self, eval_model_func, losses, observed_concepts_data, quantize_type="sub", log_files=[], progress_bar_hook=lambda weight: None):
        old_observed_concepts = 1 * observed_concepts_data
        old_observed_concepts_abs_values = list(set(torch.abs(old_observed_concepts).view(-1).tolist()))
        current_weight = 0
        if quantize_type == "sub":
            old_observed_concepts_abs_values.sort()
            observed_concepts_data[observed_concepts_data > 0] = 1
            observed_concepts_data[observed_concepts_data < 0] = -1
            losses.append(eval_model_func())
            print_log(
                "Quantized " + self.junction_display_string + "_concepts.observed_concepts to {-1, 1}       (new loss = " + str(losses[-1]) + ")",
                self.verbose,
                log_files,
            )
            for old_observed_concept_abs_value in old_observed_concepts_abs_values:
                if old_observed_concept_abs_value > 0:
                    param_out_ins = ((old_observed_concepts == old_observed_concept_abs_value) + (old_observed_concepts == -1 * old_observed_concept_abs_value)).nonzero().tolist()
                    for param_out_in in param_out_ins:
                        param_out, param_in = tuple(param_out_in)
                        old_quantized_value = observed_concepts_data[param_out, param_in].item()
                        observed_concepts_data[param_out, param_in] = 0
                        new_loss = eval_model_func()
                        if losses[-1] < new_loss:
                            observed_concepts_data[param_out, param_in] = old_quantized_value
                            losses.append(losses[-1])
                            tmp_extra_string_space = " " if old_quantized_value == 1 else ""
                            print_log(
                                "Kept Quantized "
                                + self.junction_display_string
                                + "_concepts.observed_concept["
                                + str(param_out)
                                + ", "
                                + str(param_in)
                                + f"] = "
                                + tmp_extra_string_space
                                + f"{old_observed_concepts[param_out,param_in].item():.3f} -> "
                                + tmp_extra_string_space
                                + f"{old_quantized_value:.0f}",
                                self.verbose,
                                log_files,
                            )
                        else:
                            losses.append(new_loss)
                            print_log(
                                "Pruned "
                                + self.junction_display_string
                                + "_concepts.observed_concept["
                                + str(param_out)
                                + ", "
                                + str(param_in)
                                + f"] = {old_observed_concepts[param_out,param_in].item():.3f} -> 0     (new loss = "
                                + str(losses[-1])
                                + ")",
                                self.verbose,
                                log_files,
                            )
                        current_weight += 1
                        progress_bar_hook(current_weight)
        elif quantize_type == "add":
            old_observed_concepts_abs_values.sort(reverse=True)
            observed_concepts_data[:, :] = torch.zeros_like(observed_concepts_data)
            losses.append(eval_model_func())
            print_log(
                "Zeroed " + self.junction_display_string + "_concepts.observed_concepts          (new loss = " + str(losses[-1]) + ")",
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
                                "Kept Zeroed "
                                + self.junction_display_string
                                + "_concepts.observed_concept["
                                + str(param_out)
                                + ", "
                                + str(param_in)
                                + f"] = {old_observed_concepts[param_out,param_in].item():.3f} -> 0",
                                self.verbose,
                                log_files,
                            )
                        else:
                            losses.append(new_loss)
                            print_log(
                                "Added Quantized "
                                + self.junction_display_string
                                + "_concepts.observed_concept["
                                + str(param_out)
                                + ", "
                                + str(param_in)
                                + "] = "
                                + tmp_extra_string_space
                                + f"{old_observed_concepts[param_out,param_in].item():.3f} -> "
                                + tmp_extra_string_space
                                + f"{observed_concepts_data[param_out, param_in].item():.0f}     (new loss = "
                                + str(losses[-1])
                                + ")",
                                self.verbose,
                                log_files,
                            )
                        current_weight += 1
                        progress_bar_hook(current_weight)
        elif quantize_type[:4] == "sel_":
            if quantize_type == "sel_asc":
                old_observed_concepts_abs_values.sort()
            elif quantize_type == "sel_desc":
                old_observed_concepts_abs_values.sort(reverse=True)
            else:
                raise Exception("quantize_type not legal!")
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
                                "Quantized "
                                + self.junction_display_string
                                + "_concepts.observed_concept["
                                + str(param_out)
                                + ", "
                                + str(param_in)
                                + "] = "
                                + tmp_extra_string_space
                                + f"{old_observed_concepts[param_out,param_in].item():.3f} -> "
                                + tmp_extra_string_space
                                + f"{observed_concepts_data[param_out, param_in].item():.0f}     (new loss = "
                                + str(losses[-1])
                                + ")",
                                self.verbose,
                                log_files,
                            )
                        else:
                            observed_concepts_data[param_out, param_in] = 0
                            losses.append(loss_without)
                            print_log(
                                "Pruned "
                                + self.junction_display_string
                                + "_concepts.observed_concept["
                                + str(param_out)
                                + ", "
                                + str(param_in)
                                + "] = "
                                + tmp_extra_string_space
                                + f"{old_observed_concepts[param_out,param_in].item():.3f} ->  0        (new loss = "
                                + str(losses[-1])
                                + ")",
                                self.verbose,
                                log_files,
                            )
                        current_weight += 1
                        progress_bar_hook(current_weight)
        else:
            raise Exception("quantize_type not legal!")


class AndConcepts(CombinationConcepts):
    """
    AND concepts: present if all observed and unobserved necessary concepts are present
    """

    def __init__(
        self,
        nb_in_concepts: int,
        nb_out_concepts: int,
        use_negation: bool = True,
        use_unobserved: bool = True,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        in_concepts_group_first_stop_pairs: List[Tuple[int, int]] = [],
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = VERBOSE,
    ):
        super().__init__(
            True,
            nb_in_concepts,
            nb_out_concepts,
            use_negation=use_negation,
            use_unobserved=use_unobserved,
            random_init_obs=random_init_obs,
            random_init_unobs=random_init_unobs,
            in_concepts_group_first_stop_pairs=in_concepts_group_first_stop_pairs,
            device=device,
            verbose=verbose,
        )

    def forward(self, x):
        """Forward pass"""
        if self.overridden == "stoch" or self.overridden == "thresh":
            tmp = 1 * self.observed_concepts.data
            if self.overridden == "stoch":
                self.sample_weights()
            else:
                self.threshold_weights()

        x_v = x.view(-1, 1, self.nb_in_concepts)
        observed_concepts_v = self.observed_concepts.view(1, self.nb_out_concepts, self.nb_in_concepts)
        if not self.use_negation:
            result = torch.prod(1 - observed_concepts_v * (1 - x_v), dim=2)
        else:
            if not self.training:
                result = torch.prod(
                    1 - F.relu(observed_concepts_v) * (1 - x_v) - F.relu(-1 * observed_concepts_v) * x_v,
                    dim=2,
                )
            else:
                with torch.no_grad():
                    one_hot_equals_0 = torch.zeros(1, self.nb_out_concepts, self.nb_in_concepts).to(self.device)
                    one_hot_equals_0[observed_concepts_v == 0] = 1
                result = torch.prod(
                    1 - F.relu(observed_concepts_v) * (1 - x_v) - F.relu(-1 * observed_concepts_v) * x_v + one_hot_equals_0 * observed_concepts_v * 2 * (x_v - 0.5),
                    dim=2,
                )

        if MONITOR_RESETS_REVIVES and self.training and self.memorizing and not self.out_concepts_are_all_alive:
            dead_out_concepts_max_values = torch.max(result[:, self.dead_out_concepts].detach(), dim=0)[0]
            old_dead_out_concepts = self.dead_out_concepts.clone()
            self.dead_out_concepts[old_dead_out_concepts] *= dead_out_concepts_max_values < UNUSED_CONCEPT_THRESHOLD
            if torch.logical_not(self.dead_out_concepts).all().item():
                self.out_concepts_are_all_alive = True

        if self.use_unobserved:
            unobserved_concepts_v = self.unobserved_concepts.view(1, self.nb_out_concepts)
            result = unobserved_concepts_v * result

        if self.overridden == "stoch" or self.overridden == "thresh":
            self.observed_concepts.data = tmp
        return result

    def update_unobserved_concepts(self, out_concept, unused_in_concept, unused_in_prob):
        self.unobserved_concepts.data[out_concept] *= (
            1
            - F.relu(self.observed_concepts.data[out_concept, unused_in_concept]) * (1 - unused_in_prob)
            - F.relu(-1 * self.observed_concepts.data[out_concept, unused_in_concept]) * unused_in_prob
        )


class OrConcepts(CombinationConcepts):
    """
    OR concepts: present if any observed or unobserved sufficient concept is present
    """

    def __init__(
        self,
        nb_in_concepts: int,
        nb_out_concepts: int,
        use_negation: bool = True,
        use_unobserved: bool = True,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        in_concepts_group_first_stop_pairs: List[Tuple[int, int]] = [],
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = VERBOSE,
    ):
        super().__init__(
            False,
            nb_in_concepts,
            nb_out_concepts,
            use_negation=use_negation,
            use_unobserved=use_unobserved,
            random_init_obs=random_init_obs,
            random_init_unobs=random_init_unobs,
            in_concepts_group_first_stop_pairs=in_concepts_group_first_stop_pairs,
            device=device,
            verbose=verbose,
        )

    def forward(self, x):
        """Forward pass"""
        if self.overridden == "stoch" or self.overridden == "thresh":
            tmp = 1 * self.observed_concepts.data
            if self.overridden == "stoch":
                self.sample_weights()
            else:
                self.threshold_weights()

        x_v = x.view(-1, 1, self.nb_in_concepts)
        observed_concepts_v = self.observed_concepts.view(1, self.nb_out_concepts, self.nb_in_concepts)
        if not self.use_negation:
            result = torch.prod(1 - observed_concepts_v * x_v, dim=2)
        else:
            if not self.training:
                result = torch.prod(
                    1 - F.relu(observed_concepts_v) * x_v - F.relu(-1 * observed_concepts_v) * (1 - x_v),
                    dim=2,
                )
            else:
                with torch.no_grad():
                    one_hot_equals_0 = torch.zeros(1, self.nb_out_concepts, self.nb_in_concepts).to(self.device)
                    one_hot_equals_0[observed_concepts_v == 0] = 1
                result = torch.prod(
                    1 - F.relu(observed_concepts_v) * x_v - F.relu(-1 * observed_concepts_v) * (1 - x_v) - one_hot_equals_0 * observed_concepts_v * 2 * (x_v - 0.5),
                    dim=2,
                )

        if MONITOR_RESETS_REVIVES and self.training and self.memorizing and not self.out_concepts_are_all_alive:
            dead_out_concepts_min_values = torch.min(result[:, self.dead_out_concepts].detach(), dim=0)[0]
            old_dead_out_concepts = self.dead_out_concepts.clone()
            self.dead_out_concepts[old_dead_out_concepts] *= dead_out_concepts_min_values > 1 - UNUSED_CONCEPT_THRESHOLD
            if torch.logical_not(self.dead_out_concepts).all().item():
                self.out_concepts_are_all_alive = True

        if self.use_unobserved:
            unobserved_concepts_v = self.unobserved_concepts.view(1, self.nb_out_concepts)
            result = 1 - (1 - unobserved_concepts_v) * result
        else:
            result = 1 - result

        if self.overridden == "stoch" or self.overridden == "thresh":
            self.observed_concepts.data = tmp
        return result

    def update_unobserved_concepts(self, out_concept, unused_in_concept, unused_in_prob):
        self.unobserved_concepts.data[out_concept] = 1 - (1 - self.unobserved_concepts.data[out_concept]) * (
            1
            - F.relu(self.observed_concepts.data[out_concept, unused_in_concept]) * unused_in_prob
            - F.relu(-1 * self.observed_concepts.data[out_concept, unused_in_concept]) * (1 - unused_in_prob)
        )


class Dichotomies(nn.Module):
    """
    Fuzzy dichotomies: present to some degree when associated continuous input feature is greater than the boundary
    """

    def __init__(
        self,
        nb_dichotomies: int,
        min_value: float,
        max_value: float,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    """
    Fuzzy dichotomies: present to some degree when associated periodic input feature is in the half-period centered around the center
    """

    def __init__(
        self,
        nb_dichotomies: int,
        period: float,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
        random_init_obs: bool = RANDOM_INIT_OBS,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
            device=device,
            verbose=verbose,
        )
        self.out_concepts = OrConcepts(
            nb_intervals,
            nb_out_concepts,
            use_negation=False,
            use_unobserved=False,
            random_init_obs=True if USE_RULE_MODULE else random_init_obs,
            device=device,
            verbose=verbose,
        )
        if not random_init_obs:
            if nb_intervals >= nb_dichotomies + 1:
                minus_eye = torch.cat(
                    (
                        -1 * torch.eye(nb_dichotomies, device=device),
                        torch.zeros((1, nb_dichotomies), device=device),
                    ),
                    dim=0,
                )
                plus_eye = torch.cat(
                    (
                        torch.zeros((1, nb_dichotomies), device=device),
                        torch.eye(nb_dichotomies, device=device),
                    ),
                    dim=0,
                )
                self.intervals.observed_concepts.data[: nb_dichotomies + 1, :] = minus_eye + plus_eye
            if not USE_RULE_MODULE and nb_out_concepts >= nb_intervals:
                self.out_concepts.observed_concepts.data[:nb_intervals, :] = torch.eye(nb_intervals, device=device)
        self.to(device)

    def reset_out_concept(self, out_concept):
        self.out_concepts.reset_out_concept(out_concept)

    def update_parameters(self):
        self.dichotomies.update_parameters()
        self.intervals.update_parameters()
        self.out_concepts.update_parameters()

    def override(self, new_override):
        self.intervals.override(new_override)
        self.out_concepts.override(new_override)

    def quantize(self):
        self.intervals.quantize()
        self.out_concepts.quantize()

    def activate_memory(self):
        if MONITOR_RESETS_REVIVES:
            self.intervals.activate_memory()
            self.out_concepts.activate_memory()

    def review_and_shut_memory(self, used_out_concepts):
        if MONITOR_RESETS_REVIVES:
            unused_in_concepts = self.out_concepts.new_review_and_shut_memory(used_out_concepts, [])
            self.intervals.new_review_and_shut_memory(unused_in_concepts, [])

    def forward(self, x):
        """Forward pass"""
        return self.out_concepts(self.intervals(self.dichotomies(x)))

    def add_regularization(self, loss):
        loss += self.intervals.add_regularization(loss)
        loss += self.out_concepts.add_regularization(loss)
        return loss

    def get_discrete_continuous_parameters(self):
        intervals_discrete_parameters, intervals_continuous_parameters = self.intervals.get_discrete_continuous_parameters()
        out_concepts_discrete_parameters, out_concepts_continuous_parameters = self.out_concepts.get_discrete_continuous_parameters()

        discrete_parameters = intervals_discrete_parameters + out_concepts_discrete_parameters
        continuous_parameters = list(self.dichotomies.parameters()) + intervals_continuous_parameters + out_concepts_continuous_parameters
        return discrete_parameters, continuous_parameters

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

    def new_quantize(self, eval_model_func, quantize_type="sub", goes_backward=True, goes_by_layer=False, log_files=[], progress_bar_hook=lambda weight: None):
        losses = []
        current_weight = 0
        if goes_backward:
            old_module_nb_weights = self.out_concepts.get_nb_weights()
            losses += self.out_concepts.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_by_layer=goes_by_layer,
                log_files=log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
            )
            current_weight += old_module_nb_weights
            losses += self.intervals.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_by_layer=goes_by_layer,
                log_files=log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
            )
        else:
            old_module_nb_weights = self.intervals.get_nb_weights()
            losses += self.intervals.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_by_layer=goes_by_layer,
                log_files=log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
            )
            current_weight += old_module_nb_weights
            losses += self.out_concepts.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_by_layer=goes_by_layer,
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
        nb_in_concepts: int,
        category_first_last_pairs: List[Tuple[int]] = [],
        continuous_index_min_max_triples: List[Tuple[int, float, float]] = [],
        periodic_index_period_pairs: List[Tuple[int, float]] = [],
        category_concept_multiplier: float = CATEGORY_CONCEPT_MULTIPLIER,
        category_nb_out_concepts: int = -1,
        nb_dichotomies_per_continuous: int = NB_DICHOTOMIES_PER_CONTINUOUS,
        nb_intervals_per_continuous: Union[int, None] = None,
        nb_out_concepts_per_continuous: Union[int, None] = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = VERBOSE,
    ):
        super().__init__()
        self.nb_in_concepts = nb_in_concepts
        binary_indices = set(range(nb_in_concepts))
        nb_out_concepts = 0

        category_modules = []
        self.category_first_last_pairs = []
        for category_first_last_pair in category_first_last_pairs:
            first_idx = category_first_last_pair[0]
            last_idx = category_first_last_pair[-1]
            binary_indices = binary_indices - set(range(first_idx, last_idx + 1))
            self.category_first_last_pairs.append((first_idx, last_idx))
            category_nb_in_concepts = last_idx - first_idx + 1
            if not USE_RULE_MODULE:
                category_nb_out_concepts = round(category_concept_multiplier * category_nb_in_concepts)
            nb_out_concepts += category_nb_out_concepts
            category_modules.append(
                OrConcepts(
                    category_nb_in_concepts,
                    category_nb_out_concepts,
                    use_negation=False,
                    use_unobserved=False,
                    random_init_obs=True if USE_RULE_MODULE else False,
                    device=device,
                    verbose=verbose,
                )
            )
            if not USE_RULE_MODULE and category_concept_multiplier >= 1:
                category_modules[-1].observed_concepts.data[:category_nb_in_concepts, :] = torch.eye(category_nb_in_concepts, device=device)
        self.category_modules = nn.ModuleList(category_modules)

        self.continuous_index_min_max_triples = continuous_index_min_max_triples
        binary_indices = binary_indices - set([continuous_index_min_max_triple[0] for continuous_index_min_max_triple in continuous_index_min_max_triples])
        continuous_modules = []
        for continuous_index, min_value, max_value in continuous_index_min_max_triples:
            continuous_modules.append(
                ContinuousPreProcessingModule(
                    nb_dichotomies_per_continuous,
                    nb_intervals_per_continuous,
                    nb_out_concepts_per_continuous,
                    min_value=min_value,
                    max_value=max_value,
                    random_init_obs=False,
                    device=device,
                    verbose=verbose,
                )
            )
            nb_out_concepts += continuous_modules[-1].nb_out_concepts
        self.continuous_modules = nn.ModuleList(continuous_modules)

        self.periodic_index_period_pairs = [(periodic_index_period_pair[0], periodic_index_period_pair[1]) for periodic_index_period_pair in periodic_index_period_pairs]
        binary_indices = binary_indices - set([periodic_index_period_pair[0] for periodic_index_period_pair in periodic_index_period_pairs])
        periodic_modules = []
        for periodic_index, period in self.periodic_index_period_pairs:
            periodic_modules.append(
                ContinuousPreProcessingModule(
                    nb_dichotomies_per_continuous,
                    nb_intervals_per_continuous,
                    nb_out_concepts_per_continuous,
                    period=period,
                    random_init_obs=False,
                    device=device,
                    verbose=verbose,
                )
            )
            nb_out_concepts += periodic_modules[-1].nb_out_concepts
        self.periodic_modules = nn.ModuleList(periodic_modules)

        self.binary_indices = sorted(list(binary_indices))
        nb_out_concepts += len(binary_indices)
        self.nb_out_concepts = nb_out_concepts
        self.device = device
        self.to(device)

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

    def quantize(self):
        for category_module in self.category_modules:
            category_module.quantize()
        for continuous_module in self.continuous_modules:
            continuous_module.quantize()
        for periodic_module in self.periodic_modules:
            periodic_module.quantize()

    def activate_memory(self):
        if MONITOR_RESETS_REVIVES:
            for category_module in self.category_modules:
                category_module.activate_memory()
            for continuous_module in self.continuous_modules:
                continuous_module.activate_memory()
            for periodic_module in self.periodic_modules:
                periodic_module.activate_memory()

    def review_and_shut_memory(self, used_out_concepts):
        if MONITOR_RESETS_REVIVES:
            first_out_concept_idx = len(self.binary_indices)

            for category_module in self.category_modules:
                module_used_out_concepts = [
                    used_out_concept - first_out_concept_idx
                    for used_out_concept in used_out_concepts
                    if used_out_concept >= first_out_concept_idx and used_out_concept < first_out_concept_idx + category_module.nb_out_concepts
                ]
                category_module.review_and_shut_memory(module_used_out_concepts, [])
                first_out_concept_idx += category_module.nb_out_concepts

            for continuous_module in self.continuous_modules:
                module_used_out_concepts = [
                    used_out_concept - first_out_concept_idx
                    for used_out_concept in used_out_concepts
                    if used_out_concept >= first_out_concept_idx and used_out_concept < first_out_concept_idx + continuous_module.nb_out_concepts
                ]
                continuous_module.review_and_shut_memory(module_used_out_concepts)
                first_out_concept_idx += continuous_module.nb_out_concepts

            for periodic_module in self.periodic_modules:
                module_used_out_concepts = [
                    used_out_concept - first_out_concept_idx
                    for used_out_concept in used_out_concepts
                    if used_out_concept >= first_out_concept_idx and used_out_concept < first_out_concept_idx + periodic_module.nb_out_concepts
                ]
                periodic_module.review_and_shut_memory(module_used_out_concepts)
                first_out_concept_idx += periodic_module.nb_out_concepts

    def forward(self, x):
        """Forward pass"""
        results = []

        if len(self.binary_indices) > 0:
            results.append(x[:, self.binary_indices])

        for i, category_module in enumerate(self.category_modules):
            first_idx, last_idx = self.category_first_last_pairs[i]
            results.append(category_module(x[:, first_idx : last_idx + 1]))

        for i, continuous_module in enumerate(self.continuous_modules):
            idx, min_value, max_value = self.continuous_index_min_max_triples[i]
            results.append(continuous_module(x[:, idx]))

        for i, periodic_module in enumerate(self.periodic_modules):
            idx, period = self.periodic_index_period_pairs[i]
            results.append(periodic_module(x[:, idx]))

        if len(results) == 1:
            return results[0]
        elif len(results) > 0:
            return torch.cat(results, dim=1)
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

    def get_discrete_continuous_parameters(self):
        discrete_parameters = []
        continuous_parameters = []
        for category_module in self.category_modules:
            module_discrete_parameters, module_continuous_parameters = category_module.get_discrete_continuous_parameters()
            discrete_parameters += module_discrete_parameters
            continuous_parameters += module_continuous_parameters
        for continuous_module in self.continuous_modules:
            module_discrete_parameters, module_continuous_parameters = continuous_module.get_discrete_continuous_parameters()
            discrete_parameters += module_discrete_parameters
            continuous_parameters += module_continuous_parameters
        for periodic_module in self.periodic_modules:
            module_discrete_parameters, module_continuous_parameters = periodic_module.get_discrete_continuous_parameters()
            discrete_parameters += module_discrete_parameters
            continuous_parameters += module_continuous_parameters
        return discrete_parameters, continuous_parameters

    def __repr__(self):
        string = ""
        for in_concept_idx in range(self.nb_in_concepts):
            found = False
            if not found:
                if in_concept_idx in self.binary_indices:
                    if len(string) > 0:
                        string += "\n"
                    string += "Binary " + str(in_concept_idx)
                    found = True

            if not found:
                for i, first_last_pair in enumerate(self.category_first_last_pairs):
                    first, last = first_last_pair
                    if in_concept_idx == first:
                        if len(string) > 0:
                            string += "\n"
                        string += "Category " + str(first) + "-" + str(last) + " " + str(self.category_modules[i])
                        found = True
                        break
                    elif in_concept_idx > first and in_concept_idx <= last:
                        found = True
                        break

            if not found:
                for i, continuous_index_min_max_triple in enumerate(self.continuous_index_min_max_triples):
                    index, min_value, max_value = continuous_index_min_max_triple
                    if in_concept_idx == index:
                        if len(string) > 0:
                            string += "\n"
                        string += "Continuous " + str(index) + " in [" + str(round(min_value, 3)) + ", " + str(round(max_value, 3)) + "] " + str(self.continuous_modules[i])
                        found = True
                        break

            if not found:
                for i, periodic_index_period_pair in enumerate(self.periodic_index_period_pairs):
                    index, period = periodic_index_period_pair
                    if in_concept_idx == index:
                        if len(string) > 0:
                            string += "\n"
                        string += "Periodic " + str(index) + " of period " + str(round(period, 3)) + " " + str(self.periodic_modules[i])
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
            first_idx, last_idx = self.category_first_last_pairs[i]
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
            idx, min_value, max_value = self.continuous_index_min_max_triples[i]
            if len(string) > 0:
                string += "\n"
            string += "Continuous " + feature_names[idx] + " in [" + str(round(min_value, 3)) + ", " + str(round(max_value, 3)) + "] " + str(continuous_module)

        for i, periodic_module in enumerate(self.periodic_modules):
            idx, period = self.periodic_index_period_pairs[i]
            if len(string) > 0:
                string += "\n"
            string += "Periodic " + feature_names[idx] + " of period " + str(round(period, 3)) + " " + str(periodic_module)

        return string

    def load_string(self, init_string):
        lines = init_string.split("\n")
        in_continuous_periodic_counter = 0
        binaries_to_remove = list(range(len(self.binary_indices)))
        categories_to_remove = list(range(len(self.category_first_last_pairs)))
        continuous_to_remove = list(range(len(self.continuous_index_min_max_triples)))
        periodic_to_remove = list(range(len(self.periodic_index_period_pairs)))
        for i, line in enumerate(lines):
            if len(line) > 0:
                if in_continuous_periodic_counter > 0:
                    module_init_string += line + "\n"
                    in_continuous_periodic_counter -= 1
                    if in_continuous_periodic_counter == 0:
                        found = False
                        if not found:
                            for j, continuous_index_min_max_triple in enumerate(self.continuous_index_min_max_triples):
                                (
                                    continuous_index,
                                    continuous_min_value,
                                    continuous_max_value,
                                ) = continuous_index_min_max_triple
                                if index == continuous_index:
                                    continuous_to_remove.remove(j)
                                    self.continuous_modules[j].load_string(module_init_string)
                                    found = True
                                    break
                        if not found:
                            for j, periodic_index_period_pair in enumerate(self.periodic_index_period_pairs):
                                periodic_index, periodic_period = periodic_index_period_pair
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
                        first_last_pair = (
                            int(first_last_strings[0]),
                            int(first_last_strings[1]),
                        )
                        module_init_string = line[len("Category " + first_last_string + " ") :]
                        for j, category_first_last_pair in enumerate(self.category_first_last_pairs):
                            if first_last_pair == category_first_last_pair:
                                categories_to_remove.remove(j)
                                self.category_modules[j].load_string(module_init_string)
                                break
                            if j == len(self.category_first_last_pairs) - 1:
                                raise Exception("LNNInputModule input string is not formatted correctly.\n Category not found at line " + str(i) + " --> " + str(line))
                    elif line[:10] == "Continuous":
                        words = line.split(" ")
                        index = int(words[1])
                        min_value_string = words[3]
                        max_value_string = words[4]
                        module_init_string = line[len("Continuous " + str(index) + " in " + min_value_string + " " + max_value_string + " ") :] + "\n"
                        in_continuous_periodic_counter = 2
                    elif line[:8] == "Periodic":
                        words = line.split(" ")
                        index = int(words[1])
                        period_string = words[4]
                        module_init_string = line[len("Periodic " + str(index) + " of period " + period_string + " ") :] + "\n"
                        in_continuous_periodic_counter = 2
                    else:
                        raise Exception("LNNInputModule input string is not formatted correctly.\n Input type not recognized at line " + str(i) + " --> " + str(line))

        for i in reversed(binaries_to_remove):
            del self.binary_indices[i]

        for i in reversed(categories_to_remove):
            del self.category_first_last_pairs[i]
            del self.category_modules[i]

        for i in reversed(continuous_to_remove):
            del self.continuous_index_min_max_triples[i]
            del self.continuous_modules[i]

        for i in reversed(periodic_to_remove):
            del self.periodic_index_period_pairs[i]
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
            del self.category_first_last_pairs[i]
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
            del self.continuous_index_min_max_triples[i]
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
            del self.periodic_index_period_pairs[i]
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
            del self.category_first_last_pairs[i]
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
            del self.continuous_index_min_max_triples[i]
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
            del self.periodic_index_period_pairs[i]
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

    def new_quantize(self, eval_model_func, quantize_type="sub", goes_backward=True, goes_by_layer=False, log_files=[], progress_bar_hook=lambda weight: None):
        losses = []
        current_weight = 0
        for category_module in self.category_modules:
            old_module_nb_weights = category_module.get_nb_weights()
            losses += category_module.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_by_layer=goes_by_layer,
                log_files=log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
            )
            current_weight += old_module_nb_weights
        for continuous_module in self.continuous_modules:
            old_module_nb_weights = continuous_module.get_nb_weights()
            losses += continuous_module.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_backward=goes_backward,
                goes_by_layer=goes_by_layer,
                log_files=log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
            )
            current_weight += old_module_nb_weights
        for periodic_module in self.periodic_modules:
            old_module_nb_weights = periodic_module.get_nb_weights()
            losses += periodic_module.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_backward=goes_backward,
                goes_by_layer=goes_by_layer,
                log_files=log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight),
            )
            current_weight += old_module_nb_weights
        return losses

    def reorder_lexicographically(self):
        reordered_out_idcs = list(range(len(self.binary_indices)))
        first_out_concept_idx = len(self.binary_indices)

        for i, category_module in enumerate(self.category_modules):
            reordered_module_out_idcs = find_lexicographical_order(category_module.observed_concepts.data)
            category_module.observed_concepts.data = category_module.observed_concepts.data[reordered_module_out_idcs, :]
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
        nb_in_concepts: int,
        nb_out_concepts: int,
        nb_concepts_per_hidden_layer: int = NB_RULES,
        nb_hidden_layers: int = 1,
        last_layer_is_OR_no_neg: bool = True,
        category_first_last_pairs: List[Tuple[int, int]] = [],
        continuous_index_min_max_triples: List[Tuple[int, float, float]] = [],
        periodic_index_period_pairs: List[Tuple[int, float]] = [],
        column_names: List[str] = [],
        category_concept_multiplier: float = CATEGORY_CONCEPT_MULTIPLIER,
        nb_dichotomies_per_continuous: int = NB_DICHOTOMIES_PER_CONTINUOUS,
        nb_intervals_per_continuous: Union[int, None] = None,
        nb_out_concepts_per_continuous: Union[int, None] = None,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = VERBOSE,
        init_string="",
    ):
        super().__init__()
        self.nb_in_concepts = nb_in_concepts
        self.nb_out_concepts = nb_out_concepts
        self.device = device
        self.verbose = verbose

        self.category_first_last_pairs = [(first_idx, last_idx) for first_idx, last_idx in category_first_last_pairs]
        self.continuous_index_min_max_triples = [(idx, min_value, max_value) for idx, min_value, max_value in continuous_index_min_max_triples]
        self.periodic_index_period_pairs = [(idx, period) for idx, period in periodic_index_period_pairs]
        self.feature_names = column_names
        self.input_module = NLNPreProcessingModules(
            nb_in_concepts,
            category_first_last_pairs=category_first_last_pairs,
            continuous_index_min_max_triples=continuous_index_min_max_triples,
            periodic_index_period_pairs=periodic_index_period_pairs,
            category_concept_multiplier=(-1 if USE_RULE_MODULE else category_concept_multiplier),
            category_nb_out_concepts=(nb_concepts_per_hidden_layer if USE_RULE_MODULE else -1),
            nb_dichotomies_per_continuous=nb_dichotomies_per_continuous,
            nb_intervals_per_continuous=nb_intervals_per_continuous,
            nb_out_concepts_per_continuous=(nb_concepts_per_hidden_layer if USE_RULE_MODULE else nb_out_concepts_per_continuous),
            device=device,
            verbose=verbose,
        )
        self.binary_indices = self.input_module.binary_indices.copy()
        in_concepts_group_first_stop_pairs = self.input_module.get_out_concepts_group_first_stop_pairs()

        self.nb_hidden_layers = nb_hidden_layers
        self.nb_concepts_per_hidden_layer = nb_concepts_per_hidden_layer

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
                        device=device,
                        verbose=verbose,
                    )
                )
        if last_layer_is_OR_no_neg:
            if nb_hidden_layers == 0:
                layers.append(
                    OrConcepts(
                        self.input_module.nb_out_concepts,
                        nb_out_concepts,
                        use_negation=False,
                        use_unobserved=True,
                        random_init_obs=random_init_obs,
                        random_init_unobs=random_init_unobs,
                        device=device,
                        verbose=verbose,
                    )
                )
            else:
                layers.append(
                    OrConcepts(
                        nb_concepts_per_hidden_layer,
                        nb_out_concepts,
                        use_negation=False,
                        use_unobserved=True,
                        random_init_obs=random_init_obs,
                        random_init_unobs=random_init_unobs,
                        device=device,
                        verbose=verbose,
                    )
                )
        else:
            if nb_hidden_layers == 0:
                layers.append(
                    AndConcepts(
                        self.input_module.nb_out_concepts,
                        nb_out_concepts,
                        use_negation=True,
                        use_unobserved=True,
                        random_init_obs=random_init_obs,
                        random_init_unobs=random_init_unobs,
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
                        device=device,
                        verbose=verbose,
                    )
                )
        layers[-1].observed_concepts.data = 0 * torch.ones((nb_out_concepts, nb_concepts_per_hidden_layer), device=device)
        layers[-1].unobserved_concepts.data = 1 * torch.ones(nb_out_concepts, device=device)
        self.layers = nn.Sequential(*layers)

        if len(init_string) > 0:
            self.load_string(init_string)

        self.to(device)

    def update_parameters(self):
        self.input_module.update_parameters()
        if USE_RULE_MODULE:
            for first_in_concept_idx, stop_in_concept_idx in self.layers[0].in_concepts_group_first_stop_pairs:
                if stop_in_concept_idx - first_in_concept_idx == self.layers[0].nb_out_concepts:
                    self.layers[0].observed_concepts.data[:, first_in_concept_idx:stop_in_concept_idx][
                        torch.arange(0, self.layers[0].nb_out_concepts).unsqueeze(1) != torch.arange(0, self.layers[0].nb_out_concepts).unsqueeze(0)
                    ] = 0
        for layer in self.layers:
            layer.update_parameters()

    def override(self, new_override):
        self.input_module.override(new_override)
        for layer in self.layers:
            layer.override(new_override)

    def activate_memory(self):
        if MONITOR_RESETS_REVIVES:
            self.input_module.activate_memory()
            for layer in self.layers:
                layer.activate_memory()

    def review_and_shut_memory(self):
        if MONITOR_RESETS_REVIVES:
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
                    used_in_concepts = layer.review_and_shut_memory(list(range(layer.nb_out_concepts)), also_unused_in_concepts)
                else:
                    if not USE_RULE_MODULE or i > 0:
                        used_in_concepts = layer.review_and_shut_memory(used_in_concepts, also_unused_in_concepts)
                    else:
                        used_rules = used_in_concepts
                        layer.review_and_shut_memory(used_rules, [], do_check_in_concepts=False)
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
                self.input_module.review_and_shut_memory(used_in_concepts)

    def forward(self, x):
        """Forward pass"""
        x = self.input_module(x)
        result = self.layers(x)
        # print("LNN -> "+str(result.shape))
        return result

    def add_regularization(self, loss):
        loss = self.input_module.add_regularization(loss)
        loss = self.layers[0].add_regularization(loss)
        for layer in self.layers[1:]:
            loss = layer.add_regularization(loss)
        return loss

    def get_discrete_continuous_parameters(self):
        discrete_parameters, continuous_parameters = self.input_module.get_discrete_continuous_parameters()
        for layer in self.layers:
            layer_discrete_parameters, layer_continuous_parameters = layer.get_discrete_continuous_parameters()
            discrete_parameters += layer_discrete_parameters
            continuous_parameters += layer_continuous_parameters
        return discrete_parameters, continuous_parameters

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
            for out_OR_concept_idx in range(self.nb_out_concepts):
                single_OR_copy = copy.deepcopy(self)
                single_OR_copy.feature_names = single_OR_copy.feature_names[: single_OR_copy.nb_in_concepts] + [
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
                # text_fixed_size(ax, x, y, "V", flip_vert=flip_vert, bold=True)
                if not flip_vert:
                    text_fixed_size(ax, x - 1, y + 1.5, r"$ \vee $", h=16)
                else:
                    text_fixed_size(ax, x - 1, y + 1.5, r"$ \wedge $", h=16)
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
            x_unobs_buffer = 10
            y_text_line = 12 * 1.25

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
                    input_module_group_widths.append(
                        [
                            category_module.nb_in_concepts,
                            category_module.nb_out_concepts,
                        ]
                    )
                else:
                    input_categories_used_in_values.append(torch.nonzero(category_module.observed_concepts.data[0, :]).view(-1).tolist())
                    input_module_group_widths.append([len(input_categories_used_in_values[-1]), 1])
            for continuous_module in self.input_module.continuous_modules:
                if not is_single_rule:
                    input_module_group_widths.append(
                        [
                            continuous_module.dichotomies.nb_dichotomies,
                            continuous_module.intervals.nb_out_concepts,
                            continuous_module.out_concepts.nb_out_concepts,
                        ]
                    )
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
                layer_widths_with_spacing = [layer_width + nb_variables - 1]
            else:
                layer_widths_with_spacing = [nb_variables]
            for layer in self.layers:
                layer_widths_with_spacing.append(layer.nb_out_concepts)
            max_layer_width_with_spacing = max(layer_widths_with_spacing)

            # Find output indices
            output_target_idcs = list(range(len(self.feature_names)))
            for idx in self.binary_indices:
                output_target_idcs.remove(idx)
            for first_idx, last_idx in self.category_first_last_pairs:
                for idx in range(first_idx, last_idx + 1):
                    output_target_idcs.remove(idx)
            for idx, min_value, max_value in self.continuous_index_min_max_triples:
                output_target_idcs.remove(idx)
            for idx, period in self.periodic_index_period_pairs:
                output_target_idcs.remove(idx)

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
                width += x_unobs_buffer
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
                first_idx, last_idx = self.input_module.category_first_last_pairs[i]
                current_x = 0.5 * x_step
                loc_current_y = current_y - input_y_step * (max(input_module_group_widths[group_idx]) - 1) / 2
                text_fixed_size(
                    ax,
                    current_x,
                    loc_current_y,
                    self.feature_names[first_idx].split("_")[0],
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
                            self.feature_names[idx].split("_")[1],
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
                current_y -= input_y_step * (max(input_module_group_widths[group_idx]) + 1)
                group_idx += 1

            # Continuous Inputs
            for i, continuous_module in enumerate(self.input_module.continuous_modules):
                idx, min_value, max_value = self.input_module.continuous_index_min_max_triples[i]
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
                    (
                        continuous_module_first_idx,
                        continuous_module_stop_idx,
                    ) = continuous_copy.layers[
                        -1
                    ].in_concepts_group_first_stop_pairs[len(self.input_module.category_modules) + i]
                    continuous_copy.layers[-1].observed_concepts.data[:, :continuous_module_first_idx] = 0
                    continuous_copy.layers[-1].observed_concepts.data[:, continuous_module_stop_idx:] = 0
                    continuous_copy.layers[-1].unobserved_concepts.data[:] = 1
                    continuous_copy.simplify()
                    continuous_copy.input_module.continuous_index_min_max_triples = [(0, min_value, max_value)]
                    continuous_copy.eval()
                    torch_plot_y_coordinates = continuous_copy.forward(torch_plot_x_coordinates)
                    axins3.plot(
                        plot_x_coordinates,
                        torch_plot_y_coordinates.view(-1).cpu().tolist(),
                    )
                    axins3.set_xlim([min_value, max_value])
                    axins3.set_xticks(continuous_module.dichotomies.boundaries.data.view(-1).tolist())
                    axins3.set_ylim([-0.01, 1.02])
                    axins3.set_yticks([0, 1])
                    current_x += 2.5 * x_step
                    coordinates[-1].append((current_x, loc_current_y))
                else:
                    group_coordinates = [[]]
                    current_x += x_step
                    if input_module_group_widths[group_idx][0] == max(input_module_group_widths[group_idx]):
                        loc_current_y = current_y
                    else:
                        loc_current_y = current_y - input_y_step * ((max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][0]) / 2
                    for dichotomy in range(continuous_module.dichotomies.nb_dichotomies):
                        dichotomy_string = "approx. > " if continuous_module.dichotomies.sharpnesses.data[dichotomy].item() < 2 else "exactly > "
                        dichotomy_string += str(
                            round(
                                continuous_module.dichotomies.boundaries.data[dichotomy].item(),
                                1,
                            )
                        )
                        text_fixed_size(ax, current_x, loc_current_y, dichotomy_string)
                        group_coordinates[-1].append((current_x + 0.5 * x_step, loc_current_y))
                        if input_module_group_widths[group_idx][0] == max(input_module_group_widths[group_idx]):
                            loc_current_y -= input_y_step
                        else:
                            loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][0]
                    group_coordinates.append([])
                    current_x += x_step
                    if input_module_group_widths[group_idx][1] == max(input_module_group_widths[group_idx]):
                        loc_current_y = current_y
                    else:
                        loc_current_y = current_y - input_y_step * ((max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][1]) / 2
                    for interval in range(continuous_module.intervals.nb_out_concepts):
                        AND_node(ax, current_x, loc_current_y)
                        group_coordinates[-1].append((current_x, loc_current_y))
                        if input_module_group_widths[group_idx][1] == max(input_module_group_widths[group_idx]):
                            loc_current_y -= input_y_step
                        else:
                            loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][1]
                    for in_concept in range(continuous_module.intervals.nb_in_concepts):
                        for out_concept in range(continuous_module.intervals.nb_out_concepts):
                            weight = continuous_module.intervals.observed_concepts.data[out_concept, in_concept].item()
                            if weight < 0:
                                arrow(
                                    ax,
                                    group_coordinates[-2][in_concept][0],
                                    group_coordinates[-2][in_concept][1],
                                    group_coordinates[-1][out_concept][0] - x_node_buffer,
                                    group_coordinates[-1][out_concept][1],
                                    color="r",
                                    alpha=abs(weight),
                                )
                    for in_concept in range(continuous_module.intervals.nb_in_concepts):
                        for out_concept in range(continuous_module.intervals.nb_out_concepts):
                            weight = continuous_module.intervals.observed_concepts.data[out_concept, in_concept].item()
                            if weight > 0:
                                arrow(
                                    ax,
                                    group_coordinates[-2][in_concept][0],
                                    group_coordinates[-2][in_concept][1],
                                    group_coordinates[-1][out_concept][0] - x_node_buffer,
                                    group_coordinates[-1][out_concept][1],
                                    color="k",
                                    alpha=weight,
                                )
                    group_coordinates.append([])
                    current_x += 0.5 * x_step
                    if input_module_group_widths[group_idx][2] == max(input_module_group_widths[group_idx]):
                        loc_current_y = current_y
                    else:
                        loc_current_y = current_y - input_y_step * ((max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][2]) / 2
                    for out_concept in range(continuous_module.out_concepts.nb_out_concepts):
                        OR_node(ax, current_x, loc_current_y)
                        group_coordinates[-1].append((current_x, loc_current_y))
                        coordinates[-1].append((current_x, loc_current_y))
                        if input_module_group_widths[group_idx][2] == max(input_module_group_widths[group_idx]):
                            loc_current_y -= input_y_step
                        else:
                            loc_current_y -= input_y_step * (max(input_module_group_widths[group_idx]) - 1) / input_module_group_widths[group_idx][2]
                    for in_concept in range(continuous_module.out_concepts.nb_in_concepts):
                        for out_concept in range(continuous_module.out_concepts.nb_out_concepts):
                            weight = continuous_module.out_concepts.observed_concepts.data[out_concept, in_concept].item()
                            if weight > 0:
                                arrow(
                                    ax,
                                    group_coordinates[-2][in_concept][0] + x_node_buffer,
                                    group_coordinates[-2][in_concept][1],
                                    group_coordinates[-1][out_concept][0] - x_node_buffer,
                                    group_coordinates[-1][out_concept][1],
                                    color="k",
                                    alpha=weight,
                                )
                current_y -= input_y_step * (max(input_module_group_widths[group_idx]) + 1)
                group_idx += 1

            # Periodic Inputs
            for idx, period in self.input_module.periodic_index_period_pairs:
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
                in_has_unobs = i > 0 and self.layers[i - 1].use_unobserved
                if is_single_rule:
                    max_bin_cat_idx = len(self.input_module.binary_indices) + sum([category_module.nb_out_concepts for category_module in self.input_module.category_modules])
                for in_concept in range(layer.nb_in_concepts):
                    if not is_single_rule or in_concept < max_bin_cat_idx:
                        for out_concept in range(layer.nb_out_concepts):
                            weight = layer.observed_concepts.data[out_concept, in_concept].item()
                            if weight < 0:
                                if in_has_unobs:
                                    arrow(
                                        ax,
                                        coordinates[-2][in_concept][0] + x_node_buffer + x_unobs_buffer,
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
                            weight = layer.observed_concepts.data[out_concept, in_concept].item()
                            if weight > 0:
                                if in_has_unobs:
                                    arrow(
                                        ax,
                                        coordinates[-2][in_concept][0] + x_node_buffer + x_unobs_buffer,
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
                current_x += x_unobs_buffer
            if layer_widths_with_spacing[-1] == max_layer_width_with_spacing:
                current_y = height - y_heading - 0.5 * y_step
            else:
                current_y = height - y_heading - 0.5 * y_step - y_step * ((max_layer_width_with_spacing - 1) / layer_widths_with_spacing[-1]) / 2
            for idx in output_target_idcs:
                if is_single_rule:
                    arrow(
                        ax,
                        coordinates[-1][0][0] + x_node_buffer + x_unobs_buffer,
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

            if fig_ax == -1:
                if filename == "":
                    plt.show()
                else:
                    plt.pause(1e-8)
                    plt.ioff()
                    plt.savefig(filename + ".png", dpi=save_dpi)
                    plt.close("all")

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

    def simplify(self, can_retrain=False):
        self._backward_simplify()
        self._forward_simplify()
        must_retrain = self.remove_duplicates(can_retrain)
        self.layers[0].in_concepts_group_first_stop_pairs = self.input_module.get_out_concepts_group_first_stop_pairs()
        if can_retrain:
            if must_retrain:
                self = NeuralLogicNetwork(
                    nb_in_concepts=self.nb_in_concepts,
                    nb_out_concepts=self.nb_out_concepts,
                    nb_hidden_layers=len(self.layers) - 1,
                    nb_concepts_per_hidden_layer=self.nb_concepts_per_hidden_layer,
                    last_layer_is_OR_no_neg=(not self.layers[-1].is_AND and not self.layers[-1].use_negation),
                    category_first_last_pairs=self.category_first_last_pairs,
                    continuous_index_min_max_triples=self.continuous_index_min_max_triples,
                    periodic_index_period_pairs=self.periodic_index_period_pairs,
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

    def create_simplifed_copy(self):
        simplified_model = copy.deepcopy(self)
        simplified_model.simplify()
        return simplified_model

    def prune(self, eval_model_func, init_loss=-1, filename="", do_log=True, prune_log_files=[], progress_bar_hook=lambda iteration, nb_iterations: None, init_weight=0):
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
            init_loss = self.prune(eval_model_func, init_loss=init_loss, prune_log_files=prune_log_files, progress_bar_hook=progress_bar_hook, init_weight=total_nb_weights)
        else:
            progress_bar_hook(init_weight + total_nb_weights, init_weight + total_nb_weights)
            if filename != "":
                print_log("\n" + str(self), False, prune_log_files)
                close_log_files(prune_log_files)
        return init_loss

    def _get_nb_weights(self):
        nb_weights = self.input_module.get_nb_weights()
        for layer in self.layers:
            nb_weights += layer.get_nb_weights()
        return nb_weights

    def quantize(self):
        self.input_module.quantize()
        for layer in self.layers:
            layer.quantize()
        self.simplify()

    def new_quantize(
        self, eval_model_func, quantize_type="sub", goes_backward=True, goes_by_layer=False, filename="", do_log=True, progress_bar_hook=lambda iteration, nb_iterations: None
    ):
        quantize_log_files = get_log_files(filename, do_log)
        losses = [eval_model_func()]
        total_nb_weights = self._get_nb_weights()
        current_weight = 0
        if goes_backward:
            for i, layer in enumerate(reversed(self.layers)):
                losses += layer.new_quantize(
                    eval_model_func,
                    quantize_type=quantize_type,
                    goes_by_layer=goes_by_layer,
                    log_files=quantize_log_files,
                    progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight, total_nb_weights + 1),
                )
                self.simplify()
                total_nb_weights = self._get_nb_weights()
                current_weight = sum([layer.get_nb_weights() for j, layer in enumerate(reversed(self.layers)) if j <= i])
            losses += self.input_module.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_backward=goes_backward,
                goes_by_layer=goes_by_layer,
                log_files=quantize_log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight, total_nb_weights + 1),
            )
            self.simplify()
        else:
            losses += self.input_module.new_quantize(
                eval_model_func,
                quantize_type=quantize_type,
                goes_backward=goes_backward,
                goes_by_layer=goes_by_layer,
                log_files=quantize_log_files,
                progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight, total_nb_weights + 1),
            )
            self.simplify()
            total_nb_weights = self._get_nb_weights()
            current_weight = self.input_module.get_nb_weights()
            for i, layer in enumerate(self.layers):
                losses += layer.new_quantize(
                    eval_model_func,
                    quantize_type=quantize_type,
                    goes_by_layer=goes_by_layer,
                    log_files=quantize_log_files,
                    progress_bar_hook=lambda loc_weight: progress_bar_hook(current_weight + loc_weight, total_nb_weights + 1),
                )
                self.simplify()
                total_nb_weights = self._get_nb_weights()
                current_weight = self.input_module.get_nb_weights() + sum([layer.get_nb_weights() for j, layer in enumerate(self.layers) if j <= i])
        total_nb_weights = max(1, self._get_nb_weights())
        progress_bar_hook(total_nb_weights, total_nb_weights)
        print_log("\n" + str(self), False, quantize_log_files)
        close_log_files(quantize_log_files)
        return losses

    def rearrange_for_visibility(self):
        self._simplify_category_modules()
        self._reorder_lexicographically_and_by_unobserved()

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
                        new_category_OR_nodes_used_out_concepts.append(
                            (
                                [layer0_AND_concept_idx],
                                used_positively_in_concepts,
                                used_negatively_in_concepts,
                            )
                        )
            new_category_module_observed_concepts = torch.zeros(
                (
                    len(new_category_OR_nodes_used_out_concepts),
                    category_module.nb_in_concepts,
                ),
                device=category_module.device,
            )
            new_layer0_AND_missing_columns = torch.zeros(
                (
                    self.layers[0].nb_out_concepts,
                    len(new_category_OR_nodes_used_out_concepts),
                ),
                device=self.layers[0].observed_concepts.data.device,
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
                for used_positively_in_concept in used_positively_in_concepts:
                    new_category_OR_node_in_concepts = new_category_OR_node_in_concepts & set(
                        torch.nonzero(category_module.observed_concepts.data[used_positively_in_concept, :]).view(-1).tolist()
                    )
                for used_negatively_in_concept in used_negatively_in_concepts:
                    new_category_OR_node_in_concepts = new_category_OR_node_in_concepts - set(
                        torch.nonzero(category_module.observed_concepts.data[used_negatively_in_concept, :]).view(-1).tolist()
                    )
                new_category_OR_node_is_used_positively = len(new_category_OR_node_in_concepts) <= category_module.nb_in_concepts - len(new_category_OR_node_in_concepts)
                if new_category_OR_node_is_used_positively:
                    for new_category_OR_node_in_concept in new_category_OR_node_in_concepts:
                        new_category_module_observed_concepts[new_category_OR_node_idx, new_category_OR_node_in_concept] = 1
                else:
                    for new_category_OR_node_in_concept in set(range(category_module.nb_in_concepts)) - new_category_OR_node_in_concepts:
                        new_category_module_observed_concepts[new_category_OR_node_idx, new_category_OR_node_in_concept] = 1
                duplicate_category_OR_node_idx = None
                for possible_duplicate_category_OR_node_idx in range(new_category_OR_node_idx):
                    if (
                        new_category_module_observed_concepts[new_category_OR_node_idx, :] == new_category_module_observed_concepts[possible_duplicate_category_OR_node_idx, :]
                    ).all():
                        duplicate_category_OR_node_idx = possible_duplicate_category_OR_node_idx
                        duplicate_category_sign_matches = 1
                        break
                    elif (
                        new_category_module_observed_concepts[new_category_OR_node_idx, :] != new_category_module_observed_concepts[possible_duplicate_category_OR_node_idx, :]
                    ).all():
                        duplicate_category_OR_node_idx = possible_duplicate_category_OR_node_idx
                        duplicate_category_sign_matches = -1
                        break
                if duplicate_category_OR_node_idx == None:
                    for layer0_AND_concept_idx in applicable_layer0_AND_concepts:
                        new_layer0_AND_missing_columns[layer0_AND_concept_idx, new_category_OR_node_idx] = 1 if new_category_OR_node_is_used_positively else -1
                else:
                    new_category_module_observed_concepts[new_category_OR_node_idx, :] = 0
                    for layer0_AND_concept_idx in applicable_layer0_AND_concepts:
                        new_layer0_AND_missing_columns[layer0_AND_concept_idx, duplicate_category_OR_node_idx] = (
                            duplicate_category_sign_matches if new_category_OR_node_is_used_positively else -1 * duplicate_category_sign_matches
                        )
            if self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][1] < self.layers[0].nb_in_concepts:
                self.layers[0].observed_concepts.data = torch.cat(
                    (
                        self.layers[0].observed_concepts.data[
                            :,
                            : self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][0],
                        ],
                        new_layer0_AND_missing_columns,
                        self.layers[0].observed_concepts.data[
                            :,
                            self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][1] :,
                        ],
                    ),
                    dim=1,
                )
            else:
                self.layers[0].observed_concepts.data = torch.cat(
                    (
                        self.layers[0].observed_concepts.data[
                            :,
                            : self.layers[0].in_concepts_group_first_stop_pairs[category_module_idx][0],
                        ],
                        new_layer0_AND_missing_columns,
                    ),
                    dim=1,
                )
            category_module.observed_concepts.data = new_category_module_observed_concepts
            category_module.nb_out_concepts = category_module.observed_concepts.data.size(0)
            self.layers[0].nb_in_concepts = self.layers[0].observed_concepts.data.size(1)
            self.layers[0].in_concepts_group_first_stop_pairs = self.input_module.get_out_concepts_group_first_stop_pairs()
        self.simplify()

    def _reorder_lexicographically_and_by_unobserved(self):
        reordered_input_module_out_idcs = self.input_module.reorder_lexicographically()
        self.layers[0].observed_concepts.data = self.layers[0].observed_concepts.data[:, reordered_input_module_out_idcs]
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < len(self.layers) - 1:
                reordered_layer_out_idcs = find_lexicographical_order(layer.observed_concepts.data)
                layer.observed_concepts.data = layer.observed_concepts.data[reordered_layer_out_idcs, :]
                self.layers[layer_idx + 1].observed_concepts.data = self.layers[layer_idx + 1].observed_concepts.data[:, reordered_layer_out_idcs]
                if layer.use_unobserved:
                    layer.unobserved_concepts.data = layer.unobserved_concepts.data[reordered_layer_out_idcs]

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
        for first_idx, last_idx in self.category_first_last_pairs:
            for idx in range(first_idx, last_idx + 1):
                output_target_idcs.remove(idx)
        for idx, min_value, max_value in self.continuous_index_min_max_triples:
            output_target_idcs.remove(idx)
        for idx, period in self.periodic_index_period_pairs:
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
