from math import prod
import pickle
import random
from statistics import mean
import time
from typing import Tuple, Union
import warnings
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryROC
from NLN_Dataset import NLNTabularDataset
from NLN_Logging import close_log_files, get_log_files, print_log, gpu_mem_to_string, printProgressBar
from NLN_Modules import (
    NB_RULES,
    DISCRETIZATION_METHOD,
    NB_DICHOTOMIES_PER_CONTINUOUS,
    RANDOM_INIT_OBS,
    RANDOM_INIT_UNOBS,
    EMPTY_INIT_TARGETS,
    EMPTY_RESET_IN_CONCEPTS,
    TRAIN_FORW_WEIGHT_QUANT,
    APPROX_PARAMS,
    VERBOSE,
    DEVICE,
    NeuralLogicNetwork,
)

USE_TRAIN_VAL_SPLIT = True
MIN_NB_TRAINING_STEPS_BEFORE_REVIEW = 8


class NLN:
    """Neural Logic Network (NLN) Wrapper"""

    def __init__(
        self,
        NLN_filename: str,
        train_dataset: NLNTabularDataset,
        val_dataset: Union[NLNTabularDataset, None] = None,
        use_train_val_split: bool = USE_TRAIN_VAL_SPLIT,
        test_dataset: Union[NLNTabularDataset, None] = None,
        criterion: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        nb_rules: int = NB_RULES,
        train_forw_weight_quant: str = TRAIN_FORW_WEIGHT_QUANT,
        approx_AND_OR_params: Union[None, Tuple[float, float, float]] = APPROX_PARAMS,
        adam_lr: float = 1e-3,
        train_nb_epochs: int = 3000,
        retrain_nb_epochs: int = 1000,
        do_train: bool = True,
        do_discretize: bool = True,
        do_retrain_continuous_together: bool = False,
        do_retrain_dichotomies: bool = True,
        do_retrain_unobserved_concepts: bool = True,
        do_prune: bool = True,
        do_analyze_dataset_coverage_and_adjust_biases: bool = True,
        do_reorder: bool = True,
        do_save_final_plots: bool = True,
        do_evaluate_on_train_val_dataset: bool = True,
        do_plot_stats_train_val_dataset: bool = True,
        do_evaluate_on_full_dataset: bool = True,
        do_plot_stats_full_dataset: bool = True,
        do_evaluate_on_test_dataset: bool = True,
        do_plot_stats_test_dataset: bool = True,
        discretization_method: str = DISCRETIZATION_METHOD,
        nb_hidden_layers: int = 1,
        last_layer_is_OR_no_neg: bool = True,
        nb_dichotomies_per_continuous: int = NB_DICHOTOMIES_PER_CONTINUOUS,
        nb_intervals_per_continuous: Union[int, None] = None,
        random_init_obs: bool = RANDOM_INIT_OBS,
        random_init_unobs: bool = RANDOM_INIT_UNOBS,
        empty_init_targets: bool = EMPTY_INIT_TARGETS,
        empty_reset_in_concepts: bool = EMPTY_RESET_IN_CONCEPTS,
        device=DEVICE,
        verbose: bool = VERBOSE,
        do_log: bool = False,
        do_save_intermediate_learning_model_plots: bool = False,
        do_save_intermediate_training_model_plots: bool = False,
        do_save_training_progression_plots: bool = False,
        log_model_every_training_epoch: bool = False,
        init_string="",
    ):
        self.filename = NLN_filename
        self.use_train_val_split = use_train_val_split
        if val_dataset != None:
            if use_train_val_split:
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset
                self.train_val_dataset = NLNTabularDataset.merge([train_dataset, val_dataset])
            else:
                self.train_val_dataset = NLNTabularDataset.merge([train_dataset, val_dataset])
                self.train_dataset = self.train_val_dataset
                self.val_dataset = self.train_val_dataset
        else:
            if use_train_val_split:
                self.train_val_dataset = train_dataset
                self.train_dataset, self.val_dataset = NLNTabularDataset.split_into_train_val(train_dataset, random_seed=None)
            else:
                self.train_val_dataset = train_dataset
                self.train_dataset = train_dataset
                self.val_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=0)
        self.train_val_loader = DataLoader(self.train_val_dataset, batch_size=128, shuffle=False, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=0)
        self.test_loader = None if self.test_dataset == None else DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=0)
        self.criterion = criterion
        if train_forw_weight_quant not in ["", "thresh", "stoch", "sthresh"]:
            raise Exception(
                'Undefined train_forw_weight_quant. The only possible weight discretization methods in the forward pass of the gradient grafting are\n\t"" (no weight discretization);\n\t"thresh" (threshold discretization): sign(w) if |w| >= 0.5 else 0;\n\t"stoch" (stochastic discretization): sign(w) with prob |w| else 0;\n\t"sthresh" (stochastic threshold discretization): sign(w) if |w| >= thresh else 0 where thresh is sampled in [0,1[.'
            )

        self.adam_lr = adam_lr
        self.train_nb_epochs = train_nb_epochs
        self.retrain_nb_epochs = retrain_nb_epochs
        self.do_train = do_train
        self.do_discretize = do_discretize
        self.do_retrain_continuous_together = do_retrain_continuous_together
        self.do_retrain_dichotomies = do_retrain_dichotomies
        self.do_retrain_unobserved_concepts = do_retrain_unobserved_concepts
        self.do_prune = do_prune
        self.do_analyze_dataset_coverage_and_adjust_biases = do_analyze_dataset_coverage_and_adjust_biases
        self.do_reorder = do_reorder
        self.do_save_final_plots = do_save_final_plots
        self.do_evaluate_on_train_val_dataset = do_evaluate_on_train_val_dataset
        self.do_plot_stats_train_val_dataset = do_plot_stats_train_val_dataset
        self.do_evaluate_on_full_dataset = do_evaluate_on_full_dataset
        self.do_plot_stats_full_dataset = do_plot_stats_full_dataset
        self.do_evaluate_on_test_dataset = do_evaluate_on_test_dataset and test_dataset != None
        self.do_plot_stats_test_dataset = do_plot_stats_test_dataset and test_dataset != None
        self.has_trained = False
        self.has_discretized = False
        self.has_retrained_continuous_together = False
        self.has_retrained_dichotomies = False
        self.has_retrained_unobserved_concepts = False
        self.has_pruned = False
        self.has_analyzed_dataset_coverage_and_adjusted_biases = False
        self.has_reordered = False
        self.has_saved_final_plots = False
        self.has_evaluated_and_or_plotted_stats = False
        self.has_learned = False
        if discretization_method not in ["sel_desc", "sel_asc", "sub", "add", "thresh", "stoch", "qthresh"]:
            raise Exception(
                'Undefined discretization_method. The only possible discretization_methods are\n\t"sel_desc" (selective, descending);\n\t"sel_asc" (selective, ascending);\n\t"sub" (subtractive); \n\t"add" (additive);\n\t"thresh" (threshold);\n\t"stoch" (stochastic);\n\t"qthresh" (quantile threshold).'
            )
        self.discretization_method = discretization_method
        self.verbose = verbose
        self.do_log = do_log
        self.do_save_intermediate_learning_model_plots = do_save_intermediate_learning_model_plots
        self.do_save_intermediate_training_model_plots = do_save_intermediate_training_model_plots
        self.do_save_training_progression_plots = do_save_training_progression_plots
        self.log_model_every_training_epoch = log_model_every_training_epoch
        self.torch_module = NeuralLogicNetwork(
            train_dataset.nb_features,
            train_dataset.nb_class_targets,
            nb_concepts_per_hidden_layer=nb_rules,
            nb_hidden_layers=nb_hidden_layers,
            last_layer_is_OR_no_neg=last_layer_is_OR_no_neg,
            train_forw_weight_quant=train_forw_weight_quant,
            approx_AND_OR_params=approx_AND_OR_params,
            category_first_last_has_missing_values_tuples=train_dataset.category_first_last_has_missing_values_tuples,
            continuous_index_min_max_has_missing_values_tuples=train_dataset.continuous_index_min_max_has_missing_values_tuples,
            periodic_index_period_has_missing_values_tuples=train_dataset.periodic_index_period_has_missing_values_tuples,
            column_names=train_dataset.column_names,
            nb_dichotomies_per_continuous=nb_dichotomies_per_continuous,
            nb_intervals_per_continuous=nb_intervals_per_continuous,
            random_init_obs=random_init_obs,
            random_init_unobs=random_init_unobs,
            empty_init_targets=empty_init_targets,
            empty_reset_in_concepts=empty_reset_in_concepts,
            device=device,
            verbose=verbose,
            init_string=init_string,
        )

    def set(
        self,
        filename=None,
        use_train_val_split=None,
        train_dataset=None,
        val_dataset=None,
        train_val_dataset=None,
        test_dataset="",
        criterion=None,
        adam_lr=None,
        train_nb_epochs=None,
        retrain_nb_epochs=None,
        do_train=None,
        do_discretize=None,
        do_retrain_continuous_together=None,
        do_retrain_dichotomies=None,
        do_retrain_unobserved_concepts=None,
        do_prune=None,
        do_analyze_dataset_coverage_and_adjust_biases=None,
        do_reorder=None,
        do_save_final_plots=None,
        do_evaluate_on_train_val_dataset=None,
        do_plot_stats_train_val_dataset=None,
        do_evaluate_on_full_dataset=None,
        do_plot_stats_full_dataset=None,
        do_evaluate_on_test_dataset=None,
        do_plot_stats_test_dataset=None,
        has_trained=None,
        has_discretized=None,
        has_retrained_continuous_together=None,
        has_retrained_dichotomies=None,
        has_retrained_unobserved_concepts=None,
        has_pruned=None,
        has_self_analyzed_and_updated=None,
        has_reordered=None,
        has_saved_final_plots=None,
        has_evaluated_and_or_plotted_stats=None,
        has_learned=None,
        verbose=None,
        do_log=None,
        do_save_intermediate_learning_model_plots=None,
        do_save_intermediate_training_model_plots=None,
        do_save_training_progression_plots=None,
        log_model_every_training_epoch=None,
        discretization_method=None,
        nb_rules=None,
        nb_hidden_layers=None,
        last_layer_is_OR_no_neg=None,
        train_forw_weight_quant=None,
        approx_AND_OR_params="",
        nb_dichotomies_per_continuous=None,
        nb_intervals_per_continuous="",
        random_init_obs=None,
        random_init_unobs=None,
        empty_init_targets=None,
        empty_reset_in_concepts=None,
        device=None,
    ):
        if train_dataset != None or val_dataset != None or train_val_dataset != None:
            if train_dataset != None:
                if val_dataset != None:
                    if use_train_val_split:
                        self.train_dataset = train_dataset
                        self.val_dataset = val_dataset
                        self.train_val_dataset = NLNTabularDataset.merge([train_dataset, val_dataset])
                    else:
                        self.train_val_dataset = NLNTabularDataset.merge([train_dataset, val_dataset])
                        self.train_dataset = self.train_val_dataset
                        self.val_dataset = self.train_val_dataset
                else:
                    if use_train_val_split:
                        self.train_val_dataset = train_dataset
                        self.train_dataset, self.val_dataset = NLNTabularDataset.split_into_train_val(train_dataset, random_seed=None)
                    else:
                        self.train_val_dataset = train_dataset
                        self.train_dataset = train_dataset
                        self.val_dataset = train_dataset
            elif val_dataset != None:
                if use_train_val_split:
                    self.val_dataset = val_dataset
                    self.train_val_dataset = NLNTabularDataset.merge([self.train_dataset, val_dataset])
                else:
                    self.train_val_dataset = NLNTabularDataset.merge([self.train_dataset, val_dataset])
                    self.train_dataset = self.train_val_dataset
                    self.val_dataset = self.train_val_dataset
            elif train_val_dataset != None:
                self.train_val_dataset = train_val_dataset
                if use_train_val_split:
                    self.train_dataset, self.val_dataset = NLNTabularDataset.split_into_train_val(train_val_dataset, random_seed=None)
                else:
                    self.train_dataset = train_val_dataset
                    self.val_dataset = train_val_dataset
            self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=0)
            self.train_val_loader = DataLoader(self.train_val_dataset, batch_size=128, shuffle=False, num_workers=0)
            self.val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=0)
            has_trained = False
        if use_train_val_split != None and use_train_val_split != self.use_train_val_split:
            self.use_train_val_split = use_train_val_split
            if use_train_val_split:
                self.train_dataset, self.val_dataset = NLNTabularDataset.split_into_train_val(self.train_val_dataset, random_seed=None)
            else:
                self.train_dataset = self.train_val_dataset
                self.val_dataset = self.train_val_dataset
            self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=0)
            self.val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=0)
            has_trained = False
        if test_dataset != "":
            self.test_dataset = test_dataset
            self.test_loader = None if self.test_dataset == None else DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=0)
            self.do_evaluate_on_test_dataset = self.do_evaluate_on_test_dataset and test_dataset != None
            self.do_plot_stats_test_dataset = self.do_plot_stats_test_dataset and test_dataset != None
            has_evaluated_and_or_plotted_stats = False

        if criterion != None:
            self.criterion = criterion
            has_trained = False

        if adam_lr != None and adam_lr != self.adam_lr:
            self.adam_lr = adam_lr
            has_trained = False

        if train_nb_epochs != None and train_nb_epochs != self.train_nb_epochs:
            self.train_nb_epochs = train_nb_epochs
            has_trained = False
        if retrain_nb_epochs != None and retrain_nb_epochs != self.retrain_nb_epochs:
            self.retrain_nb_epochs = retrain_nb_epochs
            has_trained = False

        if do_train != None and do_train != self.do_train:
            self.do_train = do_train
            has_trained = False
        if do_discretize != None and do_discretize != self.do_discretize:
            self.do_discretize = do_discretize
            has_discretized = False
        if do_retrain_continuous_together != None and do_retrain_continuous_together != self.do_retrain_continuous_together:
            self.do_retrain_continuous_together = do_retrain_continuous_together
            has_retrained_continuous_together = False
        if do_retrain_dichotomies != None and do_retrain_dichotomies != self.do_retrain_dichotomies:
            self.do_retrain_dichotomies = do_retrain_dichotomies
            has_retrained_dichotomies = False
        if do_retrain_unobserved_concepts != None and do_retrain_unobserved_concepts != self.do_retrain_unobserved_concepts:
            self.do_retrain_unobserved_concepts = do_retrain_unobserved_concepts
            has_retrained_unobserved_concepts = False
        if do_prune != None and do_prune != self.do_prune:
            self.do_prune = do_prune
            has_pruned = False
        if do_analyze_dataset_coverage_and_adjust_biases != None and do_analyze_dataset_coverage_and_adjust_biases != self.do_analyze_dataset_coverage_and_adjust_biases:
            self.do_analyze_dataset_coverage_and_adjust_biases = do_analyze_dataset_coverage_and_adjust_biases
            has_self_analyzed_and_updated = False
        if do_reorder != None and do_reorder != self.do_reorder:
            self.do_reorder = do_reorder
            has_reordered = False
        if do_save_final_plots != None and do_save_final_plots != self.do_save_final_plots:
            self.do_save_final_plots = do_save_final_plots
            has_saved_final_plots = False
        if do_evaluate_on_train_val_dataset != None and do_evaluate_on_train_val_dataset != self.do_evaluate_on_train_val_dataset:
            self.do_evaluate_on_train_val_dataset = do_evaluate_on_train_val_dataset
            has_evaluated_and_or_plotted_stats = False
        if do_plot_stats_train_val_dataset != None and do_plot_stats_train_val_dataset != self.do_plot_stats_train_val_dataset:
            self.do_plot_stats_train_val_dataset = do_plot_stats_train_val_dataset
            has_evaluated_and_or_plotted_stats = False
        if do_evaluate_on_full_dataset != None and do_evaluate_on_full_dataset != self.do_evaluate_on_full_dataset:
            self.do_evaluate_on_full_dataset = do_evaluate_on_full_dataset
            has_evaluated_and_or_plotted_stats = False
        if do_plot_stats_full_dataset != None and do_plot_stats_full_dataset != self.do_plot_stats_full_dataset:
            self.do_plot_stats_full_dataset = do_plot_stats_full_dataset
            has_evaluated_and_or_plotted_stats = False
        if do_evaluate_on_test_dataset != None and do_evaluate_on_test_dataset != self.do_evaluate_on_test_dataset:
            self.do_evaluate_on_test_dataset = do_evaluate_on_test_dataset and test_dataset != None
            has_evaluated_and_or_plotted_stats = False
        if do_plot_stats_test_dataset != None and do_plot_stats_test_dataset != self.do_plot_stats_test_dataset:
            self.do_plot_stats_test_dataset = do_plot_stats_test_dataset and test_dataset != None
            has_evaluated_and_or_plotted_stats = False

        if verbose != None:
            self.verbose = verbose
        if do_log != None:
            self.do_log = do_log
        if do_save_intermediate_learning_model_plots != None:
            self.do_save_intermediate_learning_model_plots = do_save_intermediate_learning_model_plots
        if do_save_intermediate_training_model_plots != None:
            self.do_save_intermediate_training_model_plots = do_save_intermediate_training_model_plots
        if do_save_training_progression_plots != None:
            self.do_save_training_progression_plots = do_save_training_progression_plots
        if log_model_every_training_epoch != None:
            self.log_model_every_training_epoch = log_model_every_training_epoch

        if discretization_method != None and discretization_method != self.discretization_method:
            self.discretization_method = discretization_method
            self.do_discretize = True
            has_discretized = False

        if (
            self.train_dataset.nb_features != self.torch_module.nb_in_concepts
            or self.train_dataset.nb_class_targets != self.torch_module.nb_out_concepts
            or [(first_idx, last_idx, has_missing_values) for first_idx, last_idx, has_missing_values in self.train_dataset.category_first_last_has_missing_values_tuples]
            != self.torch_module.category_first_last_has_missing_values_tuples
            or [
                (idx, min_value, max_value, has_missing_values)
                for idx, min_value, max_value, has_missing_values in self.train_dataset.continuous_index_min_max_has_missing_values_tuples
            ]
            != self.torch_module.continuous_index_min_max_has_missing_values_tuples
            or [(idx, period, has_missing_values) for idx, period, has_missing_values in self.train_dataset.periodic_index_period_has_missing_values_tuples]
            != self.torch_module.periodic_index_period_has_missing_values_tuples
            or self.train_dataset.column_names != self.torch_module.feature_names
            or (nb_rules != None and nb_rules != self.torch_module.nb_concepts_per_hidden_layer)
            or (nb_hidden_layers != None and nb_hidden_layers != self.torch_module.nb_hidden_layers)
            or (last_layer_is_OR_no_neg != None and last_layer_is_OR_no_neg != self.torch_module.last_layer_is_OR_no_neg)
            or (nb_dichotomies_per_continuous != None and nb_dichotomies_per_continuous != self.torch_module.nb_dichotomies_per_continuous)
            or (nb_intervals_per_continuous != "" and nb_intervals_per_continuous != self.torch_module.nb_intervals_per_continuous)
            or (random_init_obs != None and random_init_obs != self.torch_module.random_init_obs)
            or (random_init_unobs != None and random_init_unobs != self.torch_module.random_init_unobs)
            or (empty_init_targets != None and empty_init_targets != self.torch_module.empty_init_targets)
        ):
            self.torch_module = NeuralLogicNetwork(
                self.train_dataset.nb_features,
                self.train_dataset.nb_class_targets,
                nb_concepts_per_hidden_layer=nb_rules if nb_rules != None else self.torch_module.nb_concepts_per_hidden_layer,
                nb_hidden_layers=nb_hidden_layers if nb_hidden_layers != None else self.torch_module.nb_hidden_layers,
                last_layer_is_OR_no_neg=last_layer_is_OR_no_neg if last_layer_is_OR_no_neg != None else self.torch_module.last_layer_is_OR_no_neg,
                train_forw_weight_quant=train_forw_weight_quant if train_forw_weight_quant != None else self.torch_module.train_forw_weight_quant,
                approx_AND_OR_params=approx_AND_OR_params if approx_AND_OR_params != "" else self.torch_module.approx_AND_OR_params,
                category_first_last_has_missing_values_tuples=self.train_dataset.category_first_last_has_missing_values_tuples,
                continuous_index_min_max_has_missing_values_tuples=self.train_dataset.continuous_index_min_max_has_missing_values_tuples,
                periodic_index_period_has_missing_values_tuples=self.train_dataset.periodic_index_period_has_missing_values_tuples,
                column_names=self.train_dataset.column_names,
                nb_dichotomies_per_continuous=nb_dichotomies_per_continuous if nb_dichotomies_per_continuous != None else self.torch_module.nb_dichotomies_per_continuous,
                nb_intervals_per_continuous=nb_intervals_per_continuous if nb_intervals_per_continuous != "" else self.torch_module.nb_intervals_per_continuous,
                random_init_obs=random_init_obs if random_init_obs != None else self.torch_module.random_init_obs,
                random_init_unobs=random_init_unobs if random_init_unobs != None else self.torch_module.random_init_unobs,
                empty_init_targets=empty_init_targets if empty_init_targets != None else self.torch_module.empty_init_targets,
                empty_reset_in_concepts=empty_reset_in_concepts if empty_reset_in_concepts != None else self.torch_module.empty_reset_in_concepts,
                device=device if device != None else self.torch_module.device,
                verbose=verbose if verbose != None else self.verbose,
            )
            has_trained = False
        elif (
            (empty_reset_in_concepts != None and empty_reset_in_concepts != self.torch_module.empty_reset_in_concepts)
            or (train_forw_weight_quant != None and train_forw_weight_quant != self.torch_module.train_forw_weight_quant)
            or (approx_AND_OR_params != "" and approx_AND_OR_params != self.torch_module.approx_AND_OR_params)
            or (device != None and device != self.torch_module.device)
            or (verbose != None and verbose != self.verbose)
        ):
            self.torch_module.set(
                empty_reset_in_concepts=empty_reset_in_concepts,
                train_forw_weight_quant=train_forw_weight_quant,
                approx_AND_OR_params=approx_AND_OR_params,
                device=device,
                verbose=verbose,
            )
            has_trained = False

        if has_learned != None:
            self.has_learned = has_learned
            if not has_learned:
                has_trained = False
        if has_trained != None:
            self.has_trained = has_trained
            if not has_trained:
                has_discretized = False
        if has_discretized != None:
            self.has_discretized = has_discretized
            if not has_discretized:
                has_retrained_continuous_together = False
                has_retrained_dichotomies = False
                has_retrained_unobserved_concepts = False
        if has_retrained_continuous_together != None:
            self.has_retrained_continuous_together = has_retrained_continuous_together
            if not has_retrained_continuous_together:
                has_pruned = False
        if has_retrained_dichotomies != None:
            self.has_retrained_dichotomies = has_retrained_dichotomies
            if not has_retrained_dichotomies:
                has_pruned = False
        if has_retrained_unobserved_concepts != None:
            self.has_retrained_unobserved_concepts = has_retrained_unobserved_concepts
            if not has_retrained_unobserved_concepts:
                has_pruned = False
        if has_pruned != None:
            self.has_pruned = has_pruned
            if not has_pruned:
                has_self_analyzed_and_updated = False
        if has_self_analyzed_and_updated != None:
            self.has_analyzed_dataset_coverage_and_adjusted_biases = has_self_analyzed_and_updated
            if not has_self_analyzed_and_updated:
                has_reordered = False
        if has_reordered != None:
            self.has_reordered = has_reordered
            if not has_reordered:
                has_saved_final_plots = False
        if has_saved_final_plots != None:
            self.has_saved_final_plots = has_saved_final_plots
            if not has_saved_final_plots:
                has_evaluated_and_or_plotted_stats = False
        if has_evaluated_and_or_plotted_stats != None:
            self.has_evaluated_and_or_plotted_stats = has_evaluated_and_or_plotted_stats
            if not has_evaluated_and_or_plotted_stats:
                self.has_learned = False

        if filename != None and filename != self.filename:
            self.filename = filename
            self.save()

    def save(self, filename=""):
        if filename == "":
            filename = self.filename
        pickle.dump(self, open(filename + ".pkl", "wb"))

    @staticmethod
    def load(filename):
        if filename[-4:] == ".pkl":
            return pickle.load(open(filename, "rb"))
        else:
            return pickle.load(open(filename + ".pkl", "rb"))

    @staticmethod
    def merge_models(models, merged_model_filename):
        def get_most_used_value(values):
            return sorted([(value, values.count(value)) for value in set(values)], key=lambda pair: pair[1], reverse=True)[0][0]

        merged_torch_module = NeuralLogicNetwork.merge_modules([model.torch_module for model in models])

        merged_model = NLN(
            merged_model_filename,
            NLNTabularDataset.merge([model.train_dataset for model in models]),
            val_dataset=NLNTabularDataset.merge([model.val_dataset for model in models]),
            use_train_val_split=bool(round(mean([model.use_train_val_split for model in models]))),
            test_dataset=NLNTabularDataset.merge([model.test_dataset for model in models]),
            criterion=models[0].criterion,
            nb_rules=merged_torch_module.nb_concepts_per_hidden_layer,
            train_forw_weight_quant=merged_torch_module.train_forw_weight_quant,
            approx_AND_OR_params=merged_torch_module.approx_AND_OR_params,
            adam_lr=mean([model.adam_lr for model in models]),
            train_nb_epochs=int(round(mean([model.train_nb_epochs for model in models]))),
            retrain_nb_epochs=int(round(mean([model.retrain_nb_epochs for model in models]))),
            do_train=bool(prod([model.do_train for model in models])),
            do_discretize=bool(prod([model.do_discretize for model in models])),
            do_retrain_continuous_together=bool(prod([model.do_retrain_continuous_together for model in models])),
            do_retrain_dichotomies=bool(prod([model.do_retrain_dichotomies for model in models])),
            do_retrain_unobserved_concepts=bool(prod([model.do_retrain_unobserved_concepts for model in models])),
            do_prune=bool(prod([model.do_prune for model in models])),
            do_analyze_dataset_coverage_and_adjust_biases=bool(prod([model.do_analyze_dataset_coverage_and_adjust_biases for model in models])),
            do_reorder=bool(prod([model.do_reorder for model in models])),
            do_save_final_plots=bool(prod([model.do_save_final_plots for model in models])),
            do_evaluate_on_train_val_dataset=bool(prod([model.do_evaluate_on_train_val_dataset for model in models])),
            do_plot_stats_train_val_dataset=bool(prod([model.do_plot_stats_train_val_dataset for model in models])),
            do_evaluate_on_full_dataset=bool(prod([model.do_evaluate_on_full_dataset for model in models])),
            do_plot_stats_full_dataset=bool(prod([model.do_plot_stats_full_dataset for model in models])),
            do_evaluate_on_test_dataset=bool(prod([model.do_evaluate_on_test_dataset for model in models])),
            do_plot_stats_test_dataset=bool(prod([model.do_plot_stats_test_dataset for model in models])),
            discretization_method=get_most_used_value([model.discretization_method for model in models]),
            nb_hidden_layers=merged_torch_module.nb_hidden_layers,
            last_layer_is_OR_no_neg=models[0].torch_module.last_layer_is_OR_no_neg,
            nb_dichotomies_per_continuous=merged_torch_module.nb_dichotomies_per_continuous,
            nb_intervals_per_continuous=merged_torch_module.nb_intervals_per_continuous,
            random_init_obs=merged_torch_module.random_init_obs,
            random_init_unobs=merged_torch_module.random_init_unobs,
            empty_init_targets=merged_torch_module.empty_init_targets,
            empty_reset_in_concepts=merged_torch_module.empty_reset_in_concepts,
            device=merged_torch_module.device,
            verbose=merged_torch_module.verbose,
            do_log=bool(round(mean([model.do_log for model in models]))),
            do_save_intermediate_learning_model_plots=bool(round(mean([model.do_save_intermediate_learning_model_plots for model in models]))),
            do_save_intermediate_training_model_plots=bool(round(mean([model.do_save_intermediate_training_model_plots for model in models]))),
            do_save_training_progression_plots=bool(round(mean([model.do_save_training_progression_plots for model in models]))),
            log_model_every_training_epoch=bool(round(mean([model.log_model_every_training_epoch for model in models]))),
            init_string=str(merged_torch_module),
        )

        merged_model.has_trained = bool(prod([model.has_trained for model in models]))
        merged_model.has_discretized = bool(prod([model.has_discretized for model in models]))
        merged_model.has_retrained_continuous_together = bool(prod([model.has_retrained_continuous_together for model in models]))
        merged_model.has_retrained_dichotomies = bool(prod([model.has_retrained_dichotomies for model in models]))
        merged_model.has_retrained_unobserved_concepts = bool(prod([model.has_retrained_unobserved_concepts for model in models]))
        merged_model.has_pruned = False
        merged_model.has_analyzed_dataset_coverage_and_adjusted_biases = False
        merged_model.has_reordered = False
        merged_model.has_saved_final_plots = False
        merged_model.has_evaluated_and_or_plotted_stats = False
        merged_model.has_learned = False

        merged_model = merged_model.learn()

        return merged_model

    @staticmethod
    def fix_random_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def learn(self):
        if not self.has_learned:
            filename = self.filename
            print(f"Learning {filename}...")

            timing_files = get_log_files(filename + "_AllTimings", self.do_log)

            if self.do_train:
                filename += "_trained"
                if not self.has_trained:
                    if self.verbose:
                        print(f"\nInitial training...")
                    else:
                        print()
                        printProgressBar(0, self.train_nb_epochs, prefix=f"Initial training...  ", suffix="", length=50)
                    full_optimizer = torch.optim.Adam(self.torch_module.parameters(), lr=self.adam_lr)
                    start = time.time()
                    self.train_results = self._train(
                        full_optimizer,
                        self.train_nb_epochs,
                        True,
                        filename=filename,
                        use_best_val=True,  # False,
                        max_nb_epochs_since_last_best=500,
                        save_model_plots=self.do_save_intermediate_training_model_plots,
                        show_loss_epochs=self.do_save_training_progression_plots,
                        show_loss_time=self.do_save_training_progression_plots,
                    )
                    self.train_time = time.time() - start
                    print_log(f"Initial training time: {self.train_time:.3e} seconds", True, timing_files)
                    self.has_trained = True
                    self.save()

                    train_val_loss = self._eval_train_val_loss()
                    print(f"Full training set loss (train. + val.) after initial training = {train_val_loss:.5g}")
                    if self.do_save_intermediate_learning_model_plots:
                        print("Saving plot...")
                        self.show(filename=filename, valid_raw_loss=train_val_loss)

            post_process_start = time.time()

            if self.do_discretize:
                if self.discretization_method == "sub":
                    capitalized_discretization_method = "Sub"
                elif self.discretization_method == "add":
                    capitalized_discretization_method = "Add"
                elif self.discretization_method == "sel_asc":
                    capitalized_discretization_method = "SelAsc"
                elif self.discretization_method == "sel_desc":
                    capitalized_discretization_method = "SelDesc"
                elif self.discretization_method == "thresh":
                    capitalized_discretization_method = "Thresh"
                elif self.discretization_method[:5] == "stoch":
                    capitalized_discretization_method = f"Stoch{self.discretization_method[5:]}"
                elif self.discretization_method[:7] == "qthresh":
                    capitalized_discretization_method = f"QThresh{self.discretization_method[7:]}"
                filename += f"_discretized{capitalized_discretization_method}"
                if not self.has_discretized:
                    if self.verbose:
                        print(f"\nDiscretizing...")
                        progress_bar_hook = lambda iteration, nb_iterations: None
                    else:
                        print()
                        printProgressBar(0, 1, prefix=f"Discretizing...  ", suffix="", length=50)
                        progress_bar_hook = lambda iteration, nb_iterations: printProgressBar(iteration, nb_iterations, prefix=f"Discretizing...  ", suffix="", length=50)

                    start = time.time()
                    losses = self.torch_module.discretize(
                        self.discretization_method,
                        self._eval_train_val_loss,
                        self.save,
                        filename=filename,
                        do_log=self.do_log,
                        progress_bar_hook=progress_bar_hook,
                    )
                    self.discretize_time = time.time() - start

                    print_log(f"Discretization time: {self.discretize_time:.3e} seconds", True, timing_files)
                    self.has_discretized = True
                    self.save(self.filename)

                    train_val_loss = self._eval_train_val_loss()
                    print(f"Full training set loss (train. + val.) after discretization = {train_val_loss:.5g}")
                    if self.do_save_intermediate_learning_model_plots:
                        print("Saving plot...")
                        self.show(filename=filename, valid_raw_loss=train_val_loss)

            if self.do_retrain_continuous_together or self.do_retrain_dichotomies or self.do_retrain_unobserved_concepts:
                try:
                    train_val_loss
                except NameError:
                    train_val_loss = self._eval_train_val_loss()
            if self.do_retrain_continuous_together:
                if train_val_loss > 0 and len(self.train_dataset.continuous_index_min_max_has_missing_values_tuples) > 0:
                    filename += "_retrained"
                    if not self.has_retrained_continuous_together:
                        if self.verbose:
                            print(f"\nContinuous parameters retraining...")
                        else:
                            print()
                            printProgressBar(0, self.retrain_nb_epochs, prefix=f"Continuous parameters retraining...  ", suffix="", length=50)

                        if self.do_train or self.do_discretize:
                            self = NLN.load(self.filename)
                        dichotomies_parameters, observed_concepts, unobserved_concepts = self.torch_module.get_dichotomies_observed_unobserved_parameters()
                        continuous_optimizer = torch.optim.Adam(dichotomies_parameters + unobserved_concepts, lr=self.adam_lr)
                        start = time.time()
                        self.dichotomies_retrain_results = self._train(
                            continuous_optimizer,
                            self.retrain_nb_epochs,
                            False,
                            filename=filename,
                            use_best_val=True,
                            max_nb_epochs_since_last_best=100,
                            save_model_plots=False,
                            show_loss_epochs=self.do_save_training_progression_plots,
                        )
                        self.continuous_retrain_time = time.time() - start
                        print_log(f"Continuous retraining time: {self.continuous_retrain_time:.3e} seconds", True, timing_files)
                        self.has_retrained_continuous_together = True
                        self.save()

                        train_val_loss = self._eval_train_val_loss()
                        print(f"Full training set loss (train. + val.) after continuous retraining = {train_val_loss:.5g}")
                        if self.do_save_intermediate_learning_model_plots:
                            print("Saving plot...")
                            self.show(filename=filename, valid_raw_loss=train_val_loss)
            else:
                if self.do_retrain_dichotomies and train_val_loss > 0 and len(self.train_dataset.continuous_index_min_max_has_missing_values_tuples) > 0:
                    filename += "_dichRetrained"
                    if not self.has_retrained_dichotomies:
                        if self.verbose:
                            print(f"\nDichotomies retraining...")
                        else:
                            print()
                            printProgressBar(0, self.retrain_nb_epochs, prefix=f"Dichotomies retraining...  ", suffix="", length=50)

                        if self.do_train or self.do_discretize:
                            self = NLN.load(self.filename)
                        dichotomies_parameters, observed_concepts, unobserved_concepts = self.torch_module.get_dichotomies_observed_unobserved_parameters()
                        dichotomies_optimizer = torch.optim.Adam(dichotomies_parameters, lr=self.adam_lr)
                        start = time.time()
                        self.dichotomies_retrain_results = self._train(
                            dichotomies_optimizer,
                            self.retrain_nb_epochs,
                            False,
                            filename=filename,
                            use_best_val=True,
                            max_nb_epochs_since_last_best=100,
                            save_model_plots=False,
                            show_loss_epochs=self.do_save_training_progression_plots,
                        )
                        self.dichotomies_retrain_time = time.time() - start
                        print_log(f"Dichotomies retraining time: {self.dichotomies_retrain_time:.3e} seconds", True, timing_files)
                        self.has_retrained_dichotomies = True
                        self.save()

                        train_val_loss = self._eval_train_val_loss()
                        print(f"Full training set loss (train. + val.) after dichotomies retraining = {train_val_loss:.5g}")
                        if self.do_save_intermediate_learning_model_plots:
                            print("Saving plot...")
                            self.show(filename=filename, valid_raw_loss=train_val_loss)

                if self.do_retrain_unobserved_concepts and train_val_loss > 0:
                    filename += "_unobsRetrained"
                    if not self.has_retrained_unobserved_concepts:
                        if self.verbose:
                            print(f"\nUnobserved concepts retraining...")
                        else:
                            print()
                            printProgressBar(0, self.retrain_nb_epochs, prefix=f"Unobserved concepts retraining...  ", suffix="", length=50)

                        if self.do_train or self.do_discretize or self.do_retrain_dichotomies:
                            self = NLN.load(self.filename)
                        dichotomies_parameters, observed_concepts, unobserved_concepts = self.torch_module.get_dichotomies_observed_unobserved_parameters()
                        unobserved_concepts_optimizer = torch.optim.Adam(unobserved_concepts, lr=self.adam_lr)
                        start = time.time()
                        self.unobserved_concepts_retrain_results = self._train(
                            unobserved_concepts_optimizer,
                            self.retrain_nb_epochs,
                            False,
                            filename=filename,
                            use_best_val=True,
                            max_nb_epochs_since_last_best=100,
                            save_model_plots=False,
                            show_loss_epochs=self.do_save_training_progression_plots,
                        )
                        self.unobserved_concepts_retrain_time = time.time() - start
                        print_log(f"Unobserved concepts retraining time: {self.unobserved_concepts_retrain_time:.3e} seconds", True, timing_files)
                        self.has_retrained_unobserved_concepts = True
                        self.save()

                        train_val_loss = self._eval_train_val_loss()
                        print(f"Full training set loss (train. + val.) after unobserved retraining = {train_val_loss:.5g}")
                        if self.do_save_intermediate_learning_model_plots:
                            print("Saving plot...")
                            self.show(filename=filename, valid_raw_loss=train_val_loss)

            if self.do_prune:
                filename += "_pruned"
                if not self.has_pruned:
                    if self.verbose:
                        print(f"\nPruning...")
                        progress_bar_hook = lambda iteration, nb_iterations: None
                    else:
                        print()
                        printProgressBar(0, 1, prefix=f"Pruning...  ", suffix="", length=50)
                        progress_bar_hook = lambda iteration, nb_iterations: printProgressBar(iteration, nb_iterations, prefix=f"Pruning...  ", suffix="", length=50)

                    start = time.time()
                    self.torch_module.prune(self._eval_train_val_loss, self.save, filename=filename, do_log=self.do_log, progress_bar_hook=progress_bar_hook)
                    self.prune_time = time.time() - start
                    print_log(f"Pruning time: {self.prune_time:.3e} seconds", True, timing_files)
                    self.has_pruned = True
                    self.save()

                    train_val_loss = self._eval_train_val_loss()
                    print(f"Full training set loss (train. + val.) after pruning = {train_val_loss:.5g}")
                    if self.do_save_intermediate_learning_model_plots:
                        print("Saving plot...")
                        self.show(filename=filename, valid_raw_loss=train_val_loss)

            if self.do_analyze_dataset_coverage_and_adjust_biases and torch.max(self.torch_module.layers[-1].observed_concepts).item() > 0:
                filename += "_adjusted"
                if not self.has_analyzed_dataset_coverage_and_adjusted_biases:
                    print(f"\nAnalyzing dataset coverage and adjusting biases...")

                    start = time.time()
                    self.analyze_dataset_coverage_and_adjust_biases(do_adjust_biases=True, do_remove_included_rules=True, do_reorder_by_mass=False)  # , do_reorder_by_mass=True)
                    self.adjustment_time = time.time() - start
                    print_log(f"Dataset coverage anylisis and bias adjustment time: {self.adjustment_time:.3e} seconds", True, timing_files)
                    self.has_analyzed_dataset_coverage_and_adjusted_biases = True
                    self.save()

                    train_val_loss = self._eval_train_val_loss()
                    print(f"Full training set loss (train. + val.) after learning = {train_val_loss:.5g}")

            if self.do_reorder and torch.max(self.torch_module.layers[-1].observed_concepts).item() > 0:
                filename += "_reordered"
                if not self.has_reordered:
                    print(f"\nReordering...")

                    start = time.time()
                    self.torch_module.rearrange_for_visibility()
                    self.rearrange_time = time.time() - start
                    print_log(f"Reordering time: {self.rearrange_time:.3e} seconds", True, timing_files)
                    self.has_reordered = True
                    self.save()

                    # print_log(self.torch_module, True, [])

                    train_val_loss = self._eval_train_val_loss()
                    print(f"Full training set loss (train. + val.) after learning = {train_val_loss:.5g}")

            self.post_process_time = time.time() - post_process_start
            print_log(f"PostProcessing:\t{self.post_process_time:.3e}", True, timing_files)

            if self.do_save_final_plots:
                if not self.has_saved_final_plots:
                    print("Saving final plots...")
                    self.show(filename=self.filename, valid_raw_loss=train_val_loss)
                    self.show(filename=self.filename, one_rule_at_a_time=True)
                    self.show_dataset_coverage_analysis(filename=self.filename)

                    self.has_saved_final_plots = True
                    self.save()

            if (
                self.do_evaluate_on_train_val_dataset
                or self.do_plot_stats_train_val_dataset
                or self.do_evaluate_on_full_dataset
                or self.do_plot_stats_full_dataset
                or ((self.do_evaluate_on_test_dataset or self.do_plot_stats_test_dataset) and self.test_loader != None)
            ):
                if not self.has_evaluated_and_or_plotted_stats:
                    if self.do_evaluate_on_train_val_dataset or self.do_plot_stats_train_val_dataset:
                        print(f"\nEvaluating on training dataset (train. + val.)...")
                        train_val_dataset_evaluation_log_files = get_log_files(filename + "_trainvalDatasetEvaluation", True)

                        if self.do_evaluate_on_train_val_dataset:
                            start = time.time()
                            self.train_val_loss = self._validate(self.train_val_loader, override="")[0]
                            self.train_val_eval_time = time.time() - start
                            print_log(
                                "Training (train. + val.) dataset loss: " + str(self.train_val_loss),
                                True,
                                train_val_dataset_evaluation_log_files,
                            )
                            print_log(f"Training dataset inference time: {self.train_val_eval_time:.3e} seconds", True, timing_files)

                        if self.do_plot_stats_train_val_dataset:
                            res = self._plot_stats(self.train_val_loader, log_files=train_val_dataset_evaluation_log_files, filename=filename + "_trainval")
                            if not self.train_dataset.is_multi_label:
                                self.train_val_acc, self.train_val_f1 = res
                            else:
                                self.train_val_acc, self.train_val_all_correct_acc, self.train_val_f1 = res

                        close_log_files(train_val_dataset_evaluation_log_files)

                    if self.do_evaluate_on_full_dataset or self.do_plot_stats_full_dataset:
                        print(f"\nEvaluating on full dataset (train. + val. + test)...")
                        full_dataset_evaluation_log_files = get_log_files(filename + "_fullDatasetEvaluation", True)

                        if self.do_evaluate_on_full_dataset:
                            start = time.time()
                            self.full_loss = self._validate(self.train_val_loader, extra_dataloader=self.test_loader, override="")[0]
                            self.full_eval_time = time.time() - start
                            print_log(
                                "Full dataset loss: " + str(self.full_loss),
                                True,
                                full_dataset_evaluation_log_files,
                            )
                            print_log(f"Full dataset inference time: {self.full_eval_time:.3e} seconds", True, timing_files)

                        if self.do_plot_stats_full_dataset:
                            res = self._plot_stats(
                                self.train_val_loader, extra_dataloader=self.test_loader, log_files=full_dataset_evaluation_log_files, filename=filename + "_full"
                            )
                            if not self.train_dataset.is_multi_label:
                                self.full_acc, self.full_f1 = res
                            else:
                                self.full_acc, self.full_all_correct_acc, self.full_f1 = res

                        close_log_files(full_dataset_evaluation_log_files)

                    if (self.do_evaluate_on_test_dataset or self.do_plot_stats_test_dataset) and self.test_loader != None:
                        print(f"\nEvaluating on test dataset...")
                        test_dataset_evaluation_log_files = get_log_files(filename + "_testDatasetEvaluation", True)

                        if self.do_evaluate_on_test_dataset:
                            start = time.time()
                            self.test_loss = self._validate(self.test_loader, override="")[0]
                            self.test_eval_time = time.time() - start
                            print_log(
                                "Test loss: " + str(self.test_loss),
                                True,
                                test_dataset_evaluation_log_files,
                            )
                            print_log(f"Test dataset inference time: {self.test_eval_time:.3e} seconds", True, timing_files)

                        if self.do_plot_stats_test_dataset:
                            res = self._plot_stats(self.test_loader, log_files=test_dataset_evaluation_log_files, filename=filename + "_test")
                            if not self.train_dataset.is_multi_label:
                                self.test_acc, self.test_f1 = res
                            else:
                                self.test_acc, self.test_all_correct_acc, self.test_f1 = res

                        close_log_files(test_dataset_evaluation_log_files)

                    self.has_evaluated_and_or_plotted_stats = True

            self.has_learned = True
            self.save()

        close_log_files(timing_files)

        if self.do_save_intermediate_learning_model_plots or self.do_save_intermediate_training_model_plots or self.do_plot_stats_full_dataset or self.do_plot_stats_test_dataset:
            plt.close("all")

        return NLN.load(self.filename)

    def _eval_train_val_loss(self):
        return self._validate(self.train_val_loader, override="")[0]

    def _train(
        self,
        optimizer,
        nb_epochs,
        do_train_weights,
        filename="",
        use_best_val=True,
        max_nb_epochs_since_last_best=500,
        save_model_plots=True,
        show_loss_epochs=True,
        show_loss_time=False,
    ):
        results = {
            "train_no_reg_losses": [],
            "train_losses": [],
            "train_times": [],
            "valid_raw_losses": [],
            "valid_raw_times": [],
            "valid_thresh_losses": [],
            "valid_thresh_times": [],
            "gpu_mems": [],
            "test_raw_loss": 0.0,
            "test_raw_time": 0.0,
            "test_stoch_loss": 0.0,
            "test_stoch_time": 0.0,
            "test_thresh_loss": 0.0,
            "test_thresh_time": 0.0,
        }

        train_no_reg_loss, _ = self._validate(self.train_loader)
        if do_train_weights:
            train_loss, _ = self._validate(self.train_loader, use_regularization=True)
        valid_raw_loss, valid_raw_time = self._validate(self.val_loader, override="")
        if do_train_weights:
            valid_thresh_loss, valid_thresh_time = self._validate(self.val_loader, override="thresh")
        results["train_no_reg_losses"].append(train_no_reg_loss)
        if do_train_weights:
            results["train_losses"].append(train_loss)
        results["valid_raw_losses"].append(valid_raw_loss)
        results["valid_raw_times"].append(valid_raw_time)
        if do_train_weights:
            results["valid_thresh_losses"].append(valid_thresh_loss)
            results["valid_thresh_times"].append(valid_thresh_time)

        log_files = get_log_files(filename, self.do_log)
        model_string = str(self.torch_module)
        if do_train_weights:
            print_log(
                f"Before -----------------> Trn. (reg., no), Val. (raw, thresh): {train_loss:.3e}, {train_no_reg_loss:.3e}, {valid_raw_loss:.3e}, {valid_thresh_loss:.3e} | Trn., Val. (raw, thresh) Time: -----, {valid_raw_time:.3f}, {valid_thresh_time:.3f} | Model: ",
                self.verbose,
                files=log_files,
            )
        else:
            print_log(
                f"Before -----------------> Trn., Val.: {train_no_reg_loss:.3e}, {valid_raw_loss:.3e} | Trn., Val. Time: -----, {valid_raw_time:.3f} | Model: ",
                self.verbose,
                files=log_files,
            )
        print_log(model_string, self.verbose, files=log_files)

        if len(filename) > 0 and save_model_plots:
            start = time.time()
            self.show(filename=filename + "_epoch0", train_loss=train_loss, train_no_reg_loss=train_no_reg_loss, valid_raw_loss=valid_raw_loss, valid_thresh_loss=valid_thresh_loss)
            show_time = time.time() - start
            print(f"Show Time: {show_time:.3f}")

        best_valid_raw_loss = valid_raw_loss
        best_model_raw_string = model_string
        best_raw_epoch = 0
        if do_train_weights:
            best_valid_thresh_loss = valid_thresh_loss
            best_thresh_epoch = 0

        last_training_step_counter = [0]
        for epoch in range(1, nb_epochs + 1):
            train_loss, train_no_reg_loss, train_time = self._train_epoch(self.train_loader, optimizer, do_train_weights, epoch, last_training_step_counter)
            valid_raw_loss, valid_raw_time = self._validate(self.val_loader, override="")
            if do_train_weights:
                valid_thresh_loss, valid_thresh_time = self._validate(self.val_loader, override="thresh")
            gpu_mem = torch.cuda.memory_allocated(0)

            results["train_no_reg_losses"].append(train_no_reg_loss)
            if do_train_weights:
                results["train_losses"].append(train_loss)
            results["train_times"].append(train_time)
            results["valid_raw_losses"].append(valid_raw_loss)
            results["valid_raw_times"].append(valid_raw_time)
            if do_train_weights:
                results["valid_thresh_losses"].append(valid_thresh_loss)
                results["valid_thresh_times"].append(valid_thresh_time)
            results["gpu_mems"].append(gpu_mem)

            model_string = str(self.torch_module)
            if do_train_weights:
                print_log(
                    f"Epoch: {epoch} | GPU: {gpu_mem_to_string(gpu_mem)} | Trn. (reg., no), Val. (raw, thresh): {train_loss:.3e}, {train_no_reg_loss:.3e}, {valid_raw_loss:.3e}, {valid_thresh_loss:.3e} | Trn., Val. (raw, thresh) Time: {train_time:.3f}, {valid_raw_time:.3f}, {valid_thresh_time:.3f}",
                    self.verbose,
                    files=log_files,
                )
            else:
                print_log(
                    f"Epoch: {epoch} | GPU: {gpu_mem_to_string(gpu_mem)} | Trn., Val.: {train_no_reg_loss:.3e}, {valid_raw_loss:.3e} | Trn., Val. Time: {train_time:.3f}, {valid_raw_time:.3f}",
                    self.verbose,
                    files=log_files,
                )
            if self.log_model_every_training_epoch:
                print_log(model_string, False, files=log_files)

            if not self.verbose:
                if do_train_weights:
                    printProgressBar(epoch, self.train_nb_epochs, prefix=f"Initial training...  ", suffix="", length=50)
                else:
                    printProgressBar(epoch, self.retrain_nb_epochs, prefix=f"Continuous parameters retraining...  ", suffix="", length=50)

            if len(filename) > 0 and save_model_plots and (epoch < 10 or (epoch < 100 and epoch % 10 == 0) or epoch % 100 == 0):
                start = time.time()
                if do_train_weights:
                    self.show(
                        filename=filename + "_epoch" + str(epoch),
                        train_loss=train_loss,
                        train_no_reg_loss=train_no_reg_loss,
                        valid_raw_loss=valid_raw_loss,
                        valid_thresh_loss=valid_thresh_loss,
                    )
                else:
                    self.show(filename=filename + "_epoch" + str(epoch), train_no_reg_loss=train_no_reg_loss, valid_raw_loss=valid_raw_loss)
                show_time = time.time() - start
                # print(f"Show Time: {show_time:.3f}")

            if valid_raw_loss < best_valid_raw_loss:
                best_valid_raw_loss = valid_raw_loss
                best_model_raw_string = model_string
                best_raw_epoch = epoch
                # if not save_full_log:
                #   print_log(model_string, False, files=log_files)
            if do_train_weights:
                if valid_thresh_loss < best_valid_thresh_loss:
                    best_valid_thresh_loss = valid_thresh_loss
                    best_thresh_epoch = epoch

            if (use_best_val and epoch - best_raw_epoch == max_nb_epochs_since_last_best) or valid_raw_loss == 0.0:
                if not self.verbose:
                    if do_train_weights:
                        printProgressBar(self.train_nb_epochs, self.train_nb_epochs, prefix=f"Initial training...  ", suffix="", length=50)
                    else:
                        printProgressBar(self.retrain_nb_epochs, self.retrain_nb_epochs, prefix=f"Continuous parameters retraining...  ", suffix="", length=50)
                break

        actual_nb_epochs = len(results["train_no_reg_losses"]) - 1

        if use_best_val:
            print_log(f"Best Validation (Raw) Model from Epoch {best_raw_epoch} :\n", self.verbose, files=log_files)
            print_log(best_model_raw_string, self.verbose, files=log_files)

            self.torch_module.load_string(best_model_raw_string)
        else:
            print_log(f"Last Validation (Raw) Model from Epoch {epoch} :\n", self.verbose, files=log_files)
            print_log(model_string, self.verbose, files=log_files)

        if self.test_loader != None:
            test_raw_loss, test_raw_time = self._validate(self.test_loader, override="")
            results["test_raw_loss"] = test_raw_loss
            results["test_raw_time"] = test_raw_time
            if do_train_weights:
                test_thresh_loss, test_thresh_time = self._validate(self.test_loader, override="thresh")
                results["test_thresh_loss"] = test_thresh_loss
                results["test_thresh_time"] = test_thresh_time

        avg_gpu_mem = mean(results["gpu_mems"])
        if self.test_loader == None:
            print_log(
                f"\n{actual_nb_epochs} Epochs --> Avg. GPU: {gpu_mem_to_string(avg_gpu_mem)}\n",
                self.verbose,
                files=log_files,
            )
        else:
            if do_train_weights:
                print_log(
                    f"\n{actual_nb_epochs} Epochs --> Avg. GPU: {gpu_mem_to_string(avg_gpu_mem)} | Test (raw, thresh): {test_raw_loss:.3e}, {test_thresh_loss:.3e} | Test (raw, thresh) Time: {test_raw_time:.3f}, {test_thresh_time:.3f}\n",
                    self.verbose,
                    files=log_files,
                )
            else:
                print_log(
                    f"\n{actual_nb_epochs} Epochs --> Avg. GPU: {gpu_mem_to_string(avg_gpu_mem)} | Test: {test_raw_loss:.3e} | Test Time: {test_raw_time:.3f}\n",
                    self.verbose,
                    files=log_files,
                )

        best_train_no_reg_loss = min(results["train_no_reg_losses"])
        best_train_no_reg_epoch = results["train_no_reg_losses"].index(best_train_no_reg_loss)
        if do_train_weights:
            best_train_loss = min(results["train_losses"])
            best_train_epoch = results["train_losses"].index(best_train_loss)
            print_log(f"Best Training (Regularized) Loss : {best_train_loss:.3e} at " + str(best_train_epoch), self.verbose, files=log_files)
        print_log(f"Best Training Loss :               {best_train_no_reg_loss:.3e} at " + str(best_train_no_reg_epoch), self.verbose, files=log_files)
        print_log(f"Best Validation (raw) Loss :       {best_valid_raw_loss:.3e} at " + str(best_raw_epoch), self.verbose, files=log_files)
        if do_train_weights:
            print_log(f"Best Validation (thresh) Loss :    {best_valid_thresh_loss:.3e} at " + str(best_thresh_epoch), self.verbose, files=log_files)
        if self.test_loader != None:
            print_log(f"Test (raw) Loss :                  {test_raw_loss:.3e}", self.verbose, files=log_files)
            if do_train_weights:
                print_log(f"Test (thresh) Loss :               {test_thresh_loss:.3e}\n", self.verbose, files=log_files)

        close_log_files(log_files)

        if len(filename) > 0:
            pass
            # torch.save(model, filename+".pth")
            # load with: model = torch.load(filename+".pth")

        if show_loss_epochs:
            if len(filename) > 0:
                plt.close("all")
                plt.ion()
            fig, axs = plt.subplots(1, 1)
            axs = [axs]
            axs[0].set_title("Loss / Epochs")
            axs[0].set_ylabel("Loss")
            # if do_train_weights:
            #   axs[0].semilogy(list(range(actual_nb_epochs+1)),results["train_losses"], label="Training")
            axs[0].semilogy(list(range(actual_nb_epochs + 1)), results["train_no_reg_losses"], label="Train. (No Reg.)")
            axs[0].semilogy(list(range(actual_nb_epochs + 1)), results["valid_raw_losses"], label="Valid. (Raw)")
            if do_train_weights:
                axs[0].semilogy(list(range(actual_nb_epochs + 1)), results["valid_thresh_losses"], label="Valid. (Thresh.)")
            axs[0].legend(loc="upper right")
            axs[-1].set_xlabel("Epochs")
            if len(filename) > 0:
                plt.pause(1e-8)
                plt.ioff()
                plt.savefig(filename + "_EpochCurves.png", dpi=300)
            else:
                plt.show()

        if show_loss_time:
            if len(filename) > 0:
                plt.close("all")
                plt.ion()
            fig, axs = plt.subplots(1, 1)
            axs = [axs]
            axs[0].set_title("Loss / Time")
            axs[0].set_ylabel("Loss")
            times = [0] + np.cumsum(results["train_times"]).tolist()
            if do_train_weights:
                axs[0].semilogy(times, results["train_losses"], label="Training")
            axs[0].semilogy(times, results["train_no_reg_losses"], label="Train. (No Reg.)")
            axs[0].semilogy(times, results["valid_raw_losses"], label="Valid. (Raw)")
            if do_train_weights:
                axs[0].semilogy(times, results["valid_thresh_losses"], label="Valid. (Thresh.)")
            axs[0].legend(loc="upper right")
            axs[-1].set_xlabel("Time")
            if len(filename) > 0:
                plt.pause(1e-8)
                plt.ioff()
                plt.savefig(filename + "_TimeCurves.png", dpi=300)
            else:
                plt.show()

        return results

    def _train_epoch(self, dataloader, optimizer, do_train_weights, epoch, last_training_step_counter):
        train_no_reg_loss = 0.0
        train_loss = 0.0
        self.torch_module.train()
        start = time.time()

        for source, target in dataloader:
            source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device).to(torch.float32)

            optimizer.zero_grad()

            output = self.torch_module(source)
            output = output.view(-1)
            target = target.view(-1)
            loss = self.criterion(output, target)
            train_no_reg_loss += loss.item()
            if do_train_weights:
                loss = self.torch_module.add_regularization(loss)
                train_loss += loss.item()

            loss.backward()

            optimizer.step()

            self.torch_module.update_parameters()

            last_training_step_counter[0] += 1

        if do_train_weights and last_training_step_counter[0] >= MIN_NB_TRAINING_STEPS_BEFORE_REVIEW:
            self.torch_module.review_unused_concepts()
            last_training_step_counter[0] = 0

        end = time.time()
        train_no_reg_loss /= len(dataloader)
        if do_train_weights:
            train_loss /= len(dataloader)
        train_time = end - start
        return train_loss, train_no_reg_loss, train_time

    def _validate(self, dataloader, extra_dataloader=None, use_regularization=False, override=""):
        val_loss = 0.0
        self.torch_module.eval()
        self.torch_module.override(override)
        start = time.time()

        dataloaders = [dataloader] if extra_dataloader == None else [dataloader, extra_dataloader]
        for curr_dataloader in dataloaders:
            for source, target in curr_dataloader:
                source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device).to(torch.float32)

                output = self.torch_module(source)
                output = output.view(-1)
                target = target.view(-1)
                loss = self.criterion(output, target)
                if use_regularization:
                    loss = self.torch_module.add_regularization(loss)
                val_loss += loss.item()

        self.torch_module.override("")

        end = time.time()
        val_loss /= sum([len(curr_dataloader) for curr_dataloader in dataloaders])
        val_time = end - start
        return val_loss, val_time

    def _plot_stats(self, dataloader, extra_dataloader=None, log_files=[], filename=""):
        if self.train_dataset.nb_class_targets == 1 or self.train_dataset.is_multi_label:
            true_positives, false_positives, true_negatives, false_negatives = self._validate_with_threshold(dataloader, extra_dataloader=extra_dataloader)
            accs = [(true_positives[i] + true_negatives[i]) / (true_positives[i] + false_positives[i] + true_negatives[i] + false_negatives[i]) for i in range(len(true_positives))]
            best_acc = max(accs)
            thresholds = [0.01 * percentile for percentile in range(102)]
            best_index = accs.index(best_acc)
            best_threshold = thresholds[best_index]
            print_log(
                "Best threshold = " + str(best_threshold) + ",   acc = " + str(best_acc) + "   (always-true = " + str(accs[0]) + ",   always-false = " + str((accs[-1])) + ")",
                True,
                log_files,
            )
            if self.train_dataset.is_multi_label:
                all_corrects, nb_samples = self._validate_with_threshold(dataloader, extra_dataloader=extra_dataloader, multi_label=True)
                all_correct_accs = [all_corrects[i] / nb_samples for i in range(len(all_corrects))]
                best_all_correct_acc = max(all_correct_accs)
                best_all_correct_threshold = thresholds[all_correct_accs.index(best_all_correct_acc)]
                print_log(
                    "All Correct Best threshold = "
                    + str(best_all_correct_threshold)
                    + ",   All Correct acc = "
                    + str(best_all_correct_acc)
                    + "   (always-true = "
                    + str(all_correct_accs[0])
                    + ",   always-false = "
                    + str((all_correct_accs[-1]))
                    + ")",
                    True,
                    log_files,
                )
            recalls = [(true_positives[i]) / (true_positives[i] + false_negatives[i]) if true_positives[i] + false_negatives[i] > 0 else 0 for i in range(len(true_positives))]
            print_log(
                "Best Acc thresh. = "
                + str(best_threshold)
                + ",   recall = "
                + str(recalls[best_index])
                + "   (always-true = "
                + str(recalls[0])
                + ",   always-false = "
                + str((recalls[-1]))
                + ")",
                True,
                log_files,
            )
            precisions = [(true_positives[i]) / (true_positives[i] + false_positives[i]) if true_positives[i] + false_positives[i] > 0 else 0 for i in range(len(true_positives))]
            print_log(
                "Best Acc thresh. = "
                + str(best_threshold)
                + ",   prec = "
                + str(precisions[best_index])
                + "   (always-true = "
                + str(precisions[0])
                + ",   always-false = "
                + str((precisions[-1]))
                + ")",
                True,
                log_files,
            )
            f1s = [(2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) if precisions[i] + recalls[i] > 0 else 0 for i in range(len(true_positives))]
            print_log(
                "Best Acc thresh. = "
                + str(best_threshold)
                + ",   f1 = "
                + str(f1s[best_index])
                + "   (always-true = "
                + str(f1s[0])
                + ",   always-false = "
                + str((f1s[-1]))
                + ")",
                True,
                log_files,
            )

            if self.train_dataset.is_multi_label:
                if len(filename) > 0:
                    plt.close("all")
                    plt.ion()
                thresh_all_correct_acc_fig = plt.figure()
                thresh_all_correct_acc_ax = thresh_all_correct_acc_fig.add_subplot(111)
                thresh_all_correct_acc_ax.plot(thresholds[:-1], all_correct_accs[:-1])
                thresh_all_correct_acc_ax.scatter([best_all_correct_threshold], [best_all_correct_acc], color="black")
                thresh_all_correct_acc_ax.set_xlim(0, 1)
                thresh_all_correct_acc_ax.set_ylim(0, 1)
                thresh_all_correct_acc_ax.set_title("All Correct Accuracy vs Output Threshold")
                thresh_all_correct_acc_ax.set_xlabel("Output Threshold")
                thresh_all_correct_acc_ax.set_ylabel("All Correct Accuracy")
                if len(filename) > 0:
                    plt.pause(1e-8)
                    plt.ioff()
                    plt.savefig(filename + "_ALL_CORRECT_ROC.png", dpi=300)

            ROC_fig, ROC_ax, fpr, tpr, thresholds = self._plot_ROC(dataloader, best_threshold, extra_dataloader=extra_dataloader, filename=filename)

            hists_fig, hists_axs = self._plot_hists(dataloader, extra_dataloader=extra_dataloader, filename=filename)
            if filename == "":
                plt.show()
            if not self.train_dataset.is_multi_label:
                return best_acc, f1s[best_index]
            else:
                return best_acc, best_all_correct_acc, f1s[best_index]
        else:
            ambiguous, accuracy, recall, precision, f1 = self._validate_with_multiclass(dataloader, extra_dataloader=extra_dataloader)
            print_log("Ambiguous = " + str(ambiguous), True, log_files)
            print_log("Accuracy = " + str(accuracy), True, log_files)
            print_log("Recall = " + str(recall), True, log_files)
            print_log("Precision = " + str(precision), True, log_files)
            print_log("f1 = " + str(f1), True, log_files)
            return accuracy, f1

    def _validate_with_threshold(self, dataloader, extra_dataloader=None, multi_label=False):
        self.torch_module.eval()
        thresholds = [0.01 * percentile for percentile in range(102)]
        if not multi_label:
            true_positives = [0 for threshold in thresholds]
            false_positives = [0 for threshold in thresholds]
            true_negatives = [0 for threshold in thresholds]
            false_negatives = [0 for threshold in thresholds]
        else:
            all_corrects = [0 for threshold in thresholds]
            nb_samples = 0

        dataloaders = [dataloader] if extra_dataloader == None else [dataloader, extra_dataloader]
        for curr_dataloader in dataloaders:
            for source, target in curr_dataloader:
                source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device).to(torch.float32)

                output = self.torch_module(source)
                if not multi_label:
                    output = output.view(-1)
                    target = target.view(-1)
                else:
                    output = output.view(target.shape)
                    nb_samples += target.size(0)
                for i, threshold in enumerate(thresholds):
                    output_copy = 1 * output
                    output_copy[output < threshold] = 0
                    output_copy[output >= threshold] = 1
                    if not multi_label:
                        true_positives[i] += ((output_copy == target) * (output_copy == 1)).sum().item()
                        false_positives[i] += ((output_copy != target) * (output_copy == 1)).sum().item()
                        true_negatives[i] += ((output_copy == target) * (output_copy == 0)).sum().item()
                        false_negatives[i] += ((output_copy != target) * (output_copy == 0)).sum().item()
                    else:
                        all_corrects[i] += torch.prod(output_copy == target, dim=1).sum().item()

        if not multi_label:
            return true_positives, false_positives, true_negatives, false_negatives
        else:
            return all_corrects, nb_samples

    def _validate_with_multiclass(self, dataloader, extra_dataloader=None):
        self.torch_module.eval()
        nb_correct_predictions = 0
        nb_ambiguous_predictions = 0
        nb_predictions = 0
        true_positives = [0 for class_target in range(self.train_dataset.nb_class_targets)]
        false_positives = [0 for class_target in range(self.train_dataset.nb_class_targets)]
        true_negatives = [0 for class_target in range(self.train_dataset.nb_class_targets)]
        false_negatives = [0 for class_target in range(self.train_dataset.nb_class_targets)]

        dataloaders = [dataloader] if extra_dataloader == None else [dataloader, extra_dataloader]
        for curr_dataloader in dataloaders:
            for source, target in curr_dataloader:
                source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device).to(torch.float32)

                output = self.torch_module(source)
                output = output.view(target.shape)
                output_copy = torch.zeros_like(output)
                output_copy[torch.arange(output.size(0)), torch.max(output, dim=1)[1]] = 1
                non_ambiguous_predictions = torch.sum(output_copy, dim=1) == 1
                nb_ambiguous_predictions += torch.sum(~non_ambiguous_predictions).item()
                nb_predictions += output_copy.size(0)
                for class_target in range(self.train_dataset.nb_class_targets):
                    nb_correct_predictions += (
                        (
                            (output_copy[:, class_target][non_ambiguous_predictions] == target[:, class_target][non_ambiguous_predictions])
                            * (output_copy[:, class_target][non_ambiguous_predictions] == 1)
                        )
                        .sum()
                        .item()
                    )
                    true_positives[class_target] += ((output_copy[:, class_target] == target[:, class_target]) * (output_copy[:, class_target] == 1)).sum().item()
                    false_positives[class_target] += ((output_copy[:, class_target] != target[:, class_target]) * (output_copy[:, class_target] == 1)).sum().item()
                    true_negatives[class_target] += ((output_copy[:, class_target] == target[:, class_target]) * (output_copy[:, class_target] == 0)).sum().item()
                    false_negatives[class_target] += ((output_copy[:, class_target] != target[:, class_target]) * (output_copy[:, class_target] == 0)).sum().item()

        recalls = [(true_positives[i]) / (true_positives[i] + false_negatives[i]) if true_positives[i] + false_negatives[i] > 0 else 0 for i in range(len(true_positives))]
        precisions = [(true_positives[i]) / (true_positives[i] + false_positives[i]) if true_positives[i] + false_positives[i] > 0 else 0 for i in range(len(true_positives))]
        f1s = [(2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) if precisions[i] + recalls[i] > 0 else 0 for i in range(len(true_positives))]
        return nb_ambiguous_predictions / nb_predictions, nb_correct_predictions / nb_predictions, mean(recalls), mean(precisions), mean(f1s)

    def _plot_ROC(self, dataloader, best_threshold, extra_dataloader=None, filename=""):
        self.torch_module.eval()
        metric = BinaryROC(thresholds=101).to(self.torch_module.device)

        dataloaders = [dataloader] if extra_dataloader == None else [dataloader, extra_dataloader]
        for curr_dataloader in dataloaders:
            for source, target in curr_dataloader:
                source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device).to(torch.int16)

                output = self.torch_module(source)
                output = output.view(-1)
                target = target.view(-1)
                metric.update(output, target)

        fpr, tpr, thresholds = metric.compute()
        higher_bound = 2
        for i, threshold in enumerate(thresholds):
            lower_bound = threshold.item()
            if best_threshold >= lower_bound and best_threshold < higher_bound:
                if i == 0:
                    best_fpr = fpr[i].item()
                    best_tpr = tpr[i].item()
                else:
                    lower_coeff = (higher_bound - best_threshold) / (higher_bound - lower_bound)
                    best_fpr = lower_coeff * fpr[i].item() + (1 - lower_coeff) * fpr[i - 1].item()
                    best_tpr = lower_coeff * tpr[i].item() + (1 - lower_coeff) * tpr[i - 1].item()
                break
            higher_bound = lower_bound

        if len(filename) > 0:
            plt.close("all")
            plt.ion()
        fig, ax = metric.plot(score=True)
        ax.scatter([best_fpr], [best_tpr], color="black")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if len(filename) > 0:
            plt.pause(1e-8)
            plt.ioff()
            plt.savefig(filename + "_ROC.png", dpi=300)
        return fig, ax, fpr, tpr, thresholds

    def _plot_hists(self, dataloader, extra_dataloader=None, filename=""):
        self.torch_module.eval()
        all_targets = [[], []]
        all_outputs = [[], []]

        dataloaders = [dataloader] if extra_dataloader == None else [dataloader, extra_dataloader]
        for curr_dataloader in dataloaders:
            for source, target in curr_dataloader:
                source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device)

                output = self.torch_module(source)
                outputs = output.view(-1).tolist()
                targets = target.view(-1).tolist()
                all_targets[0] += [target for target in targets if target == 0]
                all_targets[1] += [target for target in targets if target == 1]
                all_outputs[0] += [outputs[i] for i, target in enumerate(targets) if target == 0]
                all_outputs[1] += [outputs[i] for i, target in enumerate(targets) if target == 1]

        if len(filename) > 0:
            plt.close("all")
            plt.ion()
        fig, axs = plt.subplots(3, 1)
        axs[0].hist(all_targets, stacked=True)
        axs[1].hist(all_outputs, [0.01 * percentile for percentile in range(101)], stacked=True)
        false_counts, _ = np.histogram(all_outputs[0], [0.01 * percentile for percentile in range(101)])
        true_counts, _ = np.histogram(all_outputs[1], [0.01 * percentile for percentile in range(101)])
        total_counts = [false_counts[percentile] + true_counts[percentile] for percentile in range(100)]
        est_probs = [0.01 * percentile + 0.005 for percentile in range(100) if total_counts[percentile] > 0]
        true_probs = [true_counts[percentile] / total_counts[percentile] for percentile in range(100) if total_counts[percentile] > 0]
        axs[2].plot([0, 1], [0, 1], color="gray")
        axs[2].scatter(est_probs, true_probs, s=[300 * total_count / max(total_counts) for total_count in total_counts if total_count > 0])
        axs[0].set_title("Probability Distributions")
        axs[0].set_ylabel("Ground-truth\ndistribution")
        axs[1].set_ylabel("Model output\ndistribution")
        axs[2].set_ylabel("Ground-truth prob.\ngiven model out.")
        axs[2].set_xlabel("Probability")
        if len(filename) > 0:
            plt.pause(1e-8)
            plt.ioff()
            plt.savefig(filename + "_PROB_MODELING.png", dpi=300)
        return fig, axs

    def get_rule_string(self):
        return self.torch_module.to_rule_string()

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
        self.torch_module.show(
            fig_ax=fig_ax,
            filename=filename,
            train_no_reg_loss=train_no_reg_loss,
            train_loss=train_loss,
            valid_raw_loss=valid_raw_loss,
            valid_stoch_loss=valid_stoch_loss,
            valid_thresh_loss=valid_thresh_loss,
            save_dpi=save_dpi,
            one_rule_at_a_time=one_rule_at_a_time,
            is_single_rule=is_single_rule,
        )

    def show_dataset_coverage_analysis(self, show_train=False, show_train_val=True, show_train_val_test=True, filename="", save_dpi=100):
        if show_train:
            train_filename = f"{filename}_trainDatasetCoverageAnalysis" if len(filename) > 0 else filename
            self._show_dataset_coverage_analysis([self.train_loader], filename=train_filename, save_dpi=save_dpi)
        if show_train_val:
            train_val_filename = f"{filename}_trainvalDatasetCoverageAnalysis" if len(filename) > 0 else filename
            self._show_dataset_coverage_analysis([self.train_val_loader], filename=train_val_filename, save_dpi=save_dpi)
        if show_train_val_test and self.test_loader != None:
            train_val_test_filename = f"{filename}_fullDatasetCoverageAnalysis" if len(filename) > 0 else filename
            self._show_dataset_coverage_analysis([self.train_val_loader, self.test_loader], filename=train_val_test_filename, save_dpi=save_dpi)

    def _show_dataset_coverage_analysis(
        self,
        dataloaders,
        filename="",
        save_dpi=100,
    ):
        is_multi_label = self.train_dataset.is_multi_label

        nb_rules = self.torch_module.layers[-1].nb_in_concepts
        nb_targets = self.torch_module.nb_out_concepts

        cumul_presences, inclusion_matrix = self._do_dataset_coverage_analysis(dataloaders)

        if not is_multi_label:
            columns = tuple([f"AND{rule_idx}" for rule_idx in range(nb_rules)] + [f"OR{target_idx}" for target_idx in range(nb_targets)] + ["Any"])
        else:
            columns = tuple([f"AND{rule_idx}" for rule_idx in range(nb_rules)] + [f"OR{target_idx}" for target_idx in range(nb_targets)])

        target_values = self.torch_module.get_target_feature_names()
        if nb_targets == 1 or is_multi_label:
            tmp_target_values = []
            for target_value in target_values:
                tmp_target_values += [target_value, f"NOT {target_value}"]
            target_values = tmp_target_values
        rows = target_values + ["full"]

        if not is_multi_label:
            nb_colors = len(rows)
            color_idcs = list(range(nb_colors))
        else:
            nb_colors = int((len(rows) - 1) / 2 + 1)
            color_idcs = [half_row_idx if bin_idx == 0 else -1 for half_row_idx in range(int((len(rows) - 1) / 2)) for bin_idx in range(2)] + [int(nb_colors - 1)]
        if nb_colors <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:nb_colors]
        elif nb_colors <= 20:
            color_positions_all_20 = np.linspace(0, 1, 20).tolist()
            color_positions_all_20 = color_positions_all_20[::2] + color_positions_all_20[1::2]
            colors = plt.cm.tab20(color_positions_all_20)[:nb_colors]
        else:
            warnings.warn("Colors for case with > 19 targets is not implemented cleanly. Colors are cycled through the 20 available colors...")
            color_positions_all_20 = np.linspace(0, 1, 20).tolist()
            color_positions_all_20 = color_positions_all_20[::2] + color_positions_all_20[1::2]
            colors = plt.cm.tab20(color_positions_all_20)[:nb_colors]
            color_idcs = [color_idx % len(colors) if color_idx >= 0 else color_idx for color_idx in color_idcs]

        if len(filename) > 0:
            plt.ion()

        bar_width = 0.5

        fig, ax = plt.subplots(1, 1, gridspec_kw={"left": 0.1, "right": 0.99, "bottom": 0.75, "top": 0.99})
        fig.set_size_inches(25.6, 13.19)

        # Plot bars and create text labels for the table
        if not is_multi_label:
            for node_idx in range(nb_rules + nb_targets + 1):
                if node_idx < nb_rules:
                    rule_uses = torch.nonzero(self.torch_module.layers[-1].observed_concepts[:, node_idx]).view(-1).tolist()
                    row_order = sorted(rule_uses) + sorted(list(set(range(len(rows) - 1)) - set(rule_uses)))
                elif node_idx < nb_rules + nb_targets:
                    target_idx = node_idx - nb_rules
                    row_order = [target_idx] + sorted(list(set(range(len(rows) - 1)) - {target_idx}))
                else:
                    row_order = range(len(rows) - 1)

                y_offset = 0
                for row in row_order:
                    bar_value = cumul_presences[row, node_idx] / cumul_presences[-1, node_idx]
                    ax.bar([node_idx + 0.5], [bar_value], bar_width, bottom=[y_offset], color=colors[color_idcs[row]])
                    y_offset += bar_value
        else:
            for node_idx in range(nb_rules + nb_targets):
                if node_idx < nb_rules:
                    rule_uses = torch.nonzero(self.torch_module.layers[-1].observed_concepts[:, node_idx]).view(-1).tolist()
                else:
                    rule_uses = [node_idx - nb_rules]

                x_offset = -0.5 * bar_width + 0.5 * bar_width / len(rule_uses)
                for half_row in rule_uses:
                    bar_value = cumul_presences[2 * half_row, node_idx] / cumul_presences[-1, node_idx]
                    ax.bar([node_idx + 0.5 + x_offset], [bar_value], bar_width / len(rule_uses), bottom=[0], color=colors[half_row])
                    x_offset += bar_width / len(rule_uses)

        cell_texts = []
        cell_colors = []

        def shorten_number(x):
            if x == 0:
                return "0"
            elif x == 1:
                return "1"
            else:
                return f"{x:.2f}".replace("0.", ".")

        for row in range(len(rows)):
            values = [cumul_presences[row, rule_idx] / cumul_presences[row, -1] for rule_idx in range(len(columns))]
            cell_texts.append([shorten_number(value) for value in values])
            row_color = colors[color_idcs[row]][:3].tolist() if color_idcs[row] >= 0 else [1, 1, 1]
            cell_colors.append([np.array(row_color + [min(cumul_presences[row, rule_idx] / cumul_presences[row, -1], 1)]) for rule_idx in range(len(columns))])

        rule_biases = self.torch_module.layers[-2].unobserved_concepts.data
        for rule_idx in range(nb_rules):
            rule_bias = rule_biases[rule_idx].item()
            line_width = 2 if rule_bias == 0 or rule_bias == 1 else 1
            ax.plot([rule_idx + 0.5 - bar_width / 2, rule_idx + 0.5 + bar_width / 2], [rule_bias, rule_bias], color="black", lw=line_width, ls="dashed")

        cell_texts.append(["" for node_jdx in range(len(columns))])
        cell_colors.append([np.array([1, 1, 1, 1]) for node_jdx in range(len(columns))])
        for rule_idx in range(nb_rules):
            # cell_texts.append([f"{round(100*inclusion_matrix[rule_idx, node_jdx],2)}" for node_jdx in range(len(columns))])
            cell_texts.append(len(columns) * [""])
            cell_colors.append([np.array([0, 0, 0, inclusion_matrix[rule_idx, node_jdx]]) for node_jdx in range(len(columns))])

        # Add a table at the bottom of the Axes
        the_table = ax.table(
            cellText=cell_texts,
            rowLabels=rows + [""] + [f"AND{rule_idx}" for rule_idx in range(nb_rules)],
            rowColours=np.array(
                [colors[color_idcs[row]].tolist() if color_idcs[row] >= 0 else [1, 1, 1, 1] for row in range(len(rows))] + (nb_rules + 1) * [np.array([1, 1, 1, 1])]
            ),
            colLabels=columns,
            loc="bottom",
            cellColours=cell_colors,
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)

        ax.set_ylabel(f"Breakdown of rule coverage")
        # plt.yticks(values * value_increment, ["%d" % val for val in values])
        ax.set_xlim(0, len(columns))
        ax.set_xticks([])
        # plt.title("Loss by Disaster")

        if filename == "":
            plt.get_current_fig_manager().window.state("zoomed")
            plt.show()
        else:
            plt.pause(1e-8)
            plt.ioff()
            plt.savefig(filename + ".png", dpi=save_dpi)
            plt.close("all")

    def _do_dataset_coverage_analysis(self, dataloaders, keep_rule_biases=False):
        is_multi_label = self.train_dataset.is_multi_label

        nb_rules = self.torch_module.layers[-1].nb_in_concepts
        nb_targets = self.torch_module.nb_out_concepts
        nb_target_values = nb_targets if nb_targets > 1 and not is_multi_label else 2 * nb_targets

        cumul_presences = np.zeros((nb_target_values + 1, nb_rules + nb_targets + 2))
        inclusion_matrix = np.zeros((nb_rules, nb_rules + nb_targets + 1))

        self.torch_module.eval()
        for dataloader in dataloaders:
            for source, target in dataloader:
                source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device).to(torch.float32)

                cumul_presences[-1, -1] += source.size(0)
                if not is_multi_label:
                    cumul_presences[:nb_targets, -1] += torch.sum(target, dim=0).detach().cpu().numpy()
                    if nb_targets == 1:
                        cumul_presences[1:2, -1] += torch.sum(1 - target, dim=0).detach().cpu().numpy()
                else:
                    for out_idx in range(nb_targets):
                        cumul_presences[2 * out_idx, -1] += torch.sum(target[:, out_idx], dim=0).detach().cpu().item()
                        cumul_presences[2 * out_idx + 1, -1] += torch.sum(1 - target[:, out_idx], dim=0).detach().cpu().item()

                tmp = self.torch_module.input_module(source)
                if not keep_rule_biases:
                    rule_layer_uses_biases = self.torch_module.layers[-2].use_unobserved
                    self.torch_module.layers[-2].use_unobserved = False
                rule_outputs_without_bias = self.torch_module.layers[:-1](tmp)
                if not keep_rule_biases:
                    self.torch_module.layers[-2].use_unobserved = rule_layer_uses_biases
                target_layer_uses_biases = self.torch_module.layers[-1].use_unobserved
                self.torch_module.layers[-1].use_unobserved = False
                target_outputs_without_bias = self.torch_module.layers[-1:](1 * rule_outputs_without_bias)
                self.torch_module.layers[-1].use_unobserved = target_layer_uses_biases
                any_output = 1 - torch.prod(1 - target_outputs_without_bias, dim=1)

                node_outputs = torch.concat((rule_outputs_without_bias, target_outputs_without_bias, any_output.unsqueeze(1)), dim=1)

                if not is_multi_label:
                    if nb_targets == 1:
                        # binary classification
                        out_0_rule_presences = torch.sum(node_outputs * target[:, 0].unsqueeze(1), dim=0)
                        cumul_presences[0, : nb_rules + nb_targets + 1] += out_0_rule_presences.detach().cpu().numpy()
                        out_not_0_rule_presences = torch.sum(node_outputs * (1 - target[:, 0].unsqueeze(1)), dim=0)
                        cumul_presences[1, : nb_rules + nb_targets + 1] += out_not_0_rule_presences.detach().cpu().numpy()
                    else:
                        # multi-class classification
                        for out_idx in range(nb_targets):
                            out_idx_rule_presences = torch.sum(node_outputs * target[:, out_idx].unsqueeze(1), dim=0)
                            cumul_presences[out_idx, : nb_rules + nb_targets + 1] += out_idx_rule_presences.detach().cpu().numpy()
                else:
                    # multi-label classification
                    for out_idx in range(nb_targets):
                        out_idx_rule_presences = torch.sum(node_outputs * target[:, out_idx].unsqueeze(1), dim=0)
                        cumul_presences[2 * out_idx, : nb_rules + nb_targets + 1] += out_idx_rule_presences.detach().cpu().numpy()
                        out_idx_not_rule_presences = torch.sum(node_outputs * (1 - target[:, out_idx].unsqueeze(1)), dim=0)
                        cumul_presences[2 * out_idx + 1, : nb_rules + nb_targets + 1] += out_idx_not_rule_presences.detach().cpu().numpy()

                out_rule_presences = torch.sum(node_outputs, dim=0)
                cumul_presences[-1, : nb_rules + nb_targets + 1] += out_rule_presences.detach().cpu().numpy()

                for rule_idx in range(nb_rules):
                    for node_jdx in range(nb_rules + nb_targets + 1):
                        inclusion_matrix[rule_idx, node_jdx] += (
                            torch.sum(torch.sqrt(node_outputs[:, rule_idx] * torch.minimum(node_outputs[:, rule_idx], node_outputs[:, node_jdx]))).detach().item()
                        )

        for rule_idx in range(nb_rules):
            for node_jdx in range(nb_rules + nb_targets + 1):
                inclusion_matrix[rule_idx, node_jdx] /= cumul_presences[-1, rule_idx]
        inclusion_matrix[inclusion_matrix > 1] = 1

        return cumul_presences, inclusion_matrix

    def analyze_dataset_coverage_and_adjust_biases(self, do_adjust_biases=True, do_remove_included_rules=True, do_reorder_by_mass=True):
        is_multi_label = self.train_dataset.is_multi_label

        nb_rules = self.torch_module.layers[-1].nb_in_concepts
        nb_targets = self.torch_module.nb_out_concepts

        cumul_presences, inclusion_matrix = self._do_dataset_coverage_analysis([self.train_val_loader])
        has_changed = False

        if do_adjust_biases:
            new_rule_idx = 0
            for old_rule_idx in range(nb_rules):
                rule_uses = sorted(torch.nonzero(self.torch_module.layers[-1].observed_concepts[:, new_rule_idx]).view(-1).tolist())

                if not is_multi_label:
                    new_biases = [cumul_presences[rule_use, old_rule_idx] / cumul_presences[-1, old_rule_idx] for rule_use in rule_uses]
                else:
                    new_biases = [cumul_presences[2 * rule_use, old_rule_idx] / cumul_presences[-1, old_rule_idx] for rule_use in rule_uses]
                new_biases_set_list = list(set(new_biases))

                if len(new_biases_set_list) == 1:
                    self.torch_module.layers[-2].unobserved_concepts.data[new_rule_idx] = new_biases[0]
                else:
                    new_biases_uses = []
                    for new_bias_value in new_biases_set_list:
                        new_uses = [rule_uses[new_bias_uses_idx] for new_bias_uses_idx in range(len(rule_uses)) if new_biases[new_bias_uses_idx] == new_bias_value]
                        new_biases_uses.append((new_bias_value, new_uses))
                    self.torch_module.duplicate_rule_with_new_biases_and_uses(new_rule_idx, new_biases_uses)
                    has_changed = True
                new_rule_idx += len(new_biases_set_list)

        if do_remove_included_rules:
            if has_changed:
                nb_rules = self.torch_module.layers[-1].nb_in_concepts
                cumul_presences, inclusion_matrix = self._do_dataset_coverage_analysis([self.train_val_loader])
                has_changed = False

            # simply_included_rules = []
            # simply_included_with_bias_rules = []
            included_rules = []
            for old_rule_idx in range(nb_rules):
                for rule_jdx in range(nb_rules):
                    if rule_jdx != old_rule_idx and inclusion_matrix[old_rule_idx, rule_jdx] >= 1:  # 0.999:
                        # simply_included_rules.append(old_rule_idx)
                        included_bias = self.torch_module.layers[-2].unobserved_concepts.data[old_rule_idx]
                        including_bias = self.torch_module.layers[-2].unobserved_concepts.data[rule_jdx]
                        if including_bias >= included_bias:
                            # simply_included_with_bias_rules.append(old_rule_idx)
                            included_uses = self.torch_module.layers[-1].observed_concepts.data[:, old_rule_idx]
                            including_uses = self.torch_module.layers[-1].observed_concepts.data[:, rule_jdx]
                            if (including_uses >= included_uses).all():
                                included_rules.append(old_rule_idx)
            included_rules = sorted(list(set(included_rules)))
            # print(sorted(list(set(simply_included_rules))))
            # print(sorted(list(set(simply_included_with_bias_rules))))
            # print(included_rules)
            self.torch_module.layers[-1].observed_concepts.data[:, included_rules] = 0
            self.torch_module.simplify()

            new_rule_idx = 0
            old_to_new_rule_idcs = []
            for old_rule_idx in range(nb_rules):
                if old_rule_idx in included_rules:
                    old_to_new_rule_idcs.append(float("inf"))
                else:
                    old_to_new_rule_idcs.append(new_rule_idx)
                    new_rule_idx += 1
        else:
            old_to_new_rule_idcs = list(range(self.torch_module.layers[-1].nb_in_concepts))

        if do_reorder_by_mass:
            if has_changed:
                nb_rules = self.torch_module.layers[-1].nb_in_concepts
                cumul_presences, inclusion_matrix = self._do_dataset_coverage_analysis([self.train_val_loader])

            rule_target_coverings = [[] for target_idx in range(nb_targets)]
            for old_rule_idx in range(nb_rules):
                new_rule_idx = old_to_new_rule_idcs[old_rule_idx]
                if new_rule_idx < self.torch_module.layers[-1].nb_in_concepts:
                    rule_uses = sorted(torch.nonzero(self.torch_module.layers[-1].observed_concepts[:, new_rule_idx]).view(-1).tolist())
                    if not is_multi_label:
                        rule_target_coverings[rule_uses[0]].append((new_rule_idx, min(cumul_presences[rule_uses[0], old_rule_idx] / cumul_presences[rule_uses[0], -1], 1)))
                    else:
                        rule_target_coverings[rule_uses[0]].append((new_rule_idx, min(cumul_presences[2 * rule_uses[0], old_rule_idx] / cumul_presences[2 * rule_uses[0], -1], 1)))

            for target_idx in range(nb_targets):
                rule_target_coverings[target_idx].sort(key=lambda pair: pair[1], reverse=True)

            rule_order = [rule for rule_target_covering in rule_target_coverings for rule, target_covering in rule_target_covering]

            self.torch_module.reorder_lexicographically_and_by_unobserved(rule_order)

        if do_adjust_biases:
            nb_rules = self.torch_module.layers[-1].nb_in_concepts

            cumul_presences, inclusion_matrix = self._do_dataset_coverage_analysis([self.train_val_loader], keep_rule_biases=True)

            for target_idx in range(nb_targets):
                if not is_multi_label:
                    new_bias = (cumul_presences[target_idx, -1] - cumul_presences[target_idx, nb_rules + target_idx]) / (
                        cumul_presences[-1, -1] - cumul_presences[-1, nb_rules + target_idx]
                    )
                else:
                    new_bias = (cumul_presences[2 * target_idx, -1] - cumul_presences[2 * target_idx, nb_rules + target_idx]) / (
                        cumul_presences[-1, -1] - cumul_presences[-1, nb_rules + target_idx]
                    )

                self.torch_module.layers[-1].unobserved_concepts.data[target_idx] = new_bias

    def show_inter_analysis(self, other_NLN):
        uncomparable = not (
            self.torch_module.nb_in_concepts == other_NLN.torch_module.nb_in_concepts
            and self.torch_module.nb_out_concepts == other_NLN.torch_module.nb_out_concepts
            and self.torch_module.feature_names == other_NLN.torch_module.feature_names
            and self.torch_module.category_first_last_has_missing_values_tuples == other_NLN.torch_module.category_first_last_has_missing_values_tuples
            and self.torch_module.continuous_index_min_max_has_missing_values_tuples == other_NLN.torch_module.continuous_index_min_max_has_missing_values_tuples
            and self.torch_module.periodic_index_period_has_missing_values_tuples == other_NLN.torch_module.periodic_index_period_has_missing_values_tuples
        )
        if uncomparable:
            raise Exception("Inter-analysis is only possible for comparable models (learned on the same problem inputs/outputs).")
        self._show_inter_analysis(other_NLN, [self.train_val_loader])
        if self.test_loader != None:
            self._show_inter_analysis(other_NLN, [self.train_val_loader, self.test_loader])

    def _show_inter_analysis(self, other_NLN, dataloaders):
        if self.train_dataset.is_multi_label:
            raise Exception("Multi-label classification not yet coded for inter-analysis, only binary and multi-class classification...")

        inclusion_matrix_i_in_j, inclusion_matrix_j_in_i = self._do_inter_analysis(other_NLN, dataloaders)

        rules_i = [f"this AND{rule_idx}" for rule_idx in range(inclusion_matrix_i_in_j.shape[0])]
        rules_j = [f"that AND{rule_jdx}" for rule_jdx in range(inclusion_matrix_i_in_j.shape[1])]

        has_no_equal_or_subrule_or_superrule_i = [True for rule_idx in range(inclusion_matrix_i_in_j.shape[0])]
        has_no_equal_or_subrule_or_superrule_j = [True for rule_jdx in range(inclusion_matrix_i_in_j.shape[1])]
        i_equals_js = []
        i_in_js = []
        j_in_is = []

        for rule_idx in range(inclusion_matrix_i_in_j.shape[0]):
            for rule_jdx in range(inclusion_matrix_i_in_j.shape[1]):
                if inclusion_matrix_i_in_j[rule_idx, rule_jdx] == 1:
                    if inclusion_matrix_j_in_i[rule_jdx, rule_idx] == 1:
                        i_equals_js.append((rule_idx, rule_jdx))
                        has_no_equal_or_subrule_or_superrule_i[rule_idx] = False
                        has_no_equal_or_subrule_or_superrule_j[rule_jdx] = False
                    else:
                        i_in_js.append((rule_idx, rule_jdx))
                        has_no_equal_or_subrule_or_superrule_i[rule_idx] = False
                        has_no_equal_or_subrule_or_superrule_j[rule_jdx] = False
                else:
                    if inclusion_matrix_j_in_i[rule_jdx, rule_idx] == 1:
                        j_in_is.append((rule_jdx, rule_idx))
                        has_no_equal_or_subrule_or_superrule_i[rule_idx] = False
                        has_no_equal_or_subrule_or_superrule_j[rule_jdx] = False

        print()
        for rule_idx, equal_rule_jdx in i_equals_js:
            print(f"{rules_i[rule_idx]} = {rules_j[equal_rule_jdx]}")

        for rule_idx, superrule_rule_jdx in i_in_js:
            print(f"{rules_i[rule_idx]} ⊂ {rules_j[superrule_rule_jdx]}")

        for rule_jdx, superrule_rule_idx in j_in_is:
            print(f"{rules_j[rule_jdx]} ⊂ {rules_i[superrule_rule_idx]}")

        is_first_unique = True
        for rule_idx in range(inclusion_matrix_i_in_j.shape[0]):
            if has_no_equal_or_subrule_or_superrule_i[rule_idx]:
                if is_first_unique:
                    print("Unique rules:")
                    is_first_unique = False
                print(f"\t{rules_i[rule_idx]}")
        for rule_jdx in range(inclusion_matrix_i_in_j.shape[1]):
            if has_no_equal_or_subrule_or_superrule_j[rule_jdx]:
                if is_first_unique:
                    print("Unique rules:")
                    is_first_unique = False
                print(f"\t{rules_j[rule_jdx]}")

        def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
            """
            Create a heatmap from a numpy array and two lists of labels.

            Parameters
            ----------
            data
                A 2D numpy array of shape (M, N).
            row_labels
                A list or array of length M with the labels for the rows.
            col_labels
                A list or array of length N with the labels for the columns.
            ax
                A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
                not provided, use current Axes or create a new one.  Optional.
            cbar_kw
                A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
            cbarlabel
                The label for the colorbar.  Optional.
            **kwargs
                All other arguments are forwarded to `imshow`.
            """

            if ax is None:
                ax = plt.gca()

            if cbar_kw is None:
                cbar_kw = {}

            # Plot the heatmap
            im = ax.imshow(data, **kwargs)

            # Create colorbar
            cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
            cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

            # Show all ticks and label them with the respective list entries.
            ax.set_xticks(range(data.shape[1]))
            ax.set_xticklabels(col_labels, rotation=-30, ha="right", rotation_mode="anchor")
            ax.set_yticks(range(data.shape[0]))
            ax.set_yticklabels(row_labels)

            # Let the horizontal axes labeling appear on top.
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            # Turn spines off and create white grid.
            ax.spines[:].set_visible(False)

            ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)

            return im, cbar

        def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
            """
            A function to annotate a heatmap.

            Parameters
            ----------
            im
                The AxesImage to be labeled.
            data
                Data used to annotate.  If None, the image's data is used.  Optional.
            valfmt
                The format of the annotations inside the heatmap.  This should either
                use the string format method, e.g. "$ {x:.2f}", or be a
                `matplotlib.ticker.Formatter`.  Optional.
            textcolors
                A pair of colors.  The first is used for values below a threshold,
                the second for those above.  Optional.
            threshold
                Value in data units according to which the colors from textcolors are
                applied.  If None (the default) uses the middle of the colormap as
                separation.  Optional.
            **kwargs
                All other arguments are forwarded to each call to `text` used to create
                the text labels.
            """

            if not isinstance(data, (list, np.ndarray)):
                data = im.get_array()

            # Normalize the threshold to the images color range.
            if threshold is not None:
                threshold = im.norm(threshold)
            else:
                threshold = im.norm(data.max()) / 2.0

            # Set default alignment to center, but allow it to be
            # overwritten by textkw.
            kw = dict(horizontalalignment="center", verticalalignment="center")
            kw.update(textkw)

            # Get the formatter in case a string is supplied
            if isinstance(valfmt, str):
                valfmt = ticker.StrMethodFormatter(valfmt)

            # Loop over the data and create a `Text` for each "pixel".
            # Change the text's color depending on the data.
            texts = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                    texts.append(text)

            return texts

        def shorten_number(x):
            if x == 0:
                return "0"
            elif x == 1:
                return "1"
            else:
                return f"{x:.2f}".replace("0.", ".")

        fig, (ax, ax2) = plt.subplots(1, 2)

        im, _ = heatmap(inclusion_matrix_i_in_j, rules_i, rules_j, ax=ax, cmap="magma_r", cbarlabel="this incl. in that")
        annotate_heatmap(im, valfmt=ticker.FuncFormatter(shorten_number), size=7)

        im, _ = heatmap(inclusion_matrix_j_in_i, rules_j, rules_i, ax=ax2, cmap="magma_r", cbarlabel="that incl. in this")
        annotate_heatmap(im, valfmt=ticker.FuncFormatter(shorten_number), size=7)

        # plt.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.state("zoomed")
        plt.show()

    def _do_inter_analysis(self, other_NLN, dataloaders, keep_rule_biases=False):

        nb_targets = self.torch_module.nb_out_concepts
        nb_target_values = nb_targets if nb_targets > 1 else 2
        nb_rules_i = self.torch_module.layers[-1].nb_in_concepts
        nb_rules_j = other_NLN.torch_module.layers[-1].nb_in_concepts

        cumul_presences_i = np.zeros((nb_rules_i))
        cumul_presences_j = np.zeros((nb_rules_j))
        inclusion_matrix_i_in_j = np.zeros((nb_rules_i, nb_rules_j))
        inclusion_matrix_j_in_i = np.zeros((nb_rules_j, nb_rules_i))

        self.torch_module.eval()
        other_NLN.torch_module.eval()
        for dataloader in dataloaders:
            for source, target in dataloader:
                source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device).to(torch.float32)

                tmp = self.torch_module.input_module(source)
                rule_outputs_i = self.torch_module.layers[:-1](tmp)
                if keep_rule_biases:
                    rule_outputs_without_bias_i = rule_outputs_i
                else:
                    rule_outputs_without_bias_i = rule_outputs_i / self.torch_module.layers[-2].unobserved_concepts.view(1, nb_rules_i)

                tmp = other_NLN.torch_module.input_module(source)
                rule_outputs_j = other_NLN.torch_module.layers[:-1](tmp)
                if keep_rule_biases:
                    rule_outputs_without_bias_j = rule_outputs_j
                else:
                    rule_outputs_without_bias_j = rule_outputs_j / other_NLN.torch_module.layers[-2].unobserved_concepts.view(1, nb_rules_j)

                cumul_presences_i += torch.sum(rule_outputs_without_bias_i, dim=0).detach().cpu().numpy()
                cumul_presences_j += torch.sum(rule_outputs_without_bias_j, dim=0).detach().cpu().numpy()

                for rule_idx in range(nb_rules_i):
                    for rule_jdx in range(nb_rules_j):
                        inclusion_matrix_i_in_j[rule_idx, rule_jdx] += (
                            torch.sum(
                                torch.sqrt(
                                    rule_outputs_without_bias_i[:, rule_idx] * torch.minimum(rule_outputs_without_bias_i[:, rule_idx], rule_outputs_without_bias_j[:, rule_jdx])
                                )
                            )
                            .detach()
                            .item()
                        )
                        inclusion_matrix_j_in_i[rule_jdx, rule_idx] += (
                            torch.sum(
                                torch.sqrt(
                                    rule_outputs_without_bias_j[:, rule_jdx] * torch.minimum(rule_outputs_without_bias_j[:, rule_jdx], rule_outputs_without_bias_i[:, rule_idx])
                                )
                            )
                            .detach()
                            .item()
                        )

        for rule_idx in range(nb_rules_i):
            for rule_jdx in range(nb_rules_j):
                inclusion_matrix_i_in_j[rule_idx, rule_jdx] /= cumul_presences_i[rule_idx]
                inclusion_matrix_j_in_i[rule_jdx, rule_idx] /= cumul_presences_j[rule_jdx]
        inclusion_matrix_i_in_j[inclusion_matrix_i_in_j > 1] = 1
        inclusion_matrix_j_in_i[inclusion_matrix_j_in_i > 1] = 1

        return inclusion_matrix_i_in_j, inclusion_matrix_j_in_i
