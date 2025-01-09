import pickle
import random
from statistics import mean
import time
from typing import Union
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryROC
from NLN_Dataset import NLNTabularDataset
from NLN_Logging import close_log_files, get_log_files, print_log, gpu_mem_to_string, printProgressBar
from NLN_Modules import NB_RULES, CATEGORY_CONCEPT_MULTIPLIER, NB_DICHOTOMIES_PER_CONTINUOUS, RANDOM_INIT_DIR, RANDOM_INIT_INDIR, VERBOSE, NeuralLogicNetwork

MIN_NB_TRAINING_STEPS_BEFORE_REVIEW = 8


class NLN:
    """
    Neural Logic Network (NLN)
    """

    def __init__(
        self,
        NLN_filename: str,
        train_dataset: NLNTabularDataset,
        val_dataset: Union[NLNTabularDataset, None] = None,
        test_dataset: Union[NLNTabularDataset, None] = None,
        criterion: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        nb_concepts_per_hidden_layer: int = NB_RULES,
        adam_lr: float = 1e-3,
        train_nb_epochs: int = 3000,
        retrain_nb_epochs: int = 1000,
        do_train: bool = True,
        do_quantize: bool = True,
        do_retrain: bool = True,
        do_prune: bool = True,
        do_reorder: bool = True,
        do_evaluate_on_full_dataset: bool = True,
        do_plot_stats_full_dataset: bool = True,
        do_evaluate_on_test_dataset: bool = True,
        do_plot_stats_test_dataset: bool = True,
        quantize_type: str = "sel_desc",
        nb_hidden_layers: int = 1,
        last_layer_is_OR_no_neg: bool = True,
        category_concept_multiplier: float = CATEGORY_CONCEPT_MULTIPLIER,
        nb_dichotomies_per_continuous: int = NB_DICHOTOMIES_PER_CONTINUOUS,
        nb_intervals_per_continuous: Union[int, None] = None,
        nb_out_concepts_per_continuous: Union[int, None] = None,
        random_init_dir: bool = RANDOM_INIT_DIR,
        random_init_indir: bool = RANDOM_INIT_INDIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = VERBOSE,
        do_log: bool = False,
        do_save_intermediate_learning_model_figures: bool = True,
        do_save_intermediate_training_model_figures: bool = False,
        log_model_every_training_epoch: bool = False,
        init_string="",
    ):
        self.filename = NLN_filename
        if val_dataset != None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.train_val_dataset = NLNTabularDataset.merge(train_dataset, val_dataset)
        else:
            self.train_val_dataset = train_dataset
            self.train_dataset, self.val_dataset = NLNTabularDataset.split_into_train_val(train_dataset, random_seed=None)
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=0)
        self.train_val_loader = DataLoader(self.train_val_dataset, batch_size=128, shuffle=False, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=0)
        self.test_loader = None if self.test_dataset == None else DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=0)
        self.criterion = criterion
        self.adam_lr = adam_lr
        self.train_nb_epochs = train_nb_epochs
        self.retrain_nb_epochs = retrain_nb_epochs
        self.do_train = do_train
        self.do_quantize = do_quantize
        self.do_retrain = do_retrain
        self.do_prune = do_prune
        self.do_reorder = do_reorder
        self.do_evaluate_on_full_dataset = do_evaluate_on_full_dataset
        self.do_plot_stats_full_dataset = do_plot_stats_full_dataset
        self.do_evaluate_on_test_dataset = do_evaluate_on_test_dataset and test_dataset != None
        self.do_plot_stats_test_dataset = do_plot_stats_test_dataset and test_dataset != None
        if quantize_type not in ["sel_desc", "sel_asc", "sub", "add", "thresh"]:
            raise Exception('Undefined quantize_type. The only possible quantize_types are "sel_desc", "sel_asc", "sub", "add" and "thresh".')
        self.quantize_type = quantize_type
        self.verbose = verbose
        self.do_log = do_log
        self.do_save_intermediate_learning_model_figures = do_save_intermediate_learning_model_figures
        self.do_save_intermediate_training_model_figures = do_save_intermediate_training_model_figures
        self.log_model_every_training_epoch = log_model_every_training_epoch
        self.torch_module = NeuralLogicNetwork(
            train_dataset.nb_features,
            train_dataset.nb_class_targets,
            nb_concepts_per_hidden_layer=nb_concepts_per_hidden_layer,
            nb_hidden_layers=nb_hidden_layers,
            last_layer_is_OR_no_neg=last_layer_is_OR_no_neg,
            category_first_last_pairs=train_dataset.category_first_last_pairs,
            continuous_index_min_max_triples=train_dataset.continuous_index_min_max_triples,
            periodic_index_period_pairs=train_dataset.periodic_index_period_pairs,
            column_names=train_dataset.column_names,
            category_concept_multiplier=category_concept_multiplier,
            nb_dichotomies_per_continuous=nb_dichotomies_per_continuous,
            nb_intervals_per_continuous=nb_intervals_per_continuous,
            nb_out_concepts_per_continuous=nb_out_concepts_per_continuous,
            random_init_dir=random_init_dir,
            random_init_indir=random_init_indir,
            device=device,
            verbose=verbose,
            init_string=init_string,
        )

    def save(self, filename):
        pickle.dump(self, open(filename + ".pkl", "wb"))

    @staticmethod
    def load(filename):
        if filename[-4:] == ".pkl":
            return pickle.load(open(filename, "rb"))
        else:
            return pickle.load(open(filename + ".pkl", "rb"))

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
        filename = self.filename
        print(f"Learning {filename}...")

        timing_files = get_log_files(filename + "_AllTimings", self.do_log)

        if self.do_train:
            if self.verbose:
                print(f"\nInitial training...")
            else:
                print()
                printProgressBar(0, self.train_nb_epochs, prefix=f"Initial training...  ", suffix="", length=50)
            full_optimizer = torch.optim.Adam(self.torch_module.parameters(), lr=self.adam_lr)
            start = time.time()
            self._train(
                full_optimizer,
                self.train_nb_epochs,
                True,
                filename=filename,
                max_nb_epochs_since_last_best=500,
                save_model_figures=self.do_save_intermediate_training_model_figures,
                show_loss_time=True,
            )
            train_time = time.time() - start
            print_log(f"Initial training time: {train_time:.3e} seconds", True, timing_files)

            filename += "_trained"

            train_val_loss = self._eval_train_val_loss()
            print(f"Full training set loss (train. + val.) after initial training = {train_val_loss:.5g}")
            if self.do_save_intermediate_learning_model_figures:
                print("Saving plot...")
                self.show(filename=filename, valid_raw_loss=train_val_loss)

        post_process_start = time.time()

        if self.do_quantize:
            if self.quantize_type == "thresh":
                print(f"\nQuantizing...")
                filename += "_quantizedThresh"

                start = time.time()
                self.torch_module.quantize()
                quantize_time = time.time() - start

            else:
                if self.verbose:
                    print(f"\nQuantizing...")
                    progress_bar_hook = lambda iteration, nb_iterations: None
                else:
                    print()
                    printProgressBar(0, 1, prefix=f"Quantizing...  ", suffix="", length=50)
                    progress_bar_hook = lambda iteration, nb_iterations: printProgressBar(iteration, nb_iterations, prefix=f"Quantizing...  ", suffix="", length=50)
                if self.quantize_type == "sub":
                    filename += "_quantizedSub"
                elif self.quantize_type == "add":
                    filename += "_quantizedAdd"
                elif self.quantize_type == "sel_asc":
                    filename += "_quantizedSelAsc"
                elif self.quantize_type == "sel_desc":
                    filename += "_quantizedSelDesc"

                start = time.time()
                losses = self.torch_module.new_quantize(
                    self._eval_train_val_loss,
                    quantize_type=self.quantize_type,
                    goes_backward=True,
                    goes_by_layer=True,
                    filename=filename,
                    do_log=self.do_log,
                    progress_bar_hook=progress_bar_hook,
                )
                quantize_time = time.time() - start

            print_log(f"Quantizing time: {quantize_time:.3e} seconds", True, timing_files)
            self.save(self.filename)

            train_val_loss = self._eval_train_val_loss()
            print(f"Full training set loss (train. + val.) after quantizing = {train_val_loss:.5g}")
            if self.do_save_intermediate_learning_model_figures and torch.max(self.torch_module.layers[-1].observed_concepts).item() > 0:
                print("Saving plot...")
                self.show(filename=filename, valid_raw_loss=train_val_loss)

                # if self.quantize_type != "thresh":
                #     plt.close("all")
                #     fig, ax = plt.subplots(1, 1, num=2)
                #     ax.semilogy(list(range(len(losses))), losses)
                #     ax.set_title("Prune&Quant. " + self.quantize_type)
                #     plt.show()

        if not self.do_train and not self.do_quantize and self.do_retrain:
            train_val_loss = self._eval_train_val_loss()
        if self.do_retrain and train_val_loss > 0:
            if self.verbose:
                print(f"\nContinuous parameters retraining...")
            else:
                print()
                printProgressBar(0, self.retrain_nb_epochs, prefix=f"Continuous parameters retraining...  ", suffix="", length=50)
            filename += "_retrain"

            if self.do_train or self.do_quantize:
                self = NLN.load(self.filename)
            discrete_parameters, continuous_parameters = self.torch_module.get_discrete_continuous_parameters()
            continuous_optimizer = torch.optim.Adam(continuous_parameters, lr=self.adam_lr)
            start = time.time()
            self._train(
                continuous_optimizer,
                self.retrain_nb_epochs,
                False,
                filename=filename,
                max_nb_epochs_since_last_best=100,
                save_model_figures=False,
                show_loss_time=True,
            )
            retrain_time = time.time() - start
            print_log(f"Continuous parameters retraining time: {retrain_time:.3e} seconds", True, timing_files)

            train_val_loss = self._eval_train_val_loss()
            print(f"Full training set loss (train. + val.) after continuous retraining = {train_val_loss:.5g}")
            if self.do_save_intermediate_learning_model_figures and torch.max(self.torch_module.layers[-1].observed_concepts).item() > 0:
                print("Saving plot...")
                self.show(filename=filename, valid_raw_loss=train_val_loss)

        if self.do_prune:
            if self.verbose:
                print(f"\nPruning...")
                progress_bar_hook = lambda iteration, nb_iterations: None
            else:
                print()
                printProgressBar(0, 1, prefix=f"Pruning...  ", suffix="", length=50)
                progress_bar_hook = lambda iteration, nb_iterations: printProgressBar(iteration, nb_iterations, prefix=f"Pruning...  ", suffix="", length=50)
            filename += "_pruned"

            start = time.time()
            self.torch_module.prune(self._eval_train_val_loss, filename=filename, do_log=self.do_log, progress_bar_hook=progress_bar_hook)
            prune_time = time.time() - start
            print_log(f"Pruning time: {prune_time:.3e} seconds", True, timing_files)
            self.save(self.filename)

            train_val_loss = self._eval_train_val_loss()
            print(f"Full training set loss (train. + val.) after pruning = {train_val_loss:.5g}")
            if self.do_save_intermediate_learning_model_figures and torch.max(self.torch_module.layers[-1].observed_concepts).item() > 0:
                print("Saving plot...")
                self.show(filename=filename, valid_raw_loss=train_val_loss)

        if self.do_reorder and torch.max(self.torch_module.layers[-1].observed_concepts).item() > 0:
            print(f"\nReordering...")
            filename += "_reordered"

            start = time.time()
            self.torch_module.rearrange_for_visibility()
            rearrange_time = time.time() - start
            print_log(f"Reordering time: {rearrange_time:.3e} seconds", True, timing_files)
            self.save(self.filename)

            # print_log(self.torch_module, True, [])

            train_val_loss = self._eval_train_val_loss()
            print(f"Full training set loss (train. + val.) after learning = {train_val_loss:.5g}")
            print("Saving final plots...")
            self.show(filename=filename, valid_raw_loss=train_val_loss)
            self.show(filename=filename, one_rule_at_a_time=True)

        post_process_time = time.time() - post_process_start
        print_log(f"PostProcessing:\t{post_process_time:.3e}", True, timing_files)

        if self.do_evaluate_on_full_dataset or self.do_plot_stats_full_dataset:
            print(f"\nEvaluating on full dataset (train. + val. + test)...")
            full_dataset_evaluation_log_files = get_log_files(filename + "_fullDatasetEvaluation", True)

            if self.do_evaluate_on_full_dataset:
                start = time.time()
                self.full_loss = self._validate(self.train_val_loader, extra_dataloader=self.test_loader, override="raw")[0]
                full_eval_time = time.time() - start
                print_log(
                    "Full dataset loss: " + str(self.full_loss),
                    True,
                    full_dataset_evaluation_log_files,
                )
                print_log(f"Full dataset inference time: {full_eval_time:.3e} seconds", True, timing_files)

            if self.do_plot_stats_full_dataset:
                res = self._plot_stats(self.train_val_loader, extra_dataloader=self.test_loader, log_files=full_dataset_evaluation_log_files, filename=filename + "_full")
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
                self.test_loss = self._validate(self.test_loader, override="raw")[0]
                test_eval_time = time.time() - start
                print_log(
                    "Test loss: " + str(self.test_loss),
                    True,
                    test_dataset_evaluation_log_files,
                )
                print_log(f"Test dataset inference time: {test_eval_time:.3e} seconds", True, timing_files)

            if self.do_plot_stats_test_dataset:
                res = self._plot_stats(self.test_loader, log_files=test_dataset_evaluation_log_files, filename=filename + "_full")
                if not self.train_dataset.is_multi_label:
                    self.test_acc, self.test_f1 = res
                else:
                    self.test_acc, self.test_all_correct_acc, self.test_f1 = res

            close_log_files(test_dataset_evaluation_log_files)

        close_log_files(timing_files)

        if (
            self.do_save_intermediate_learning_model_figures
            or self.do_save_intermediate_training_model_figures
            or self.do_plot_stats_full_dataset
            or self.do_plot_stats_test_dataset
        ):
            plt.close("all")

    def _eval_train_val_loss(self):
        return self._validate(self.train_val_loader, override="raw")[0]

    def _train(
        self,
        optimizer,
        nb_epochs,
        do_train_weights,
        filename="",
        max_nb_epochs_since_last_best=500,
        save_model_figures=True,
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

        if len(filename) > 0 and save_model_figures:
            start = time.time()
            self.show(filename=filename + "_epoch0", train_loss=train_loss, train_no_reg_loss=train_no_reg_loss, valid_raw_loss=valid_raw_loss, valid_thresh_loss=valid_thresh_loss)
            show_time = time.time() - start
            print(f"Show Time: {show_time:.3f}")

        best_valid_raw_loss = valid_raw_loss
        best_model_raw_string = model_string
        best_raw_epoch = 0
        self.save(self.filename)
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
                    f"Epoch: {epoch} | GPU: {gpu_mem_to_string(gpu_mem)} | Trn. (reg., no), Val. (raw, thresh): {train_loss:.3e}, {train_no_reg_loss:.3e}, {valid_raw_loss:.3e}, {valid_thresh_loss:.3e} | Trn., Val. (raw, thresh) Time: {train_time:.3f}, {valid_raw_time:.3f}, {valid_thresh_time:.3f} | Model: ",
                    self.verbose,
                    files=log_files,
                )
            else:
                print_log(
                    f"Epoch: {epoch} | GPU: {gpu_mem_to_string(gpu_mem)} | Trn., Val.: {train_no_reg_loss:.3e}, {valid_raw_loss:.3e} | Trn., Val. Time: {train_time:.3f}, {valid_raw_time:.3f} | Model: ",
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

            if len(filename) > 0 and save_model_figures and (epoch < 10 or (epoch < 100 and epoch % 10 == 0) or epoch % 100 == 0):
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
                self.save(self.filename)
                # if not save_full_log:
                #   print_log(model_string, False, files=log_files)
            if do_train_weights:
                if valid_thresh_loss < best_valid_thresh_loss:
                    best_valid_thresh_loss = valid_thresh_loss
                    best_thresh_epoch = epoch

            if epoch - best_raw_epoch == max_nb_epochs_since_last_best or valid_raw_loss == 0.0:
                if not self.verbose:
                    if do_train_weights:
                        printProgressBar(self.train_nb_epochs, self.train_nb_epochs, prefix=f"Initial training...  ", suffix="", length=50)
                    else:
                        printProgressBar(self.retrain_nb_epochs, self.retrain_nb_epochs, prefix=f"Continuous parameters retraining...  ", suffix="", length=50)
                break

        actual_nb_epochs = len(results["train_no_reg_losses"]) - 1

        print_log(f"Best Validation (Raw) Model from Epoch {best_raw_epoch} :\n", self.verbose, files=log_files)
        print_log(best_model_raw_string, self.verbose, files=log_files)

        self.torch_module.load_string(best_model_raw_string)

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

    def _train_epoch(self, dataloader, optimizer, do_train_weights, epoch, last_training_step_counter):
        train_no_reg_loss = 0.0
        train_loss = 0.0
        self.torch_module.train()
        start = time.time()

        if do_train_weights:
            # grads = torch.zeros((self.torch_module.layers[0].nb_out_concepts), device="cuda")

            if last_training_step_counter[0] == 0:
                self.torch_module.activate_memory()

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

            # if do_train_weights:
            #     grads = torch.maximum(grads, torch.max(torch.abs(self.torch_module.layers[0].observed_concepts.grad), dim=1)[0])

            optimizer.step()

            self.torch_module.update_parameters()

            last_training_step_counter[0] += 1

        # if do_train_weights and self.verbose:
        #     not_learning_concepts = [idx for idx, max_grad in enumerate(grads.detach().cpu().numpy().tolist()) if max_grad < 1e-6]
        #     if len(not_learning_concepts) > 0:
        #         if len(not_learning_concepts) == 1:
        #             display_string = "concept w/ grad<1e-6: "
        #         else:
        #             display_string = "concepts w/ grad<1e-6: "
        #         for i, useless_concept in enumerate(not_learning_concepts):
        #             if i == 0:
        #                 display_string += str(useless_concept)
        #             else:
        #                 display_string += ", " + str(useless_concept)
        #         print(display_string)

        if do_train_weights and last_training_step_counter[0] >= MIN_NB_TRAINING_STEPS_BEFORE_REVIEW:
            self.torch_module.review_and_shut_memory()
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
                all_corrects = self._validate_with_threshold(dataloader, extra_dataloader=extra_dataloader, multi_label=True)
                dataloaders = [dataloader] if extra_dataloader == None else [dataloader, extra_dataloader]
                nb_samples = sum([len(curr_dataloader) for curr_dataloader in dataloaders])
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
            return all_corrects

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
                source, target = source.to(self.torch_module.device).to(torch.float32), target.to(self.torch_module.device)

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
