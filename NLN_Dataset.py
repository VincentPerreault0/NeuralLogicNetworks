from math import ceil
from pathlib import Path
import random
from typing import List, Tuple, Union, Dict
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

TRAIN_VAL_SPLIT = 0.8


def print_df(df, column_values=False):
    print(df.info())
    print()
    if column_values:
        for column in df.columns:
            print(column + ":")
            print(df[column].value_counts())
            print()
    print(df.describe(include="all"))
    print()
    print(df.head)
    print("\n\n")


class NLNTabularDataset(Dataset):
    """NLN tabular dataset."""

    def __init__(self, dataset_filename: str, data_frame_indices: Union[List[int], None] = None):
        if dataset_filename[-4:] != ".csv":
            raise Exception(f'Dataset filename {dataset_filename} must end in ".csv".')

        if not Path(NLNTabularDataset._get_NLN_info_txt_filename(dataset_filename)).is_file():
            if not Path(dataset_filename).is_file():
                raise Exception(f"Dataset filename {dataset_filename} does not exist. Please provide a valid filename.")

            dataset_filename, nb_class_targets, is_multi_label, targets, categories, has_missing_values, df = NLNTabularDataset._format_dataset_file(dataset_filename)
            NLNTabularDataset._create_NLN_info_txt_file(
                NLNTabularDataset._get_NLN_info_txt_filename(dataset_filename), nb_class_targets, is_multi_label, targets, categories, has_missing_values, df
            )
        else:
            if Path(dataset_filename[:-4] + "_NLN.csv").is_file():
                dataset_filename = dataset_filename[:-4] + "_NLN.csv"

        self.filename = dataset_filename
        full_data_frame = pd.read_csv(dataset_filename, skipinitialspace=True)
        self.data_frame = full_data_frame if data_frame_indices == None else full_data_frame.iloc[data_frame_indices, :]
        self.data_frame_indices = data_frame_indices
        self.column_names = self.data_frame.columns.values.tolist()

        with open(NLNTabularDataset._get_NLN_info_txt_filename(dataset_filename), "r") as feature_types_file:
            NLN_info_txt_file_lines = [line.strip() for line in feature_types_file.readlines()]

        self.nb_class_targets, self.is_multi_label = eval(NLN_info_txt_file_lines[0])
        self.nb_features = len(full_data_frame.columns) - self.nb_class_targets

        self.category_first_last_has_missing_values_tuples = eval(NLN_info_txt_file_lines[1])
        self.continuous_index_min_max_has_missing_values_tuples = eval(NLN_info_txt_file_lines[2])
        self.periodic_index_period_has_missing_values_tuples = []

    @staticmethod
    def _get_NLN_info_txt_filename(dataset_filename: str):
        dataset_filename_contains_NLN = dataset_filename[-8:-4] == "_NLN"
        return dataset_filename[:-4] + "_info.txt" if dataset_filename_contains_NLN else dataset_filename[:-4] + "_NLN_info.txt"

    @staticmethod
    def _format_dataset_file(dataset_filename: str):

        sep = NLNTabularDataset._get_csv_sep(dataset_filename)
        must_save_processed_dataset = sep != ","

        df = pd.read_csv(dataset_filename, skipinitialspace=True, sep=sep)

        columns_to_ignore = NLNTabularDataset._get_input_from_choices(
            f"All columns (input features and output targets) found in {dataset_filename}:",
            df.columns.values.tolist(),
            "column",
            "Are there any columns that should be ignored? (for instance, an ID column)",
            True,
            "The following columns will be ignored:",
            "All columns will be used.",
        )
        must_save_processed_dataset = must_save_processed_dataset or len(columns_to_ignore) > 0
        for column_to_ignore in columns_to_ignore:
            df = df.drop(column_to_ignore, axis=1)

        vars = []
        while len(vars) == 0:
            targets = NLNTabularDataset._get_input_from_choices(
                f"\n\nAll remaining columns (input features and output targets) found in {dataset_filename}:",
                df.columns.values.tolist(),
                "column",
                "Which column(s) are the output target(s) that should be predicted from the rest (input features)?",
                True,
                "The following column(s) will be output target(s):",
            )
            vars = df.columns.tolist()
            for target in targets:
                vars.remove(target)
            out_string = "\nThe following column(s) will be input feature(s):\n\t"
            for var_idx, var in enumerate(vars):
                if var_idx > 0:
                    out_string += ", "
                out_string += var
            print(out_string)
            if len(vars) == 0:
                print("\nThere was an error in the input. There must remain input feature(s) to predict the output target(s).")
        all_vars = vars + targets
        if all_vars != df.columns.tolist():
            df = df[all_vars]
            must_save_processed_dataset = True

        categories = []
        two_value_category_values_to_remove = []
        targets_value_counts = []
        one_hot_targets = []
        for target in targets:
            value_counts = df[target].value_counts()
            values = [value for value in value_counts.index]
            values = sorted(values)
            targets_value_counts.append(len(values))

            is_string_category = isinstance(values[0], str)
            is_0_1 = set(values).issubset({0, 1})

            if is_0_1:
                one_hot_targets.append(target)
            elif len(values) == 2:
                two_value_target_value_to_keep = NLNTabularDataset._get_input_from_choices(
                    f"\n\nTarget {target} has only two possible values:",
                    [target + "_" + str(value) for value in values],
                    "value",
                    "Which value should be predicted?",
                    False,
                    "The following binary target will be predicted:",
                )
                two_value_category_values_to_remove += list(set([target + "_" + str(value) for value in values]) - set(two_value_target_value_to_keep))
                categories.append((target, values))
                one_hot_targets += two_value_target_value_to_keep
            elif is_string_category:
                categories.append((target, values))
                one_hot_targets += [target + "_" + str(value) for value in values]
            else:
                out_string = f"\n\nTarget {target} is a number, but it should be either binary or categorical. Its values are:\n\t"
                for value_idx, value in enumerate(values):
                    if value_idx > 0:
                        out_string += ", "
                    out_string += f"{value:.5g}"
                    if value_idx == 29 and len(values) > 30:
                        out_string += ", ..."
                        break
                out_string += f"\n\nShould target {target} be treated as categorical with {len(values)} classes?"
                print(out_string)

                treat_continuous_as_categorical_in_string = input("Please write yes (y) or no (n): ").strip()
                while treat_continuous_as_categorical_in_string not in ["yes", "Yes", "YES", "y", "Y", "no", "No", "NO", "n", "N"]:
                    treat_continuous_as_categorical_in_string = input("There was an error in the input. Please write yes (y) or no (n): ").strip()
                treat_continuous_as_categorical = treat_continuous_as_categorical_in_string in ["yes", "Yes", "YES", "y", "Y"]

                if treat_continuous_as_categorical:
                    categories.append((target, values))
                    one_hot_targets += [target + "_" + str(value) for value in values]
                else:
                    raise Exception(
                        "The current Neural Logic Network (NLN) can only classify one or more binary targets (single- or multi-label) or between >2 classes (multi-class)."
                    )

            if not is_0_1 and len(values) < 2:
                raise Exception(f"The current Neural Logic Network (NLN) cannot classify a target with {len(values)} class. Target {target} for possible values {values}.")

        if len(targets) > 1 and max(targets_value_counts) > 2:
            raise Exception(
                "The current Neural Logic Network (NLN) cannot do multi-label (more than one targets) and multi-class (more than two classes per target) classification simultaneously."
            )

        is_multi_label = len(targets) > 1
        if is_multi_label:
            nb_class_targets = len(targets)
        else:
            nb_class_targets = 1 if targets_value_counts[0] == 2 else targets_value_counts[0]

        has_missing_values_dict = dict()
        for var in vars:
            value_counts = df[var].value_counts()
            values = [value for value in value_counts.index]
            values = sorted(values)

            is_string_category = isinstance(values[0], str)
            is_0_1 = set(values) == {0, 1}
            has_missing_values = bool(df[var].isnull().any()) or "?" in values

            has_missing_values_dict[var] = has_missing_values

            if is_0_1:
                is_category = False
            elif len(values) == 2:
                if not has_missing_values:
                    two_value_feature_value_to_keep = NLNTabularDataset._get_input_from_choices(
                        f"\n\nFeature {var} has only two possible values:",
                        [var + "_" + str(value) for value in values],
                        "value",
                        "Which value should be used as the positive case?",
                        False,
                        "The following binary feature will be used:",
                    )
                    two_value_category_values_to_remove += list(set([var + "_" + str(value) for value in values]) - set(two_value_feature_value_to_keep))
                is_category = True
            elif is_string_category:
                is_category = True
            elif len(values) <= 30:
                out_string = f"\n\nFeature {var} is a number, but it has only {len(values)} possible values. Its values are:\n\t"
                for value_idx, value in enumerate(values):
                    if value_idx > 0:
                        out_string += ", "
                    out_string += f"{value:.5g}"
                out_string += f"\n\nShould feature {var} be treated as categorical with {len(values)} possible values (yes) or remain a number (no)?"
                print(out_string)

                treat_continuous_as_categorical_in_string = input("Please write yes (y) or no (n): ").strip()
                while treat_continuous_as_categorical_in_string not in ["yes", "Yes", "YES", "y", "Y", "no", "No", "NO", "n", "N"]:
                    treat_continuous_as_categorical_in_string = input("There was an error in the input. Please write yes (y) or no (n): ").strip()
                is_category = treat_continuous_as_categorical_in_string in ["yes", "Yes", "YES", "y", "Y"]
            else:
                is_category = False

            if is_category:
                if has_missing_values:
                    df[var].fillna("?", inplace=True)
                    df[var] = df[var].apply(lambda val: str(val) if not isinstance(val, float) or round(val) != val else str(int(val)))
                    value_counts = df[var].value_counts()
                    values = [value for value in value_counts.index]
                    values = sorted(values)
                    two_value_category_values_to_remove += [var + "_?"]
                categories.append((var, values))

        if len(categories) > 0:
            must_save_processed_dataset = True

            category_names = [category[0] for category in categories]

            df[category_names] = df[category_names].astype("category")
            for column_name, column_values in categories:
                df[column_name] = df[column_name].cat.set_categories(column_values)
                df[column_name] = df[column_name].cat.set_categories(column_values)

            # print_df(df, column_values=True)

            df = pd.get_dummies(df, dtype=int)  # one-hot encoding

            one_hot_all_vars = [column for column in df.columns if column not in two_value_category_values_to_remove]
            one_hot_vars = 1 * one_hot_all_vars
            for one_hot_target in one_hot_targets:
                one_hot_vars.remove(one_hot_target)
            one_hot_all_vars = one_hot_vars + one_hot_targets
            df = df[one_hot_all_vars]

            # print_df(df)

        if must_save_processed_dataset:
            dataset_filename = dataset_filename[:-4] + "_NLN.csv"
            df.to_csv(dataset_filename, encoding="utf-8", index=False)

        return dataset_filename, nb_class_targets, is_multi_label, targets, categories, has_missing_values_dict, df

    @staticmethod
    def _get_csv_sep(dataset_filename: str):
        possible_seps = [",", ";", " "]

        with open(dataset_filename, "r") as dataset_file:
            dataset_file_lines = [line.strip() for line in dataset_file.readlines()]

        possible_seps_counts = [[] for possible_sep in possible_seps]
        for dataset_file_line in dataset_file_lines:
            if dataset_file_line != "\n" and dataset_file_line != "":
                for possible_sep_idx, possible_sep in enumerate(possible_seps):
                    possible_seps_counts[possible_sep_idx].append(dataset_file_line.count(possible_sep))

        remaining_possible_sep_idcs = []
        for possible_sep_idx in range(len(possible_seps)):
            if len(set(possible_seps_counts[possible_sep_idx])) == 1:
                remaining_possible_sep_idcs.append(possible_sep_idx)
        if len(remaining_possible_sep_idcs) == 0:
            raise Exception('Dataset csv file {dataset_filename} is not formatted properly. The separator could not be found among [",", ";", " "].')

        max_count_value = -1
        for possible_sep_idx in remaining_possible_sep_idcs:
            possible_sep_count = list(set(possible_seps_counts[possible_sep_idx]))[0]
            if possible_sep_count > max_count_value:
                max_count_value = possible_sep_count
                sep = possible_seps[possible_sep_idx]
        if max_count_value <= 0:
            raise Exception('Dataset csv file {dataset_filename} is not formatted properly. The separator could not be found among [",", ";", " "].')

        return sep

    @staticmethod
    def _get_input_from_choices(
        pre_choices_message: str,
        choices: List[str],
        choice_label: str,
        post_choices_message: str,
        multiple_is_possible: bool,
        accepted_message: str,
        none_selected_message: str = "",
    ):
        out_string = pre_choices_message + "\n\t"
        for choice_idx, choice in enumerate(choices):
            if choice_idx > 0:
                out_string += ", "
            out_string += choice
        out_string += "\n\n" + post_choices_message
        print(out_string)

        input_message = "Please write them here (if multiple, separated by commas)" if multiple_is_possible else "Please write it here"
        input_message += " (if none, leave blank and press ENTER): " if none_selected_message != "" else ": "
        selected_choices_in_string = input(input_message)

        attempted_selected_choices = [
            attempted_selected_choice.strip() for attempted_selected_choice in selected_choices_in_string.split(",") if attempted_selected_choice.strip() != ""
        ]
        selected_choices = []
        attempted_selected_choices_with_errors = []
        for attempted_selected_choice in attempted_selected_choices:
            if attempted_selected_choice in choices:
                if attempted_selected_choice not in selected_choices:
                    selected_choices.append(attempted_selected_choice)
            else:
                attempted_selected_choices_with_errors.append(attempted_selected_choice)

        if len(selected_choices) > 0:
            out_string = "\n" + accepted_message + "\n\t"
            for selected_choice_idx, selected_choice in enumerate(selected_choices):
                if selected_choice_idx > 0:
                    out_string += ", "
                out_string += selected_choice
            print(out_string)

        while len(attempted_selected_choices_with_errors) > 0 or (none_selected_message == "" and len(selected_choices) == 0):
            if len(attempted_selected_choices_with_errors) > 0:
                out_string = f"\nThere was an error in the input. The following {choice_label}s were not recognized:\n\t"
                for attempted_selected_choice_with_error_idx, attempted_selected_choice_with_error in enumerate(attempted_selected_choices_with_errors):
                    if attempted_selected_choice_with_error_idx > 0:
                        out_string += ", "
                    out_string += attempted_selected_choice_with_error
                out_string += "\n"
            else:
                out_string = f"\nThere was an error in the input. A {choice_label} must be selected. "
            out_string += f"\nThe available {choice_label}s are:\n\t"
            for choice_idx, choice in enumerate(choices):
                if choice not in selected_choices:
                    if choice_idx > 0:
                        out_string += ", "
                    out_string += choice
            print(out_string)

            selected_choices_in_string = input(input_message)

            attempted_selected_choices = [
                attempted_selected_choice.strip() for attempted_selected_choice in selected_choices_in_string.split(",") if attempted_selected_choice.strip() != ""
            ]
            attempted_selected_choices_with_errors = []
            for attempted_selected_choice in attempted_selected_choices:
                if attempted_selected_choice in choices:
                    if attempted_selected_choice not in selected_choices:
                        selected_choices.append(attempted_selected_choice)
                else:
                    attempted_selected_choices_with_errors.append(attempted_selected_choice)

            if len(selected_choices) > 0:
                out_string = "\n" + accepted_message + "\n\t"
                for selected_choice_idx, selected_choice in enumerate(selected_choices):
                    if selected_choice_idx > 0:
                        out_string += ", "
                    out_string += selected_choice
                print(out_string)

        if len(selected_choices) == 0:
            out_string = "\n" + none_selected_message
            print(out_string)

        return selected_choices

    @staticmethod
    def _create_NLN_info_txt_file(
        NLN_info_txt_filename: str,
        nb_class_targets: int,
        is_multi_label: bool,
        targets: List[str],
        categories: List[Tuple[str, List[Union[str, int, float]]]],
        has_missing_values: Dict[str, bool],
        df: pd.DataFrame,
    ):
        # print((nb_class_targets, is_multi_label))

        categories_without_binary = [(category, values) for category, values in categories if len(values) > 2]
        one_hot_category_values_without_binary = [[category + "_" + str(value) for value in values] for category, values in categories_without_binary]
        category_first_last_has_missing_values_tuples = []
        column_names = df.columns.values.tolist()
        current_category_name = ""
        in_category = False
        for idx, column_name in enumerate(column_names):
            if "_" in column_name:
                category_name = ""
                for category_idx in range(len(categories_without_binary)):
                    for one_hot_category_value_without_binary in one_hot_category_values_without_binary[category_idx]:
                        if column_name == one_hot_category_value_without_binary:
                            category_name = categories_without_binary[category_idx][0]
                            break
                    if category_name != "":
                        break
                if category_name != "":
                    if category_name == current_category_name:
                        in_category = True
                    else:
                        if in_category:
                            if current_category_name not in targets:
                                category_first_last_has_missing_values_tuples.append((current_category_first_idx, idx - 1, has_missing_values[current_category_name]))
                            in_category = False
                        current_category_name = category_name
                        current_category_first_idx = idx
                else:
                    if in_category and current_category_name not in targets:
                        category_first_last_has_missing_values_tuples.append((current_category_first_idx, idx - 1, has_missing_values[current_category_name]))
                    current_category_name = ""
                    in_category = False
            else:
                if in_category and current_category_name not in targets:
                    category_first_last_has_missing_values_tuples.append((current_category_first_idx, idx - 1, has_missing_values[current_category_name]))
                current_category_name = ""
                in_category = False
        idx = len(column_names)
        if in_category and current_category_name not in targets:
            category_first_last_has_missing_values_tuples.append((current_category_first_idx, idx - 1, has_missing_values[current_category_name]))
        # print(category_first_last_pairs)

        continuous_index_min_max_has_missing_values_tuples = []
        for idx, column_name in enumerate(column_names):
            value_counts = df[column_name].value_counts()
            is_continuous = False
            min = float("inf")
            max = float("-inf")
            for value in value_counts.index:
                if isinstance(value, (int, float)):
                    if value < min:
                        min = value
                    if value > max:
                        max = value
                    if value != 0 and value != 1:
                        is_continuous = True
                else:
                    is_continuous = False
                    break
            if is_continuous:
                continuous_index_min_max_has_missing_values_tuples.append((idx, min, max, has_missing_values[column_name]))
        # print(continuous_index_min_max_triples)

        with open(NLN_info_txt_filename, "w", encoding="utf-8") as NLN_info_txt_file:
            NLN_info_txt_file.write(str((nb_class_targets, is_multi_label)) + "\n")
            NLN_info_txt_file.write(str(category_first_last_has_missing_values_tuples) + "\n")
            NLN_info_txt_file.write(str(continuous_index_min_max_has_missing_values_tuples))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        items = self.data_frame.iloc[idx, :]
        items = np.asarray(items)
        if isinstance(idx, list):
            return (items[:, : self.nb_features], items[:, self.nb_features :])
        else:
            return (items[: self.nb_features], items[self.nb_features :])

    @staticmethod
    def merge(datasets):
        datasets = [dataset for dataset in datasets if dataset != None]
        if len(datasets) == 0:
            return None
        else:
            if len(set([dataset.filename for dataset in datasets])) > 1:
                raise Exception("The merged datasets must be slices of the same csv file.")
            else:
                dataset_filename = datasets[0].filename
            if None in [dataset.data_frame_indices for dataset in datasets]:
                data_frame_indices = None
            else:
                data_frame_indices = set()
                for dataset in datasets:
                    data_frame_indices = data_frame_indices | set(dataset.data_frame_indices)
                data_frame_indices = sorted(list(data_frame_indices))
            return NLNTabularDataset(dataset_filename, data_frame_indices)

    @staticmethod
    def split_into_train_val(merged_dataset, train_val_split=TRAIN_VAL_SPLIT, random_seed=None):
        dataset_filename = merged_dataset.filename
        data_frame_indices = merged_dataset.data_frame_indices if merged_dataset.data_frame_indices != None else list(range(len(merged_dataset.data_frame)))
        if random_seed != None:
            random.seed(random_seed)
        random.shuffle(data_frame_indices)
        train_indices = data_frame_indices[: round(train_val_split * len(data_frame_indices))]
        val_indices = data_frame_indices[round(train_val_split * len(data_frame_indices)) :]
        return NLNTabularDataset(dataset_filename, train_indices), NLNTabularDataset(dataset_filename, val_indices)

    @staticmethod
    def k_fold_split_into_train_val_test(dataset_filename: str, fold_idx: int, nb_folds: int, random_seed=0, train_val_split=TRAIN_VAL_SPLIT, incomplete_dataset_ratio=1):
        full_data_frame = pd.read_csv(dataset_filename)
        nb_items = len(full_data_frame)
        full_data_frame_indices = list(range(nb_items))
        if random_seed != None:
            random.seed(random_seed)
        random.shuffle(full_data_frame_indices)
        if incomplete_dataset_ratio < 1:
            full_data_frame_indices = full_data_frame_indices[: int(round(incomplete_dataset_ratio * nb_items))]
            nb_items = len(full_data_frame_indices)

        if fold_idx > 0:
            train_val_indices = full_data_frame_indices[: round((fold_idx / nb_folds) * nb_items)]
        else:
            train_val_indices = []
        if fold_idx < nb_folds - 1:
            train_val_indices += full_data_frame_indices[round(((fold_idx + 1) / nb_folds) * nb_items) :]
        train_indices = train_val_indices[: round(train_val_split * len(train_val_indices))]
        val_indices = train_val_indices[round(train_val_split * len(train_val_indices)) :]
        test_indices = full_data_frame_indices[round((fold_idx / nb_folds) * nb_items) : round(((fold_idx + 1) / nb_folds) * nb_items)]

        train_dataset = NLNTabularDataset(dataset_filename, train_indices)
        val_dataset = NLNTabularDataset(dataset_filename, val_indices)
        test_dataset = NLNTabularDataset(dataset_filename, test_indices)
        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def k_fold_split_into_train_test(dataset_filename: str, fold_idx: int, nb_folds: int, random_seed=0, incomplete_dataset_ratio=1):
        full_data_frame = pd.read_csv(dataset_filename)
        nb_items = len(full_data_frame)
        full_data_frame_indices = list(range(nb_items))
        if random_seed != None:
            random.seed(random_seed)
        random.shuffle(full_data_frame_indices)
        if incomplete_dataset_ratio < 1:
            full_data_frame_indices = full_data_frame_indices[: int(round(incomplete_dataset_ratio * nb_items))]
            nb_items = len(full_data_frame_indices)

        if fold_idx > 0:
            train_indices = full_data_frame_indices[: round((fold_idx / nb_folds) * nb_items)]
        else:
            train_indices = []
        if fold_idx < nb_folds - 1:
            train_indices += full_data_frame_indices[round(((fold_idx + 1) / nb_folds) * nb_items) :]
        test_indices = full_data_frame_indices[round((fold_idx / nb_folds) * nb_items) : round(((fold_idx + 1) / nb_folds) * nb_items)]

        train_dataset = NLNTabularDataset(dataset_filename, train_indices)
        test_dataset = NLNTabularDataset(dataset_filename, test_indices)
        return train_dataset, test_dataset
