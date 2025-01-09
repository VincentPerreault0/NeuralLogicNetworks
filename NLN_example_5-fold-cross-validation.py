from statistics import mean
from NLN_Dataset import NLNTabularDataset
from NLN_Logging import close_log_files, get_log_files, print_log
from NLN_Model import NLN

# IN ORDER FOR THIS METHOD TO BE INTERPRETABLE, THE FIRST LINE OF THE DATASET SHOULD ALWAYS BE THE COLUMN LABELS.
dataset_filename = "datasets/tic-tac-toe.csv"
output_filename = "results/tic-tac-toe"

k_fold_split_random_seed = 0

full_loss_values = []
full_acc_values = []
full_all_correct_acc_values = []
full_f1_values = []
test_loss_values = []
test_acc_values = []
test_all_correct_acc_values = []
test_f1_values = []
for fold_idx in range(5):

    train_dataset, val_dataset, test_dataset = NLNTabularDataset.k_fold_split_into_train_val_test(dataset_filename, fold_idx, 5, random_seed=k_fold_split_random_seed)

    NLN.fix_random_seed()

    model = NLN(f"{output_filename}_{fold_idx}", train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

    model.learn()

    ## To retrieve the i-th model, use
    # model = NLN.load(f"{output_filename}_{i}")

    full_loss_values.append(model.full_loss)
    full_acc_values.append(model.full_acc)
    full_f1_values.append(model.full_f1)
    test_loss_values.append(model.test_loss)
    test_acc_values.append(model.test_acc)
    test_f1_values.append(model.test_f1)
    if train_dataset.is_multi_label:
        full_all_correct_acc_values.append(model.full_all_correct_acc)
        test_all_correct_acc_values.append(model.test_all_correct_acc)

summary_files = get_log_files(output_filename + "_SUMMARY", True)
print_log("SUMMARY\n-------\n", True, summary_files)
print_log(f"FULL MSE:         {mean(full_loss_values):#.4g}", True, summary_files)
print_log(f"FULL ACCURACY:    {mean(full_acc_values):#.4g}", True, summary_files)
if len(full_all_correct_acc_values) > 0:
    print_log(f"FULL ALL COR. ACC.{mean(full_all_correct_acc_values):#.4g}", True, summary_files)
print_log(f"FULL F1:          {mean(full_f1_values):#.4g}\n", True, summary_files)
print_log(f"TEST MSE:         {mean(test_loss_values):#.4g}", True, summary_files)
print_log(f"TEST ACCURACY:    {mean(test_acc_values):#.4g}", True, summary_files)
if len(test_all_correct_acc_values) > 0:
    print_log(f"TEST ALL COR. ACC.{mean(test_all_correct_acc_values):#.4g}", True, summary_files)
print_log(f"TEST F1:          {mean(test_f1_values):#.4g}", True, summary_files)
close_log_files(summary_files)
