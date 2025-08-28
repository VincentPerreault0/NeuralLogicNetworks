from NLN_Model import NLN

# THIS EXAMPLE REQUIRES THE 5-FOLD CROSS-VALIDATION EXAMPLE TO BE RUN FIRST
input_filenames = [f"results/tic-tac-toe_{fold_idx}" for fold_idx in range(5)]
output_filename = "results/tic-tac-toe_merged"


models = [NLN.load(input_filename) for input_filename in input_filenames]

merged_model = NLN.merge_models(models, output_filename)

merged_model.show()

## To retrieve model, use
# model = NLN.load(output_filename)
