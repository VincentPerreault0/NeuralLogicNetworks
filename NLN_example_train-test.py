from NLN_Dataset import NLNTabularDataset
from NLN_Model import NLN

# IN ORDER FOR THIS METHOD TO BE INTERPRETABLE, THE FIRST LINE OF THE DATASETS SHOULD ALWAYS BE THE COLUMN LABELS.
training_dataset_filename = "datasets/tic-tac-toe_train.csv"
test_dataset_filename = "datasets/tic-tac-toe_test.csv"
output_filename = "results/tic-tac-toe"


train_dataset = NLNTabularDataset(training_dataset_filename)
test_dataset = NLNTabularDataset(test_dataset_filename)

model = NLN(output_filename, train_dataset, test_dataset=test_dataset)

model = model.learn()

model.show()

## To retrieve model, use
# model = NLN.load(output_filename)
