from NLN_Dataset import NLNTabularDataset
from NLN_Model import NLN

# IN ORDER FOR THIS METHOD TO BE INTERPRETABLE, THE FIRST LINE OF THE DATASET SHOULD ALWAYS BE THE COLUMN LABELS.
# training_dataset_filename = "datasets/mammalian_unshuffled.csv"
# training_dataset_filename = "datasets/adult.csv"
training_dataset_filename = "datasets/tic-tac-toe.csv"
# training_dataset_filename = "datasets/tictactoe_OLD.csv"
# training_dataset_filename = "datasets/monk2.csv"
# training_dataset_filename = "datasets/wine.csv"
output_filename = "results/tic-tac-toe"


train_dataset = NLNTabularDataset(training_dataset_filename)

model = NLN(output_filename, train_dataset)

# model = NLN.load(output_filename)

# model.do_train = False
# model.do_quantize = False

model.learn()

model.show()

## To retrieve model, use
# model = NLN.load(output_filename)
