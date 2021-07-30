import json
import matplotlib.pyplot as plt
from utils import part2Plots
from utils import visualizeWeights
import numpy as np

model_names = ["mlp1", "mlp2", "cnn3", "cnn4", "cnn5"]

for model_name in model_names:

    file = "PART2RESULTS\\" + model_name + "_results.json"
    with open(file) as json_file:
        data = json.load(json_file)

    train_acc = np.zeros((1, 810))
    val_acc = np.zeros((1, 810))
    train_loss = np.zeros((1, 810))

    for item in list(data.keys()):
        res_dict = data[item]
        train_acc += np.array(res_dict["train_acc_curve"])
        train_loss += res_dict["train_loss_curve"]
        val_acc += res_dict["validation_acc_curve"]

    test_accuracies = [data[item]["test_acc"] for item in data.keys()]
    max_test_acc = max(test_accuracies)
    maximum_test_acc_model = "Trial" + str(test_accuracies.index(max_test_acc) + 1)

    first_layer = data[maximum_test_acc_model]["first_layer"]
    visualizeWeights(
        np.array(first_layer),
        save_dir="PART2RESULTS\\weights",
        filename=model_name + "weights",
    )
    print("SHAPES: ", np.array(first_layer).shape)

    mean_train_acc = (train_acc * 0.1).tolist()[0]
    mean_train_loss = (train_loss * 0.1).tolist()[0]
    mean_val_acc = (val_acc * 0.1).tolist()[0]

    RESULTS_DICT = {}
    RESULTS_DICT["test_acc"] = max_test_acc
    RESULTS_DICT["weights"] = first_layer
    RESULTS_DICT["train_acc_curve"] = mean_train_acc
    RESULTS_DICT["val_acc_curve"] = mean_val_acc
    RESULTS_DICT["loss_curve"] = mean_train_loss
    RESULTS_DICT["name"] = model_name

    ## Save dictionary
    with open("PART2RESULTS/Part2_" + model_name + "_final_results.json", "w") as fp:
        json.dump(RESULTS_DICT, fp)

ALL_DICTS = []
for model_name in model_names:
    file = "PART2RESULTS\\Part2_" + model_name + "_final_results.json"
    with open(file) as json_file:
        data = json.load(json_file)
    ALL_DICTS.append(data)

part2Plots(
    ALL_DICTS, save_dir="PLOTS", filename="Part2_AllPlots"
)


with open(file) as json_file:
    data = json.load(json_file)
