from utils import part4Plots
import json
import matplotlib.pyplot as plt

RESULTS_DICT = {}
model_name = "mlp2"

# For remaining parts
"""
file = "PART4RESULTS\\" + model_name + "_part_4_6_history.json"
with open(file) as json_file:
    data = json.load(json_file)

RESULTS_DICT["name"] = model_name
RESULTS_DICT["val_acc_curve"] = data["val_sparse_categorical_accuracy"]
"""

## For Part 3.1
LRs = [0.1, 0.01, 0.001]
for LR in LRs:
    LR = str(LR)
    file = "PART4RESULTS\\" + model_name + "LR" + LR + "_history.json"
    with open(file) as json_file:
        data = json.load(json_file)
    RESULTS_DICT["loss_curve_" + LR[2:]] = data["loss"]
    RESULTS_DICT["val_acc_curve_" + LR[2:]] = data["val_sparse_categorical_accuracy"]

## Save dictionary
with open(model_name + "LR_final_results.json", "w") as fp:
    json.dump(RESULTS_DICT, fp)

plt.plot(data["val_sparse_categorical_accuracy"])
plt.title("Validation Accuracy Curve for 2nd training experiment")
plt.legend("mlp2")
plt.show()

part4Plots(
    RESULTS_DICT,
    save_dir="D:\\EE4.2\\EE449\\HW1\\PART4RESULTS",
    filename="Part4Results_4_3",
)
