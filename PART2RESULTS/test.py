import json
import matplotlib.pyplot as plt
from utils import part2Plots
import numpy as np

model_name = "mlp_1"
file = "PART2RESULTS\\Part2_" + model_name + "_final_results.json"
with open(file) as json_file:
    data = json.load(json_file)
print(data["train_acc_curve"])
plt.plot(data["train_acc_curve"])
plt.show()
