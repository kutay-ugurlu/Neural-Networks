import matplotlib.pyplot as plt
from utils import part3Plots
import json

model_names = ["mlp1", "mlp2", "cnn3", "cnn4", "cnn5"]
container = []

for model_name in model_names:
    file = "PART3RESULTS\\" + model_name + "_new_results.json"
    with open(file) as json_file:
        data = json.load(json_file)
    container.append(data)

part3Plots(
    container,
    save_dir="D:\\EE4.2\\EE449\\HW1\\PART3RESULTS",
    filename="Part3_results",
)
