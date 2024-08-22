import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import json

# Load data


def load_data(filename):
    return np.load(filename)


def main():
    datas = []  # forgive me for me crimes against English
    for file in os.listdir("vectors"):
        if "s_tests" in file:
            data2 = load_data(os.path.join("vectors", file))
        if file.endswith(".npy"):
            datas.append(load_data(os.path.join("vectors", file)))
    data1 = np.vstack(datas)

    train_indices = json.load(open("results/alpaca_7b/lora_saved_instances_1.json", "r"))

    # Combine the datasets for PCA
    combined_data = np.vstack([data1, data2])

    # Perform PCA
    # pca = PCA(n_components=2)
    # transformed_data = pca.fit_transform(combined_data)

    # perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(combined_data)

    # Split the transformed data back
    transformed_data1 = transformed_data[: len(data1)]
    transformed_data2 = transformed_data[len(data1):]

    # Plot
    plt.scatter(transformed_data1[:, 0], transformed_data1[:, 1], color="blue", label="Train grads")
    plt.scatter(transformed_data2[:, 0], transformed_data2[:, 1], color="red", label="Test hvp+grads")
    for index in train_indices:
        plt.scatter(
            transformed_data1[index, 0], transformed_data1[index, 1], color="green", label="Train grads (select)"
        )
    plt.legend()
    plt.title("t-SNE Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("tsne.png")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Visualize high-dimensional vectors from two .npy files using PCA.")
    # parser.add_argument("file1", type=str, help="Path to the first .npy file.")
    # parser.add_argument("file2", type=str, help="Path to the second .npy file.")
    # args = parser.parse_args()

    main()
