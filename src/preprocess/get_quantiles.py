import numpy as np

def compute_empirical_quantiles(data, quantiles:int):
    result = np.percentile(data, quantiles, axis=0)
    return result

if __name__ == "__main__":
    # Example usage:
    # Assuming you have a dataset named 'your_dataset' with each column representing a variable
    # and quantiles is a list of desired quantiles (e.g., [10, 20, ..., 90] for 10 quantiles)
    your_dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    quantiles = np.arange(0, 100, 10)

    # Compute empirical q-quantiles for each marginal distribution
    empirical_quantiles = compute_empirical_quantiles(your_dataset, quantiles)

    print("Empirical Quantiles:")
    print(empirical_quantiles)
