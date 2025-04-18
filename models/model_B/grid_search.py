
"""
The goal here is to save the best parameters for each dataset type.
1. create the train and evaluate function 
2. run the shit in the supercomputer
3. save the best parameters in a json file 
"""
datasets = ["imbalanced", "oversample_70_30", "downsample_70_30"]

results = {}

for dataset_type in datasets:
    config["data"]["sampling_type"] = dataset_type  # or however you switch datasets
    best_score = -float("inf")
    best_params = {}

    for d_model in [64, 128, 256]:
        for num_heads in [2, 4, 8]:
            if d_model % num_heads != 0:
                continue
            for num_layers in [1, 2, 3]:
                for drop_out in [0.1, 0.2, 0.3]:
                    score = train_and_evaluate(
                        config, d_model, num_heads, num_layers, drop_out)
                    print(
                        f"[{dataset_type}] d_model={d_model}, heads={num_heads}, layers={num_layers}, drop={drop_out} -> val_score={score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "d_model": d_model,
                            "num_heads": num_heads,
                            "num_layers": num_layers,
                            "drop_out": drop_out
                        }

    results[dataset_type] = {"score": best_score, "params": best_params}

print("Final Best Results:", results)
