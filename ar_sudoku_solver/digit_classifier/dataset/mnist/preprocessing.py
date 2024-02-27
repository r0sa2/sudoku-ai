import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("mnist.csv")

    for column in df.columns:
        if column.startswith("pixel"):
            df[column] = df[column].apply(lambda x: 255 - x)

    df.to_csv("mnist_modified.csv", index=False)
