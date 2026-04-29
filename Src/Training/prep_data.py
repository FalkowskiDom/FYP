import random
from pathlib import Path

INPUT_FILE = Path("C:\\Users\\domin\\OneDrive - South East Technological University\\Sem 2\\FYP\\Project\\Data\\HDFS\\HDFS.log")
OUTPUT_DIR = Path("C:\\Users\\domin\\OneDrive - South East Technological University\\Sem 2\\FYP\\Project\\Data\\HDFS\\processed")

TOTAL_LINES_TO_SAMPLE = 100000  
TRAIN_SPLIT = 0.9              
SEED = 42


def main():
    # Sets the random seed so the sample is repeatable.
    random.seed(SEED)

    # Checks that the input log file exists.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    # Creates the output folder if it does not exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading and sampling logs...")

    sampled_lines = []

    # Samples lines from the large log file without loading everything at once.
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < TOTAL_LINES_TO_SAMPLE:
                sampled_lines.append(line.strip())
            else:
                j = random.randint(0, i)
                if j < TOTAL_LINES_TO_SAMPLE:
                    sampled_lines[j] = line.strip()

    print(f"Sampled {len(sampled_lines)} lines")

    # Shuffles the sampled logs before splitting.
    random.shuffle(sampled_lines)

    # Calculates where to split the training and test data.
    split_index = int(len(sampled_lines) * TRAIN_SPLIT)

    # Splits the sampled logs into training and test sets.
    train_lines = sampled_lines[:split_index]
    test_lines = sampled_lines[split_index:]

    train_file = OUTPUT_DIR / "hdfs_train.log"
    test_file = OUTPUT_DIR / "hdfs_test.log"

    print("Saving files...")

    # Saves the training log file.
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    # Saves the test log file.
    with open(test_file, "w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line + "\n")

    print(f"Train file: {train_file} ({len(train_lines)} lines)")
    print(f"Test file: {test_file} ({len(test_lines)} lines)")
    print("Done.")


if __name__ == "__main__":
    # Runs the script.
    main()