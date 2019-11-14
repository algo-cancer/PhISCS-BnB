import pandas as pd
import re
import time
import os

def main():
    script_name = os.path.basename(__file__).split(".")[0]
    input_file = "subsam_incom_result.txt"
    output_folder_path = "/home/esadeqia/PhISCS_BnB/reports/Erfan"
    pattern = re.compile("\{.*\}")
    with open(input_file, "r") as file:
        text = file.read()
    df = pd.DataFrame(columns=["hash", "n", "m", "k", "t", "n_flips", "method", "runtime"])
    for m in pattern.findall(text):
        row = eval(m)
        df = df.append(row, ignore_index=True)
    now_time = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
    csv_file_name = f"{script_name}_{input_file}_{now_time}.csv"
    csv_path = os.path.join(output_folder_path, csv_file_name)
    df.to_csv(csv_path)
    print(f"CSV file stored at {csv_path}")

if __name__ == '__main__':
    main()
