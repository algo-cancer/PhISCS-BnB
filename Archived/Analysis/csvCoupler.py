from Utils.const import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", help=f"csvFileName in {output_folder_path}", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    if "/" in args.filename:
        fileAddress = args.filename
        fileName = (args.filename.split("/")[-1]).split(".")[-1]
    else:
        fileAddress = os.path.join(output_folder_path, args.filename)
        fileName = args.filename
    print(f"Processing {fileAddress}")
    original_df = pd.read_csv(fileAddress)
    new_df = pd.merge(original_df, original_df, on=["hash", "n", "m"], how="inner")
    new_df.rename(columns={"hash": "_hash", "n": "_n", "m": "_m"}, inplace=True)
    cols = sorted(new_df.columns)
    cols.remove("Unnamed: 0_x")
    cols.remove("Unnamed: 0_y")
    new_df = new_df[cols]

    print(new_df)

    nowTime = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
    csvFileName = f"Pair_{fileName}_{new_df.shape}_{nowTime}.csv"
    csvPath = os.path.join(output_folder_path, csvFileName)
    new_df.to_csv(csvPath)
    print(f"CSV file stored at {csvPath}")
