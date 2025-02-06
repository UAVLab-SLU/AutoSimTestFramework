import argparse
import pandas as pd
from pyulog import ULog

def ulg_to_csv(ulg_file, output_csv):
    # Load ULog file
    ulog = ULog(ulg_file)

    # Collect data from all messages
    all_dataframes = []
    
    for msg in ulog.data_list:
        msg_name = msg.name

        # Convert message to a DataFrame
        df = pd.DataFrame(msg.data)
        
        if not df.empty:
            df["message_name"] = msg_name  # Add a column to identify the message type
            all_dataframes.append(df)

    # Merge all data into a single DataFrame
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"CSV file saved: {output_csv}")
    else:
        print("No valid data found in the ULog file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ULog file to CSV")
    parser.add_argument("ulg_file", help="Path to the ULog file")
    parser.add_argument("csv_file", help="Path to save the output CSV file")
    args = parser.parse_args()

    ulg_to_csv(args.ulg_file, args.csv_file)
