import pandas as pd
import os
# Input file paths
file1 = './result/target_predictions_abs.csv'
file2 = './result/target_predictions_em.csv'
file3 = './result/target_predictions_plqy.csv'
file4 = './result/target_predictions_k.csv'

# Read each file (single column)
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)

# Merge all columns
combined_df = pd.concat([df1, df2, df3, df4], axis=1)

# Save as new file
combined_df.to_csv('./result/target_predictions.csv', index=False)

print("✅ Merge completed, saved as target_predictions.csv")

def delete_all_bin_files(folder_path):
    deleted_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"✅ Deleted: {file_path}")
                except Exception as e:
                    print(f"❌ Deletion failed: {file_path}, reason: {e}")
    if not deleted_files:
        print("📂 No .bin files found")
    else:
        print(f"🧹 Total deleted {len(deleted_files)} .bin files")

# Execute directly
delete_all_bin_files('./')  # Replace with actual path
