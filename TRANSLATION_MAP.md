# Chinese to English Filename Translation Map

## Python Scripts

- `01_数据预处理.py` → `01_data_preprocessing.py` (Data Preprocessing)
- `02_性质预测.py` → `02_property_prediction.py` (Property Prediction)
- `03_文件组合.py` → `03_file_merge.py` (File Merge/Combination)

## CSV Files in results/

- `基团替换.csv` → `group_substitution.csv` (Group Substitution)
- `H替换.csv` → `h_substitution.csv` (H Substitution)

## CSV Files in data/

- `转换规则.csv` → `transformation_rules.csv` (Transformation Rules)
- `转换规则_MACCS.csv` → `transformation_rules_maccs.csv` (Transformation Rules MACCS)
- `转换规则_Morgan.csv` → `transformation_rules_morgan.csv` (Transformation Rules Morgan)

## Common Chinese Terms in Code

- `设置超参数` → "Set hyperparameters"
- `所有分子处理完成` → "All molecules processed"
- `已删除` → "Deleted"
- `删除失败` → "Deletion failed"
- `原因` → "Reason"
- `目标分子拆分` → "Target molecule fragmentation"
- `片段化分子` → "Fragment molecules"
- `处理分子碎片的函数` → "Function to process molecular fragments"
- `输入分子无效` → "Invalid input molecule"
- `运行结束` → "Run ended"
- `请检查` → "Please check"
- `该分子无法被有效拆分` → "This molecule cannot be effectively fragmented"
- `无法拆分，跳过` → "Cannot fragment, skipping"
- `规则寻找` → "Rule finding"
- `计算Tanimoto相似性` → "Calculate Tanimoto similarity"
- `生成 MACCS 指纹` → "Generate MACCS fingerprints"
- `规则过滤` → "Rule filtering"
- `第一行第一列为` → "First row, first column is"
- `第一行其余列为分子指纹` → "Remaining columns in first row are molecular fingerprints"
- `第三列为` → "Third column is"
- `第四列到最后一列为` → "Columns 4 to end are"
- `转换为RDKit的分子指纹对象` → "Convert to RDKit molecular fingerprint object"
- `计算相似性` → "Calculate similarity"
- `如果相似性大于阈值，将该行保存` → "If similarity > threshold, save this row"
- `没有找到匹配的规则` → "No matching rules found"
- `完成目标分子规则筛选，生成文件` → "Completed target molecule rule filtering, generated file"
- `文件优化` → "File optimization"
- `仅保留` → "Keep only"
- `去重` → "Deduplicate"
- `检查...列是否存在` → "Check if column exists"
- `文件中没有找到` → "Not found in file"
- `使用 str.split() 方法拆分` → "Split using str.split() method"
- `遍历每一行` → "Iterate through each row"
- `对每一行生成三行` → "Generate three rows for each row"
- `复制当前行` → "Copy current row"
- `替换` → "Replace"
- `将新行添加到列表中` → "Add new row to list"
- `将新行列表转换为新的 DataFrame` → "Convert new row list to new DataFrame"
- `将使用H进行替换的规则单独分开` → "Separate rules that use H for replacement"
- `检查...列是否为空，并提取这些行` → "Check if column is empty and extract those rows"
- `如果没有找到空值行，提示用户` → "If no null rows found, notify user"
- `筛选出每隔3行的数据` → "Filter data every 3 rows"
- `保存筛选后的结果到新的 CSV 文件` → "Save filtered results to new CSV file"
- `改造部位确定` → "Determine modification sites"
- `创建一个字典，将...映射到` → "Create dictionary mapping...to"
- `创建一个集合来记录被使用的键` → "Create set to record used keys"
- `定义一个函数，用于替换` → "Define function to replace"
- `如果...是空值，直接返回` → "If...is null, return directly"
- `确保...是字符串类型` → "Ensure...is string type"
- `记录被使用的键` → "Record used keys"
- `保存结果到文件` → "Save results to file"
- `预测结果文件缺失` → "Prediction results file missing"
- `请确认已安装...依赖并重试` → "Please confirm...dependencies are installed and retry"

## Notebook Chinese Content (figure_code/)

Most are plot labels and comments in Chinese - these can remain as-is for now since they're in the figure code folder which is not part of runtime.
