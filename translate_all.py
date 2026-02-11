#!/Users/pandora/miniforge3/envs/fluor/bin/python
# -*- coding: utf-8 -*-
"""
Comprehensive translation script for fluor_tools
Translates ALL Chinese text and renames files
"""

import os
import shutil
import re

BASE_DIR = "/Users/pandora/Projects/fluor_tools-main"

# File renames mapping (Chinese -> English)
FILE_RENAMES = {
    'NIRFluor-opt and web/predict/打包.ipynb': 'NIRFluor-opt and web/predict/packaging.ipynb',
    'NIRFluor-opt and web/predict/.ipynb_checkpoints/打包-checkpoint.ipynb': 'NIRFluor-opt and web/predict/.ipynb_checkpoints/packaging-checkpoint.ipynb',
    'NIRFluor-opt and web/results/基团替换.csv': 'NIRFluor-opt and web/results/group_replacement.csv',
    'NIRFluor-opt and web/results/H替换.csv': 'NIRFluor-opt and web/results/H_replacement.csv',
    'figure_code/Figure_code_for_Fluor-pred/data/02_消融数据.csv': 'figure_code/Figure_code_for_Fluor-pred/data/02_ablation_data.csv'
}

# Content translations (Chinese -> English)
CONTENT_TRANSLATIONS = {
    # Filenames in code references
    '打包.ipynb': 'packaging.ipynb',
    '基团替换.csv': 'group_replacement.csv',
    'H替换.csv': 'H_replacement.csv',
    '02_消融数据.csv': '02_ablation_data.csv',
    
    # Processing messages and print statements
    '已删除 PNG 文件:': 'Deleted PNG file:',
    '删除 PNG 失败:': 'Failed to delete PNG:',
    '原因:': 'Reason:',
    '已删除 CSV 文件:': 'Deleted CSV file:',
    '删除 CSV 失败:': 'Failed to delete CSV:',
    '----------完成目标分子规则筛选，生成文件：target_similary_rules.csv----------': '---------- Target molecule rule screening completed, file generated: target_similary_rules.csv ----------',
    '----------完成目标分子规则整理，生成文件：target_rules.csv 和 target_rules_replace.csv----------': '---------- Target molecule rule organization completed, files generated: target_rules.csv and target_rules_replace.csv ----------',
    '----------完成目标分子规则整理，生成文件：target_rules.csv----------': '---------- Target molecule rule organization completed, file generated: target_rules.csv ----------',
    '--------已保存被使用的键值对到': '-------- Saved used key-value pairs to',
    '警告：数据框的列数不是 4，而是': 'Warning: DataFrame column count is not 4, but',
    '列，请检查数据。': 'columns, please check data.',
    '----------完成改造部位对应，生成文件：new_m_replace.csv----------': '---------- Modification site mapping completed, file generated: new_m_replace.csv ----------',
    '----------完成分子生成，生成文件：基团替换.csv----------': '---------- Molecule generation completed, file generated: group_replacement.csv ----------',
    '----------完成H原子替换，生成文件：H替换.csv----------': '---------- H atom replacement completed, file generated: H_replacement.csv ----------',
    '两个CSV文件已成功合并并保存为 \'merged_file.csv\'': 'Two CSV files successfully merged and saved as \'merged_file.csv\'',
    '成功加载': 'Successfully loaded',
    '个分子': 'molecules',
    '正在检查SMILES有效性...': 'Checking SMILES validity...',
    '发现': 'Found',
    '个无效SMILES，将被移除': 'invalid SMILES, will be removed',
    '剩余有效分子数:': 'Remaining valid molecules:',
    '所有SMILES都是有效的': 'All SMILES are valid',
    '开始生成Morgan指纹...': 'Starting Morgan fingerprint generation...',
    '正在转换指纹数据...': 'Converting fingerprint data...',
    '正在加载模型并进行预测...': 'Loading model and performing prediction...',
    '模型加载或预测出错:': 'Model loading or prediction error:',
    '预测结果统计：': 'Prediction results summary:',
    '标签为 0 的数量:': 'Count of label 0:',
    '标签为 1 的数量:': 'Count of label 1:',
    '预测概率范围:': 'Predicted probability range:',
    '完成可合成性分析': 'Synthesis feasibility analysis completed',
    '所有分子处理完成！': 'All molecules processed!',
    '预测结果文件缺失；请确认已安装 lightgbm 依赖并重试。': 'Prediction result file missing; please confirm lightgbm dependency is installed and retry.',
    
    # Comments in code
    '# 指纹部分一：smiles + extra，使用 CNN-attention 提取': '# Fingerprint part one: smiles + extra, using CNN-attention extraction',
    '# 指纹部分二：solvent，使用全连接提取': '# Fingerprint part two: solvent, using fully connected extraction',
    '# === 分离 fingerprints 三部分 ===': '# === Separate fingerprints into three parts ===',
    '# 假设 fingerprints.shape = [B, S + M + E]（solvent, smiles, extra）': '# Assume fingerprints.shape = [B, S + M + E] (solvent, smiles, extra)',
    '# 你可以根据各自维度切分': '# You can split according to their respective dimensions',
    '# 分别提取特征': '# Extract features separately',
    '# 拼接三部分特征': '# Concatenate three part features',
    '# === 读取 target 数据 ===': '# === Read target data ===',
    '# === 数值特征标准化 ===': '# === Numeric feature standardization ===',
    '# 加载预训练的 scaler_num': '# Load pretrained scaler_num',
    '# === 拼接最终指纹 ===': '# === Concatenate final fingerprints ===',
    '# === 标签标准化（只为保持接口一致，其实预测时不需操作标签） ===': '# === Label standardization (only for interface consistency, not needed for prediction) ===',
    '# 注意：仅为构造 dataset，不影响预测结果': '# Note: only for dataset construction, does not affect prediction results',
    '# === 构造 dataset 与 dataloader ===': '# === Construct dataset and dataloader ===',
    '# 初始化模型': '# Initialize model',
    '# === 从 train_data / valid_data 中提取额外特征（列索引 8:152）===': '# === Extract extra features from train_data / valid_data (column indices 8:152) ===',
    '# === 数值部分（8列）归一化 ===': '# === Normalize numeric part (8 columns) ===',
    '# 拆分：前 8 列为数值特征，后面为补充指纹': '# Split: first 8 columns are numeric features, rest are supplementary fingerprints',
    '# tensor 后部分': '# tensor latter part',
    '# 拟合并归一化前8列': '# Fit and normalize first 8 columns',
    '# 转换回 tensor 并拼接': '# Convert back to tensor and concatenate',
    '# === 拼接最终特征：solvent + smiles + extra ===': '# === Concatenate final features: solvent + smiles + extra ===',
    '# 分开声明两个 fp 输入维度': '# Separate declarations for two fp input dimensions',
    '# 图神经网络部分': '# Graph neural network part',
    '# 加载保存的模型参数': '# Load saved model parameters',
    '# 模型预测': '# Model prediction',
    '# 保存预测结果': '# Save prediction results',
    '# 将预测结果保存到 ./result 文件': '# Save prediction results to ./result file',
    '# 最终拼接后预测（3 * graph_feat_size）': '# Final concatenation for prediction (3 * graph_feat_size)',
    '# 预测': '# Prediction',
    '# 包含权重': '# Contains weights',
    '# 不含权重': '# No weights',
    '# 预测完成后，反向转换标准化的预测结果': '# After prediction completes, inverse transform the standardized prediction results',
    
    # Data preprocessing comments
    '# 1.1 对应溶剂序号': '# 1.1 Map solvent numbers',
    '# 文件路径': '# File paths',
    '# 原始数据文件': '# Original data file',
    '# 溶剂 与 solvent_num 的映射表': '# Solvent to solvent_num mapping table',
    '# 替换后的输出文件': '# Output file after replacement',
    '# 读取数据': '# Read data',
    '# 创建 solvent -> solvent_num 映射字典': '# Create solvent -> solvent_num mapping dictionary',
    '# 替换原列中的 solvent_num': '# Replace solvent_num in original column',
    '# 保存为新的文件': '# Save as new file',
    '# 1.2 生成分子性质数据': '# 1.2 Generate molecular property data',
    '# 读取 CSV 文件': '# Read CSV file',
    '# 替换为你的文件路径': '# Replace with your file path',
    '# 初始化存储计算结果的列表': '# Initialize lists to store calculation results',
    '# 计算双键数量的函数': '# Function to calculate double bond count',
    '# 处理无效的 SMILES': '# Handle invalid SMILES',
    '# 遍历 SMILES 计算各种性质': '# Iterate through SMILES to calculate various properties',
    '# 计算分子量、logP、芳香环数量、TPSA、Gasteiger部分电荷': '# Calculate molecular weight, logP, aromatic ring count, TPSA, Gasteiger partial charge',
    '# 分子量': '# Molecular weight',
    '# 芳香环数量': '# Aromatic ring count',
    '# 近似极化率': '# Approximate polarizability',
    '# 计算 Gasteiger 部分电荷': '# Calculate Gasteiger partial charge',
    '# 计算双键数量': '# Calculate double bond count',
    '# 获取环的信息并计算环的数量': '# Get ring information and calculate ring count',
    '# 处理无效 SMILES': '# Handle invalid SMILES',
    '# 将计算结果添加到对应的列表': '# Add calculation results to corresponding lists',
    '# 将计算结果添加到 DataFrame': '# Add calculation results to DataFrame',
    '# 保存到新的 CSV 文件': '# Save to new CSV file',
    '完成分子性质预测': 'Molecular property prediction completed',
    '# 1.3 对目标分子进行骨架定义': '# 1.3 Define scaffold for target molecules',
    
    # Solvent mapping messages
    'solvent_num 替换完成，结果已保存为：': 'solvent_num replacement completed, results saved to:',
    
    # General translations
    '设置超参数': 'Set hyperparameters',
    '正在运行': 'Running',
    '第一行': 'First row',
    '已替换为：': 'replaced with:',
    '并保存至：': 'and saved to:',
    '输入的溶剂名称': 'Input solvent name',
    '未在映射表中找到': 'not found in mapping table',
    'CSV 文件中未找到': 'Not found in CSV file',
    '或': 'or',
    '列': 'column',
    
    # Error messages
    '无法拆分，跳过': 'Cannot fragment, skipping',
    '该分子无法被有效拆分：': 'This molecule cannot be effectively fragmented:',
    '没有找到匹配的规则，运行结束': 'No matching rules found, execution ended',
    '没有找到匹配的行，运行结束': 'No matching rows found, execution ended',
    '文件无\'combined\' 列': 'File does not have \'combined\' column',
    '警告': 'Warning',
    
    # Notebook specific
    '数据预处理': 'Data Preprocessing',
    '性质预测': 'Property Prediction',
    '文件组合': 'File Merge',
}

def rename_files():
    """Rename files with Chinese characters"""
    print("=== Renaming files ===")
    os.chdir(BASE_DIR)
    
    for old_path, new_path in FILE_RENAMES.items():
        old_full = os.path.join(BASE_DIR, old_path)
        new_full = os.path.join(BASE_DIR, new_path)
        
        if os.path.exists(old_full):
            # Ensure target directory exists
            os.makedirs(os.path.dirname(new_full), exist_ok=True)
            shutil.move(old_full, new_full)
            print(f"Renamed: {old_path} -> {new_path}")
        else:
            print(f"File not found (may already be renamed): {old_path}")

def translate_content_in_file(filepath):
    """Translate Chinese content in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all translations
        for chinese, english in CONTENT_TRANSLATIONS.items():
            content = content.replace(chinese, english)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def translate_all_files():
    """Translate content in all relevant files"""
    print("\n=== Translating file contents ===")
    os.chdir(BASE_DIR)
    
    # Files to translate (after renaming)
    files_to_translate = [
        'Fluor-RLAT/01_data_preprocessing.py',
        'Fluor-RLAT/02_property_prediction.py',
        'Fluor-RLAT/03_file_merge.py',
        'Fluor-RLAT/run.py',
        'NIRFluor-opt and web/predict/01_data_preprocessing.py',
        'NIRFluor-opt and web/predict/02_property_prediction.py',
        'NIRFluor-opt and web/predict/03_file_merge.py',
        'NIRFluor-opt and web/predict/run.py',
        'NIRFluor-opt and web/processing.py',
        'NIRFluor-opt and web/run.py',
        'NIRFluor-opt and web/app.py',
        'NIRFluor-opt and web/predict/packaging.ipynb',
        'figure_code/Figure_code_for_Fluor-opt/Fluor-opt_figures_code.ipynb',
        'figure_code/Figure_code_for_Fluor-pred/Fluor-pred_figures_code.ipynb',
    ]
    
    translated_count = 0
    for filepath in files_to_translate:
        full_path = os.path.join(BASE_DIR, filepath)
        if os.path.exists(full_path):
            if translate_content_in_file(full_path):
                print(f"Translated: {filepath}")
                translated_count += 1
            else:
                print(f"No changes: {filepath}")
        else:
            print(f"File not found: {filepath}")
    
    print(f"\nTranslated {translated_count} files")

def main():
    print("Starting comprehensive translation...")
    print(f"Base directory: {BASE_DIR}\n")
    
    # Step 1: Rename files
    rename_files()
    
    # Step 2: Translate content
    translate_all_files()
    
    print("\n✅ Translation completed!")

if __name__ == '__main__':
    main()
