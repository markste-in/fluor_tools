import pandas as pd
from rdkit import Chem
from rdkit.Chem import (MolToSmiles, rdMMPA, MACCSkeys, AllChem, rdMolDescriptors, 
                        DataStructs, MolFromSmiles, CombineMols, Draw)
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
from tqdm import tqdm
import csv
import os
import re
import random
import numpy as np
import joblib
from typing import List, Dict, Optional, Tuple
import glob
from PIL import Image, ImageDraw, ImageFont


def process(similarity_value):

    image_folder = './results/molecule_images'
    png_files = glob.glob(os.path.join(image_folder, '*.png'))

    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Deleted PNG file: {file_path}")
        except Exception as e:
            print(f"Failed to delete PNG: {file_path}, reason: {e}")


    image_folder = './results/rules_images'
    png_files = glob.glob(os.path.join(image_folder, '*.png'))

    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Deleted PNG file: {file_path}")
        except Exception as e:
            print(f"Failed to delete PNG: {file_path}, reason: {e}")
            

    results_folder = './results'
    csv_files = glob.glob(os.path.join(results_folder, '*.csv'))

    for file_path in csv_files:
        try:
            os.remove(file_path)
            print(f"Deleted CSV file: {file_path}")
        except Exception as e:
            print(f"Failed to delete CSV: {file_path}, reason: {e}")


    ##############################################  1_Target molecule fragmentation ##############################################
    
    similarity_value = similarity_value

    def fragment_molecules(molecules): # Fragment molecules
        fragments = []
        for mol in tqdm(molecules, desc="Fragmenting molecules", ncols=100):
            if mol is not None:
                fragment = rdMMPA.FragmentMol(mol, minCuts=min_cuts, maxCuts=max_cuts, maxCutBonds=max_cut_bonds, resultsAsMols=asmol)
                fragments.append(fragment)
        return fragments

    def frag(fragments, original_molecules): # Function to process molecular fragments
        mol_nums = 0
        results = []
        for mols, original_mol in zip(fragments, original_molecules):
            mol_nums += 1
            cut_nums = 0
            for cuts in mols:
                cut_nums += 1
                scaffold = cuts[0]
                frag = cuts[1]
                if scaffold is not None:
                    scaffold_smiles = MolToSmiles(scaffold)
                else:
                    scaffold_smiles = None
                if frag is not None:
                    frag_smiles = MolToSmiles(frag)
                else:
                    frag_smiles = None
                if original_mol is not None:
                    original_smiles = MolToSmiles(original_mol)
                else:
                    original_smiles = None
                results.append([original_smiles, cut_nums, scaffold_smiles, frag_smiles])
        return results

    data_target = pd.read_csv('./input/target_m.csv') 

    try:
        first_smiles = data_target['smiles'].iloc[0]
        mol = Chem.MolFromSmiles(first_smiles)
        if mol is not None:
            abs_molecules = [mol]
        else:
            raise RuntimeError(f"Invalid input molecule, run ended, please check SMILES: {first_smiles}")

        cut_nums = [1, 2, 3]
        fragments_df = pd.DataFrame()

        for cut_num in cut_nums:
            min_cuts = cut_num
            max_cuts = cut_num
            max_cut_bonds = 100
            asmol = True

            abs_fragments = fragment_molecules(abs_molecules)
            fragments = abs_fragments
            columns = ['smiles', 'cutnum', 'scaffold', 'fragment']
            results = []

            results = frag(fragments, abs_molecules)
            df = pd.DataFrame(results, columns=columns)

            df['combined'] = df.apply(
                lambda row: f"{row.iloc[2]}.{row.iloc[3]}" if pd.notna(row.iloc[2]) and pd.notna(row.iloc[3]) else (row.iloc[2] if pd.notna(row.iloc[2]) else row.iloc[3]),
                axis=1
            )
            
            fragments_df = pd.concat([fragments_df, df], ignore_index=True)

        # Save results to file
        fragments_df.to_csv('./results/target_fragment.csv', index=False)

        if fragments_df.empty:
            raise RuntimeError(f"This molecule cannot be effectively fragmented: {first_smiles}")

    except Exception as e:
        raise RuntimeError("Cannot fragment, skipping")
        


    ##############################################  2_Rule finding ##############################################

    def tanimoto_similarity(fp1, fp2): # Calculate Tanimoto similarity
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    first_smiles = data_target.loc[0, 'smiles']
    mol = Chem.MolFromSmiles(first_smiles)

    # Generate MACCS fingerprints
    if mol is not None:
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_list = list(fp)
        maccs_df = pd.DataFrame([fp_list], columns=[f'bit_{i}' for i in range(167)])
        target_maccs_df = pd.concat([data_target.loc[[0]], maccs_df], axis=1)

    ########### Rule filtering ###########
    rules_maccs = pd.read_csv('./data/transformation_rules_maccs.csv')

    smiles1 = target_maccs_df.iloc[0, 0]  # First row, first column is smiles
    fingerprint1 = target_maccs_df.iloc[0, 1:].values.astype(int)  # Remaining columns in first row are molecular fingerprints

    mol1 = Chem.MolFromSmiles(smiles1)
    fp1 = DataStructs.CreateFromBitString(''.join(fingerprint1.astype(str)))

    output_rows = []

    for _, row in rules_maccs.iterrows():
        smiles2 = row[2]  # Third column is smiles
        fingerprint2 = row[3:].values.astype(int)  # Columns 4 to end are molecular fingerprints
        
        # Convert to RDKit molecular fingerprint object
        fp2 = DataStructs.CreateFromBitString(''.join(fingerprint2.astype(str)))
        
        # Calculate similarity
        similarity = tanimoto_similarity(fp1, fp2)
        
        # If similarity > threshold, save this row
        if similarity > similarity_value:
            output_rows.append(row)

    target_similary_rules_df = pd.DataFrame(output_rows)
    target_similary_rules_df.to_csv('./results/target_similary_rules.csv', index=False)

    if target_similary_rules_df.empty:
        raise RuntimeError("No matching rules found, run ended")

    print('----------Completed target molecule rule filtering, generated file: target_similary_rules.csv----------')


    ########### File optimization ###########
    element_tran_unique = target_similary_rules_df[['element_tran']] # Keep only element_tran column
    element_tran_unique = element_tran_unique.drop_duplicates() # Deduplicate

    if 'element_tran' not in element_tran_unique.columns: # Check if element_tran column exists
        raise ValueError("'element_tran' column not found in target_similary_rules.csv file, please check file content!")

    element_tran_unique[['node1', 'node2']] = element_tran_unique['element_tran'].str.split(' --->>>--- ', expand=True) # Split element_tran column using str.split() method

    new_rows = []
    for index, row in element_tran_unique.iterrows(): # Iterate through each row
        for i in range(1, 4): # Generate three rows for each row, replacing [*:*] with [*:1], [*:2], [*:3]
            new_row = row.copy()  # Copy current row
            new_row = new_row.replace(r'\[\*:\*\]', f'[*:{i}]', regex=True)  # Replace [*:*] with [*:i]
            new_rows.append(new_row)  # Add new row to list

    # Convert new row list to new DataFrame
    target_rules_df = pd.DataFrame(new_rows)
    target_rules_df.to_csv('./results/target_rules.csv', index=False)


    # Separate rules that use H for replacement
    file_path = './results/target_rules.csv'
    df = pd.read_csv(file_path)

    # Check if node1 column is empty and extract those rows
    empty_node1_rows = df[df['node1'].isna()]

    # If no null rows found, notify user
    if empty_node1_rows.empty:
        print('----------Completed target molecule rule organization, generated file: target_rules.csv----------')
    else:
        # Filter data every 3 rows (index starts from 0, so select rows 0, 3, 6...)
        filtered_rows = empty_node1_rows.iloc[::3]

        # Save filtered results to new CSV file
        output_file = './results/target_rules_replace.csv'
        filtered_rows.to_csv(output_file, index=False)
        print(f'----------Completed target molecule rule organization, generated files: target_rules.csv and target_rules_replace.csv----------')


    ##############################################  4_Determine modification sites ##############################################
    df1 = fragments_df
    file2_path = './results/target_rules.csv'
    df2 = pd.read_csv(file2_path)

    # Create dictionary mapping node1 to node2 from second file
    node1_to_node2 = dict(zip(df2['node1'], df2['node2']))

    # Create set to record used keys
    used_keys = set()

    # Define function to replace node1 part with node2 in combined column
    def replace_node1_with_node2(combined_value):
        if pd.isna(combined_value):
            return combined_value  # If combined is null, return directly
        nodes = str(combined_value).split('.')  # Ensure combined_value is string type
        replaced_nodes = []
        for node in nodes:
            if node in node1_to_node2:
                used_keys.add(node)  # Record used keys
                replaced_nodes.append(str(node1_to_node2[node]))
            else:
                replaced_nodes.append(node)
        return '.'.join(replaced_nodes)  # Recombine as string

    # Check if combined column contains values from node1 column of second file, and replace matched rows
    matched_rows = df1[df1['combined'].apply(lambda x: any(node in str(x).split('.') for node in node1_to_node2.keys()))].copy()

    # If there are matching rows, perform replacement
    if not matched_rows.empty:
        matched_rows['combined'] = matched_rows['combined'].apply(replace_node1_with_node2)

    # If no matching rows, notify user
    if matched_rows.empty:
        raise RuntimeError("No matching rows found, run ended")
    else:
        # Save results to new CSV file
        output_file = './results/new_m_replace.csv'
        matched_rows.to_csv(output_file, index=False)
        
        # Extract and save used key-value pairs
        used_pairs = df2[df2['node1'].isin(used_keys)]
        used_pairs_file = './results/used_mapping_pairs.csv'
        used_pairs.to_csv(used_pairs_file, index=False)
        print(f"--------Saved used key-value pairs to {used_pairs_file}--------")


    ################# new_m_replace.csv file organization #################
    file_path = './results/new_m_replace.csv'
    df = pd.read_csv(file_path)

    # Check if file contains 'combined' column
    if 'combined' not in df.columns:
        raise RuntimeError("File has no 'combined' column")


    # Keep only 'combined' column
    combined_df = df[['combined']]
    df = combined_df


    # Split 'combined' column and force fill to 4 columns to prevent missing columns from insufficient fragments
    split_columns = df['combined'].str.split('\.', expand=True)
    max_cols = 4
    split_columns = split_columns.reindex(columns=range(max_cols), fill_value=np.nan)
    split_columns.columns = [f'element_{i+1}' for i in range(max_cols)]

    # Merge split columns back to original DataFrame
    df = pd.concat([df, split_columns], axis=1)

    # Delete original 'combined' column
    df.drop(columns=['combined'], inplace=True)


    # Define function to check and assign elements to different columns
    def reassign_columns(row):
        # Get values from second, third, and fourth columns of current row
        col2, col3, col4 = row[1], row[2], row[3]
        
        # Ensure each value is string type and handle null values
        col2 = str(col2) if pd.notna(col2) else ''
        col3 = str(col3) if pd.notna(col3) else ''
        col4 = str(col4) if pd.notna(col4) else ''
        
        # Initialize new columns
        new_col2, new_col3, new_col4 = [], [], []
        
        # Check each column
        if "[*:1]" in col2:
            new_col2.append(col2)
        elif "[*:2]" in col2:
            new_col3.append(col2)
        elif "[*:3]" in col2:
            new_col4.append(col2)
        
        if "[*:1]" in col3:
            new_col2.append(col3)
        elif "[*:2]" in col3:
            new_col3.append(col3)
        elif "[*:3]" in col3:
            new_col4.append(col3)
        
        if "[*:1]" in col4:
            new_col2.append(col4)
        elif "[*:2]" in col4:
            new_col3.append(col4)
        elif "[*:3]" in col4:
            new_col4.append(col4)
        
        return ','.join(new_col2), ','.join(new_col3), ','.join(new_col4)

    # Process each row and reassign to new columns
    for i, row in df.iterrows():
        df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3] = reassign_columns(row)

    df.to_csv('./results/new_m_replace.csv', index=False)


    ################# Further organization of new_m_replace.csv file #################
    ################# The following content was added to handle non-circular generation, also applicable to normal cases #################
    input_file = "./results/new_m_replace.csv"
    output_file = "./results/new_m_replace.csv"

    df = pd.read_csv(input_file)

    # Check element count in each row, delete rows with less than 2 elements
    df = df.dropna(thresh=2)
    df.to_csv(output_file, index=False)


    ################# Data supplementation
    input_file = "./results/new_m_replace.csv"
    output_file = "./results/new_m_replace.csv"

    df = pd.read_csv(input_file)

    # Define processing function
    def process_row(row):
        element_1 = row["element_1"]
        element_2 = row["element_2"]
        element_3 = row["element_3"]
        element_4 = row["element_4"]

        # Check markers in element_1
        has_marker_1 = "[*:1]" in element_1
        has_marker_2 = "[*:2]" in element_1
        has_marker_3 = "[*:3]" in element_1

        # Case 1: element_1 has [*:1] and [*:2], but not [*:3]
        if has_marker_1 and has_marker_2 and not has_marker_3:
            if pd.isna(element_2) and pd.isna(element_3):
                # If both element_2 and element_3 are empty, cannot fill, skip
                pass
            elif pd.isna(element_2):
                # If element_2 is empty, copy element_3 to element_2
                row["element_2"] = element_3
            elif pd.isna(element_3):
                # If element_3 is empty, copy element_2 to element_3
                row["element_3"] = element_2

        # Case 2: element_1 has [*:1], [*:2] and [*:3]
        elif has_marker_1 and has_marker_2 and has_marker_3:
            # Count null values in element_2, element_3, element_4
            null_count = sum(pd.isna([element_2, element_3, element_4]))

            if null_count == 2:
                # If two columns are empty, copy the non-empty column to the other two
                non_null_value = next((x for x in [element_2, element_3, element_4] if not pd.isna(x)), None)
                if non_null_value is not None:
                    row["element_2"] = non_null_value
                    row["element_3"] = non_null_value
                    row["element_4"] = non_null_value
            elif null_count == 1:
                # If only one column is empty, randomly select a column to copy to the empty one
                non_null_columns = [col for col in ["element_2", "element_3", "element_4"] if not pd.isna(row[col])]
                if non_null_columns:
                    random_column = random.choice(non_null_columns)
                    for col in ["element_2", "element_3", "element_4"]:
                        if pd.isna(row[col]):
                            row[col] = row[random_column]

        return row

    # Process each row
    df = df.apply(process_row, axis=1)
    df.to_csv(output_file, index=False)

    ### Label organization ###
    file_path = './results/new_m_replace.csv'
    df = pd.read_csv(file_path)

    # Define function to replace [*:1], [*:2], [*:3] in specified columns
    def replace_elements_in_column(column, replacement):
        return column.str.replace(r'\[\*:1\]|\[\*:2\]|\[\*:3\]', replacement, regex=True)

    # Replace elements in second, third, and fourth columns
    df.iloc[:, 1] = replace_elements_in_column(df.iloc[:, 1], '[*:1]')
    df.iloc[:, 2] = replace_elements_in_column(df.iloc[:, 2], '[*:2]')
    df.iloc[:, 3] = replace_elements_in_column(df.iloc[:, 3], '[*:3]')

    output_file = './results/new_m_replace.csv'  # Replace with output file path
    df.to_csv(output_file, index=False)

    print('----------Completed modification site correspondence, generated file: new_m_replace.csv----------')

    ##############################################  5_Generate new molecules ##############################################


    def find_connection_atom(mol, map_num):
        """Find atom index with specified mapping number"""
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == map_num:
                return atom.GetIdx()
        return None

    # Read CSV file
    df = pd.read_csv('./results/new_m_replace.csv')
    unique_molecules = set()

    for i, row in df.iterrows():
        try:
            # Get non-empty elements and store in list
            non_empty_elements = row.dropna().tolist()
            num_elements = len(non_empty_elements)

            if num_elements == 2:
                smiles_part1 = non_empty_elements[0]
                smiles_part2 = non_empty_elements[1]
                mol_part1 = Chem.MolFromSmiles(smiles_part1)
                mol_part2 = Chem.MolFromSmiles(smiles_part2)

                if None in (mol_part1, mol_part2):
                    raise ValueError("SMILES parsing failed")

                atom_to_connect_1 = find_connection_atom(mol_part1, 1)
                atom_to_connect_2 = find_connection_atom(mol_part2, 1)

                if None in (atom_to_connect_1, atom_to_connect_2):
                    raise ValueError("Cannot find all connection points")

                combined_mol = Chem.RWMol(Chem.CombineMols(mol_part1, mol_part2))
                offset = mol_part1.GetNumAtoms()
                combined_mol.AddBond(atom_to_connect_1, atom_to_connect_2 + offset, Chem.BondType.SINGLE)

                # Clear mapping numbers and generate canonical SMILES
                for atom in combined_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                final_mol = combined_mol.GetMol()
                final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                final_smiles = final_smiles.replace('**', '')
                
                # Real-time deduplication
                if final_smiles not in unique_molecules:
                    unique_molecules.add(final_smiles)

            elif num_elements == 3:
                smiles_part1 = non_empty_elements[0]
                smiles_part2 = non_empty_elements[1]
                smiles_part3 = non_empty_elements[2]
                mol_part1 = Chem.MolFromSmiles(smiles_part1)
                mol_part2 = Chem.MolFromSmiles(smiles_part2)
                mol_part3 = Chem.MolFromSmiles(smiles_part3)

                if None in (mol_part1, mol_part2, mol_part3):
                    raise ValueError("SMILES parsing failed")

                atom_to_connect_1 = find_connection_atom(mol_part1, 1)
                atom_to_connect_2 = find_connection_atom(mol_part1, 2)
                atom_to_connect_3 = find_connection_atom(mol_part2, 1)
                atom_to_connect_4 = find_connection_atom(mol_part3, 2)

                if None in (atom_to_connect_1, atom_to_connect_2, atom_to_connect_3, atom_to_connect_4):
                    raise ValueError("Cannot find all connection points")

                combined_mol = Chem.RWMol(Chem.CombineMols(mol_part1, mol_part2))
                combined_mol = Chem.RWMol(Chem.CombineMols(combined_mol, mol_part3))
                offset_2 = mol_part1.GetNumAtoms()
                offset_3 = offset_2 + mol_part2.GetNumAtoms()

                combined_mol.AddBond(atom_to_connect_1, atom_to_connect_3 + offset_2, Chem.BondType.SINGLE)
                combined_mol.AddBond(atom_to_connect_2, atom_to_connect_4 + offset_3, Chem.BondType.SINGLE)

                # Clear mapping numbers and generate canonical SMILES
                for atom in combined_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                final_mol = combined_mol.GetMol()
                final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                final_smiles = final_smiles.replace('**', '')
                
                if final_smiles not in unique_molecules:
                    unique_molecules.add(final_smiles)

            elif num_elements == 4:
                smiles_part1 = non_empty_elements[0]
                smiles_part2 = non_empty_elements[1]
                smiles_part3 = non_empty_elements[2]
                smiles_part4 = non_empty_elements[3]
                mol_part1 = Chem.MolFromSmiles(smiles_part1)
                mol_part2 = Chem.MolFromSmiles(smiles_part2)
                mol_part3 = Chem.MolFromSmiles(smiles_part3)
                mol_part4 = Chem.MolFromSmiles(smiles_part4)

                if None in (mol_part1, mol_part2, mol_part3, mol_part4):
                    raise ValueError("SMILES parsing failed")

                atom_to_connect_1 = find_connection_atom(mol_part1, 1)
                atom_to_connect_2 = find_connection_atom(mol_part1, 2)
                atom_to_connect_3 = find_connection_atom(mol_part1, 3)
                atom_to_connect_4 = find_connection_atom(mol_part2, 1)
                atom_to_connect_5 = find_connection_atom(mol_part3, 2)
                atom_to_connect_6 = find_connection_atom(mol_part4, 3)

                if None in (atom_to_connect_1, atom_to_connect_2, atom_to_connect_3, 
                        atom_to_connect_4, atom_to_connect_5, atom_to_connect_6):
                    raise ValueError("Cannot find all connection points")

                combined_mol = Chem.RWMol(Chem.CombineMols(mol_part1, mol_part2))
                combined_mol = Chem.RWMol(Chem.CombineMols(combined_mol, mol_part3))
                combined_mol = Chem.RWMol(Chem.CombineMols(combined_mol, mol_part4))

                offset_2 = mol_part1.GetNumAtoms()
                offset_3 = offset_2 + mol_part2.GetNumAtoms()
                offset_4 = offset_3 + mol_part3.GetNumAtoms()

                combined_mol.AddBond(atom_to_connect_1, atom_to_connect_4 + offset_2, Chem.BondType.SINGLE)
                combined_mol.AddBond(atom_to_connect_2, atom_to_connect_5 + offset_3, Chem.BondType.SINGLE)
                combined_mol.AddBond(atom_to_connect_3, atom_to_connect_6 + offset_4, Chem.BondType.SINGLE)

                # Clear mapping numbers and generate canonical SMILES
                for atom in combined_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                final_mol = combined_mol.GetMol()
                final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                final_smiles = final_smiles.replace('**', '')
                
                if final_smiles not in unique_molecules:
                    unique_molecules.add(final_smiles)

        except Exception as e:
            raise RuntimeError("Warning")

    df_final = pd.DataFrame(list(unique_molecules), columns=['smiles'])
    df_final = df_final.drop_duplicates()

    output_path = './results/group_substitution.csv'
    df_final.to_csv(output_path, index=False)

    print('----------Completed molecule generation, generated file: group_substitution.csv----------')


    ##############################################  6_Generate H substitution molecules ##############################################
    new_m_finally_H = []
    smiles = data_target['smiles'].iloc[0]
    file_path2 = './results/target_rules_replace.csv'

    if not os.path.exists(file_path2):
        raise RuntimeError("File does not exist, no H atom replacement rules")

    df2 = pd.read_csv(file_path2)
    substitute_smiles_list = df2['node2'].dropna().tolist()

    # Store canonical SMILES for deduplication
    canonical_smiles_set = set()

    for substitute_smiles in substitute_smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        substitute_mol = Chem.MolFromSmiles(substitute_smiles)

        replaceable_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetTotalNumHs() > 0:
                replaceable_atoms.append(atom.GetIdx())

        if len(replaceable_atoms) == 0:
            raise RuntimeError("Warning")

        replace_idx = None
        for atom in substitute_mol.GetAtoms():
            if atom.GetSymbol() == "*":
                replace_idx = atom.GetIdx()
                break

        if replace_idx is None:
            raise ValueError("[*:1] marker location not found")

        combined_mol = Chem.CombineMols(mol, substitute_mol)

        for chosen_atom_idx in replaceable_atoms:
            mol_copy = Chem.RWMol(combined_mol)
            substitute_atom = mol_copy.GetAtomWithIdx(len(mol.GetAtoms()) + replace_idx)
            mol_copy.AddBond(chosen_atom_idx, substitute_atom.GetIdx(), Chem.BondType.SINGLE)
            
            try:
                final_mol = mol_copy.GetMol()
                # Canonical SMILES representation
                final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                final_smiles = final_smiles.replace('[*:1]', '')
                
                # Check if already exists (canonical SMILES)
                if final_smiles not in canonical_smiles_set:
                    canonical_smiles_set.add(final_smiles)
                    new_m_finally_H.append(final_smiles)
            except:
                raise RuntimeError("Warning")

    unique_smiles = list(set(new_m_finally_H))
    data_target = pd.DataFrame(unique_smiles, columns=['smiles'])
    output_file_path = './results/h_substitution.csv'
    data_target.to_csv(output_file_path, index=False)
    print('----------Completed H atom replacement, generated file: h_substitution.csv----------')

    ### File merge ###
    file1 = './results/group_substitution.csv'
    file2 = './results/h_substitution.csv'
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_csv('./results/merged_file.csv', index=False)
    print("Two CSV files successfully merged and saved as 'merged_file.csv'")

    ##############################################  7_New molecule prediction ##############################################

    # 1. Data loading and preprocessing
    df = pd.read_csv('./results/merged_file.csv')
    total_molecules = len(df)
    print(f"Successfully loaded {total_molecules} molecules")

    # 2. SMILES validity check
    print("\nChecking SMILES validity...")
    def is_valid_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    valid_mask = df['smiles'].apply(is_valid_smiles)
    invalid_count = (~valid_mask).sum()

    if invalid_count > 0:
        print(f"Found {invalid_count} invalid SMILES, will be removed")
        df = df[valid_mask].copy()
        print(f"Remaining valid molecules: {len(df)}")
    else:
        print("All SMILES are valid")

    # 3. Generate Morgan fingerprints
    print("\nStarting Morgan fingerprint generation...")
    def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
        mol = Chem.MolFromSmiles(smiles)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(morgan_fp)

    fingerprints = []
    for smiles in tqdm(df['smiles'], desc="Generating fingerprints", total=len(df)):
        fp = get_morgan_fingerprint(smiles)
        fingerprints.append(fp)

    print("\nConverting fingerprint data...")
    fingerprint_df = pd.DataFrame(fingerprints, columns=[f'Bit_{i+1}' for i in range(2048)])
    combined_df = pd.concat([df.reset_index(drop=True), fingerprint_df], axis=1)

    # 4. Model prediction
    print("\nLoading model and making predictions...")
    pred_path = './results/merged_file_pred.csv'
    try:
        stacking_model = joblib.load("./data/stacking_model_full.pkl")
        X_new = combined_df.iloc[:, 1:].values
        y_new_prob = stacking_model.predict_proba(X_new)[:, 1]
        y_new_pred = stacking_model.predict(X_new)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'smiles': combined_df['smiles'],
            'predicted_prob': y_new_prob,
            'predicted_label': y_new_pred
        }).sort_values(by='predicted_prob', ascending=False)
        
        label_counts = result_df['predicted_label'].value_counts()
        print("\nPrediction results statistics:")
        print(f"Label 0 count: {label_counts.get(0, 0)}")
        print(f"Label 1 count: {label_counts.get(1, 0)}")
        print(f"Predicted probability range: {result_df['predicted_prob'].min():.3f} - {result_df['predicted_prob'].max():.3f}")
        
        result_df.to_csv(pred_path, index=False)

    except Exception as e:
        print(f"\nModel loading or prediction error: {str(e)}")
        pred_path = None

    ################################################################### Display used rules

    # Read two CSV files
    file1_path = './results/used_mapping_pairs.csv'
    file2_path = './results/target_rules_replace.csv'

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge two DataFrames (vertical stack)
    combined_df = pd.concat([df1, df2], ignore_index=True)

    df = combined_df

    # Define function to replace [*:1], [*:2], ... with [*]
    def normalize_smiles(smiles):
        if pd.isna(smiles):
            return smiles
        return re.sub(r'\[\*:\d+\]', '[*:1]', smiles)

    # Normalize element_tran, node1, node2 columns
    df['element_tran'] = df['element_tran'].apply(normalize_smiles)
    df['node1'] = df['node1'].apply(normalize_smiles)
    df['node2'] = df['node2'].apply(normalize_smiles)

    # Deduplicate (based on all columns)
    df_unique = df.drop_duplicates()

    # Save deduplicated results
    output_path1 = './results/transform_rules.csv'
    df_unique.to_csv(output_path1, index=False)

    ################################################################ Synthesizability analysis
    # Lower SA Score indicates easier synthesis (typical range 1-10, <3 indicates easy synthesis)

    def calculate_sa_scores(input_csv, output_csv, smiles_column=0):
        with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Read header row and add new column
            headers = next(reader)
            headers.append('SA_Score')
            writer.writerow(headers)
                                                                                                                                                                                    
            for row in reader:
                smiles = row[smiles_column]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        sa_score = sascorer.calculateScore(mol)
                        row.append(f"{sa_score:.2f}")  # Keep two decimal places
                    else:
                        row.append('Invalid SMILES')
                except:
                    row.append('Calculation Error')
                
                writer.writerow(row)

    # Usage example - replace with your actual file path
    if pred_path and os.path.exists(pred_path):
        input_file = pred_path
        output_file = './results/new_molecules.csv'
        calculate_sa_scores(input_file, output_file)
        print("Completed synthesizability analysis")
    else:
        raise RuntimeError("Prediction results file missing; please confirm lightgbm dependencies are installed and retry.")