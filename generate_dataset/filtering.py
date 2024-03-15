"""
过滤出含特定角色对话的文件
"""
import os
import pandas as pd
import shutil

def move_csv_files(csv_path,character_name):
    print(f"正在过滤含角色{character_name}的对话记录")
    others_folder = os.path.join(csv_path, 'others')
    character_included_folder = os.path.join(csv_path, character_name)
    os.makedirs(character_included_folder,exist_ok=True)
    os.makedirs(others_folder,exist_ok=True)
    for filename in os.listdir(csv_path):
        if filename.endswith('.csv'): 
            file_path = os.path.join(csv_path, filename)
            try:
                df = pd.read_csv(file_path)
                contains_character = df['Character Name'].str.contains(character_name, case=False, na=False).any()
                if contains_character:
                    shutil.move(file_path, character_included_folder)
                else:
                    shutil.move(file_path, others_folder)
            except Exception as e:
                print(f'Error processing {filename}: {e}')
        else:
            continue

if __name__ == "__main__":
    move_csv_files("dataset/raw_data",'安吉拉')