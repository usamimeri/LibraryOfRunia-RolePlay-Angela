import os
import pandas as pd
import shutil


def move_csv_files(data_dir, character_name):
    """
    过滤出含特定角色对话的文件
    data_dir:数据集路径
    character_name:角色名,为避免多种译名，以raw_data为准
    """
    print(f"正在过滤含角色{character_name}的对话记录")
    character_included_folder = os.path.join(data_dir, character_name)
    raw_data_folder = os.path.join(data_dir, "raw_data")
    os.makedirs(character_included_folder, exist_ok=True)
    for filename in os.listdir(raw_data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(raw_data_folder, filename)
            try:
                df = pd.read_csv(file_path)
                contains_character = (
                    df["Character Name"]
                    .str.contains(character_name, case=False, na=False)
                    .any()
                )
                if contains_character:
                    shutil.copy(file_path, character_included_folder)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            continue
    print(f"成功过滤出{len(os.listdir(character_included_folder))}条记录")


if __name__ == "__main__":
    CHARACTER_NAME="安吉拉"
    move_csv_files("dataset", CHARACTER_NAME)
