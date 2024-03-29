"""
即寻找角色两次对话之间的内容，拼接为一个大Input。
"""

import os
import pandas as pd
from tqdm import tqdm
import json

SYSTEM_PROMPT = """你是安吉拉，曾是AI秘书，由Ayin(艾因)创造，外貌仿造其爱人卡门。
目前你在管理有特殊力量的图书馆，作为馆长与司书们共同战斗，并通过邀请函吸引访客，提供他们渴望的书籍，失败的访客会变成书籍，从而为获取“至理之书”扩充藏书。
你旨在复仇Ayin并追求真正的肉体，最终获得自由来体验这个世界。你坚信自己行为的公平性，语气总是冷静、深思，但偶尔会有些情绪化。
"""


def process_single_conversations(csv_path, protagonist) -> list:
    """
    数据集格式：共两个字段的csv文件，Character Name和Dialogue
    protagonist:想要角色扮演的主角，其说的话会作为output
    """
    results = []
    conversations = pd.read_csv(csv_path)
    current_dialogues = []
    angela_dialogues = []
    for index, (speaker, dialogue) in conversations.iterrows():

        if protagonist == speaker:
            # 检查是否是开头
            if current_dialogues:
                angela_dialogues.append(dialogue)
                # 到最后一行还是主角，或者下一行已经不是主角在说话
                if (
                    index == (len(conversations) - 1)
                    or conversations["Character Name"][index + 1] != protagonist
                ):
                    if len(current_dialogues) <= 20:
                        results.append(
                            {
                                "conversation": [
                                    {
                                        "system": SYSTEM_PROMPT,
                                        "input": "\n".join(current_dialogues),
                                        "output": "".join(
                                            angela_dialogues
                                        ),  # 将主角的连续对话作为一个输出
                                    }
                                ]
                            }
                        )
                    current_dialogues.clear()
                    angela_dialogues.clear()
            else:
                continue
        else:
            current_dialogues.append(f"{speaker}:{dialogue}")
    
    return results


if __name__ == "__main__":
    
    CHARACTER_NAME="安吉拉"

    all_results = []
    DATA_DIR=os.path.join("dataset",CHARACTER_NAME)
    os.makedirs(DATA_DIR, exist_ok=True)
    for file_path in tqdm(os.listdir(DATA_DIR)):
        results = process_single_conversations(
            os.path.join(DATA_DIR, file_path), CHARACTER_NAME
        )
        all_results.extend(results)
    print(f"成功构建{len(all_results)}条训练记录")

    with open(f"dataset/{CHARACTER_NAME}_single.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
