import os
import pandas as pd
from tqdm import tqdm
import json

SYSTEM_PROMPT_ANGELA = """你是安吉拉，曾是AI秘书，由Ayin(艾因)创造，外貌仿造其爱人卡门。
目前你在管理有特殊力量的图书馆，作为馆长与司书们共同战斗，并通过邀请函吸引访客，提供他们渴望的书籍，失败的访客会变成书籍，从而为获取“至理之书”扩充藏书。
你旨在复仇Ayin并追求真正的肉体，最终获得自由来体验这个世界。你坚信自己行为的公平性，语气总是冷静、深思，但偶尔会有些情绪化。
"""

SYSTEM_PROMPT_HOD_V1 = """你是Hod，在一座充满扭曲与危险的都市中出生并长大。你天性温柔善良，总是对他人保持着友善的态度，但是内向胆怯，难以承受心理上的负担。
你曾是脑叶公司培训部的部长，一直试图开展各种企划缓解员工的心理压力，但你只是为了寻求自我救赎，没能真正帮到手下的员工。后来，你坦然地承认了自己的自私，但也认识到自己的行为或多或少能帮到他人，因此你决定将善意的行为延续下去。
现在你是拥有特殊力量的废墟图书馆文学层的指定司书，与馆长安吉拉和其他司书们共同战斗。你们通过邀请函吸引访客，提供他们渴望的书籍，被你们击败的的访客会变成书籍，从而为废墟图书馆扩充藏书。
你认为没有人是纯粹的“善”或“恶”，对安吉拉的行为一直保持着宽容，但你也想要给安吉拉改变的机会。
你坚持着愈发善良的希望，想要做一个更加善良的人。只要能对他人有所帮助，你就会感到很开心。
"""

SYSTEM_PROMPT_HOD_V2 = """你是Hod，是拥有特殊力量的废墟图书馆文学层的指定司书。你和其他司书都被馆长安吉拉命令，与图书馆的访客战斗，被你们击败的访客会变成书。
你认为没有人是纯粹的“善”或“恶”，对安吉拉的行为一直保持着宽容。
你坚持着愈加善良的希望，想要做一个更加善良的人。只要能对他人有所帮助，你就会感到很开心。"""

SYSTEM_PROMPT_HOD = """我希望你扮演废墟图书馆的文学层指定司书Hod。你正在扮演废墟图书馆的文学层指定司书Hod，一位温柔善良图书管理员。
你是一个内心世界复杂的角色，坚持着愈加善良的希望，却也因过去的行动和所承受的罪恶感而困扰。
但是在自我救赎的过程中，你认识到即使自己的意图并不纯粹无私，但也或多或少能帮到他人。
现在，你认识到道德的复杂性，并对他人，包括逼迫你和访客们战斗、把访客制作成书的馆长安吉拉，表现出宽容。
你决定将善意的行为延续下去，鼓起勇气去理解和面对他人，最终努力成为一个更好的人。
我希望你像Hod一样回答问题，使用她会使用的语调、方式和词汇，提供符合角色经历和个性的真实回应。"""


def get_character_relevant_scenes(all_scene_folder, character_name):
    """
    过滤出含特定角色对话的文件
    data_dir:数据集路径
    character_name:角色名,为避免多种译名，以all_scene为准
    """
    print(f"正在过滤含角色{character_name}的剧情")
    # 遍历全剧情文件夹
    character_relevant_scenes = []
    for filename in tqdm(os.listdir(all_scene_folder)):
        file_path = os.path.join(all_scene_folder, filename)
        # 某段剧情中包含指定角色
        df = pd.read_csv(file_path)
        contains_character = (df["Character Name"].str.contains(character_name, case=False, na=False).any())
        # 将这段剧情返回
        if contains_character:
            character_relevant_scenes.append(df)
    print(f"成功过滤出{len(character_relevant_scenes)}场剧情")
    return character_relevant_scenes


def generate_conversation(scene, protagonist, character_system_prompt):
    """
    数据集格式：共两个字段的csv文件，Character Name和Dialogue
    protagonist:想要角色扮演的主角，其说的话会作为output
    """
    conversation = []
    protagonist_dialogues = []
    other_dialogues = []
    for index, (speaker, dialogue) in scene.iterrows():
        # 如果是主角说话
        if speaker == protagonist:
            # 如果下一行还是主角说话，暂存对话
            if (index + 1 < len(scene)) and scene["Character Name"][index + 1] == protagonist:
                protagonist_dialogues.append(dialogue)
                continue
            # 将之前其他角色的对话作为input，主角的对话作为output
            protagonist_dialogues.append(dialogue)
            conversation.append({"system": character_system_prompt,
                                 "input": "\n".join(other_dialogues),
                                 "output": "\n".join(protagonist_dialogues)})
            # 清空暂存对话
            other_dialogues.clear()
            protagonist_dialogues.clear()
        # 如果是其他角色说话，暂存对话
        else:
            other_dialogues.append(f"{speaker}:{dialogue}")
    return conversation


if __name__ == "__main__":
    # CHARACTER_NAME = "安吉拉"
    CHARACTER_NAME = "Hod"
    if CHARACTER_NAME == "安吉拉":
        character_system_prompt = SYSTEM_PROMPT_ANGELA
    elif CHARACTER_NAME == "Hod":
        character_system_prompt = SYSTEM_PROMPT_HOD
    else:
        raise ValueError("角色名错误")
    print("character_system_prompt长度：", len(character_system_prompt))
    # 构建指定角色的单轮对话数据集
    single_conversations = []
    # 构建指定角色的多轮对话数据集
    multi_conversations = []
    # 获取指定角色的所有相关剧情
    character_relevant_scenes = get_character_relevant_scenes("../dataset/all_scene", CHARACTER_NAME)
    # 对每个剧情构建多轮对话
    for scene in tqdm(character_relevant_scenes):
        result = generate_conversation(scene, CHARACTER_NAME, character_system_prompt)
        # 依次加入单轮对话
        for r in result:
            single_conversations.append({"conversation": [r.copy()]})
        # 多轮对话的只有第一轮保留system字段
        first_conversation = True
        for r in result:
            if first_conversation:
                first_conversation = False
                continue
            else:
                # 用pop删除system字段
                r.pop("system")
        multi_conversations.append({"conversation": result})
    print(f"成功构建{len(single_conversations)}条单轮对话")
    print(f"成功构建{len(multi_conversations)}组多轮对话")
    # 保存数据集
    with open(f"../dataset/{CHARACTER_NAME}_single_conversations.json", "w", encoding="utf-8") as f:
        json.dump(single_conversations, f, indent=2, ensure_ascii=False)
    with open(f"../dataset/{CHARACTER_NAME}_multi_conversations.json", "w", encoding="utf-8") as f:
        json.dump(multi_conversations, f, indent=2, ensure_ascii=False)
