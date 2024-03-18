from lxml import etree
import requests
import pandas as pd
import os
from urllib.parse import urljoin
from tqdm import tqdm


def get_all_scene_urls(save_path="../dataset/all_scene_urls.csv"):
    """获取所有剧情对话url,并保存本地"""
    # 检查本地是否有已保存的数据
    if os.path.exists(save_path):
        urls_df = pd.read_csv(save_path)
        print("从本地加载剧情url成功,共%d条剧情url" % len(urls_df))
        return urls_df
    # 剧情总揽界面url
    response = requests.get("https://library-of-ruina.fandom.com/zh/wiki/Category:%E5%89%A7%E6%83%85")
    # 获取每段剧情的名称与url后缀
    html_tree = etree.HTML(response.text)
    urls = html_tree.xpath("//a[@class='category-page__member-link']/@href")
    scene_names = html_tree.xpath("//a[@class='category-page__member-link']/@title")
    # 将剧情url的前缀补全
    urls = [urljoin("https://library-of-ruina.fandom.com", url) for url in urls]
    # 保存到本地
    urls_df = pd.DataFrame({"scene_name": scene_names, "url": urls})
    urls_df.drop_duplicates(inplace=True)
    urls_df.reset_index(drop=True, inplace=True)
    urls_df["scene_name"] = urls_df["scene_name"].astype(str).str.replace("/", "")
    urls_df.to_csv(save_path, index=False)
    print("从wiki获取剧情url成功,共%d条剧情url" % len(urls_df))
    return urls_df


def extract_dialogues(scene_name, url):
    """抽取给定url中的剧情对话"""
    # url访问失败时返回None
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"错误：剧情: {scene_name}, 请求地址: {url}\n请求错误: {e}")
        return None
    # 获取角色名和对话内容
    html_tree = etree.HTML(response.text)
    character_names = html_tree.xpath("//span[@style='font-size: 1.2em']")
    # 避免一些旁白无文本导致的xpath无法匹配
    character_names = [node.text if node.text is not None else "" for node in character_names]
    dialogues = html_tree.xpath("//div[@class='story-mobile']")
    dialogues = [node.text if node.text is not None else "" for node in dialogues]
    # 跳过空白对话
    character_names_length = len(character_names)
    dialogues_length = len(dialogues)
    if character_names_length == 0:
        print(f"警告：剧情: {scene_name}, 请求地址: {url}中角色出现次数为0")
        return None
    if dialogues_length == 0:
        print(f"警告：剧情: {scene_name}, 请求地址: {url}中对话数为0")
        return None
    # 保证对话数和角色出现次数一致
    if character_names_length != dialogues_length:
        print(
            f"警告：剧情: {scene_name}, 请求地址: {url}中对话数为:{dialogues_length},而角色出现次数为{character_names_length}")
        return None
    dialogue_df = pd.DataFrame({"Character Name": character_names, "Dialogue": dialogues})
    return dialogue_df


if __name__ == "__main__":
    # 获取游戏中每段剧情的name与url
    urls_df = get_all_scene_urls()
    # 依次获取每段剧情对话内容，并保存到本地
    # 考虑到部分剧情对话可能获取失败，本文件可以多次重复运行
    # 每次运行时会检查本地是否已经保存了某段剧情对话，只获取未保存的剧情对话
    SCENE_SAVE_DIR = "../dataset/all_scene"
    os.makedirs(SCENE_SAVE_DIR, exist_ok=True)
    success_count = 0
    for index, (scene_name, url) in tqdm(urls_df.iterrows(), total=len(urls_df)):
        if os.path.exists(os.path.join(SCENE_SAVE_DIR, scene_name + ".csv")):
            success_count += 1
            continue
        else:
            dialogue_df = extract_dialogues(scene_name, url)
            if dialogue_df is not None:
                dialogue_df.to_csv(os.path.join(SCENE_SAVE_DIR, scene_name + ".csv"), index=False)
                success_count += 1
    print(f"共{len(urls_df)}条剧情url，已成功获取其中{success_count}段剧情对话")
