from lxml import etree
import requests
import pandas as pd
import os
from urllib.parse import urljoin
from tqdm import tqdm

def get_all_urls(save_path="all_urls.csv"):
    """获取所有剧情对话url,并保存本地"""
    response = requests.get("https://library-of-ruina.fandom.com/zh/wiki/%E5%89%A7%E6%83%85")
    html_tree = etree.HTML(response.text)
    urls = html_tree.xpath("//td//a[@title]/@href")
    scene_names = html_tree.xpath("//td//a[@title]/@title")
    urls = [urljoin("https://library-of-ruina.fandom.com", url) for url in urls]
    urls_df = pd.DataFrame({"scene_name": scene_names, "url": urls})
    urls_df.drop_duplicates(inplace=True)
    urls_df["scene_name"] = urls_df["scene_name"].str.replace("/", "")
    urls_df.to_csv(save_path, index=False)
    print(f"Successfully saved to {save_path}")
    return urls_df


def extract_dialogues(url):
    """抽取给定url中的剧情对话"""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request error: {e} When scraping {url}")
        return pd.DataFrame()
    html_tree = etree.HTML(response.text)
    character_names = html_tree.xpath("//span[@style='font-size: 1.2em']/text()")
    dialogues = html_tree.xpath("//div[contains(@style, 'margin-top:10px') and contains(@style, 'width:70%')]/text()")
    # 保证对话数和角色出现次数一致
    if len(character_names) != len(dialogues):
        print("Warning: Mismatch between number of characters and dialogues.")
        return pd.DataFrame()

    dialogue_df = pd.DataFrame({"Character Name": character_names, "Dialogue": dialogues})
    return dialogue_df

if __name__ == "__main__":
    SAVE_DIR = "dataset/raw_data"
    urls_df = get_all_urls()
    os.makedirs(SAVE_DIR, exist_ok=True)
    for index, (scene_name, url) in tqdm(urls_df.iterrows(), total=len(urls_df)):
        if os.path.exists(os.path.join(SAVE_DIR, scene_name)):
            continue
        else:
            dialogue_df = extract_dialogues(url)
            dialogue_df.to_csv(os.path.join(SAVE_DIR, scene_name + ".csv"), index=None)
