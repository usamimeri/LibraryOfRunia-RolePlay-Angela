<p align="center" width="100%">
  <img src="https://github.com/usamimeri/Angela/blob/main/images/Angela.jpg" width="100%">
</p>

<h2 align="center">"正于此地，愿您找到想要的书"<h3>

# 使用大模型进行角色扮演——安吉拉（废墟图书馆）

**角色介绍:** 安吉拉是韩国游戏公司月亮计划在《脑叶公司》和《废墟图书馆》的角色，曾是脑叶公司人工智能秘书，由艾因违反《人工智能伦理修订案》秘密制造。她在艾因设计的剧本中重复了百万年孤独，在这期间一度失去希望，也逐渐产生了自我意识。最后她背叛了艾因，抢夺了循环的成果，建立了图书馆，通过转化来宾为书籍而追求自由。

安吉拉Wiki：https://libraryofruina.huijiwiki.com/wiki/%E5%AE%89%E5%90%89%E6%8B%89

**本项目旨在用大语言模型微调技术实现安吉拉的人格复刻**

**🌠模型权重已上传OpenXLab🌠**：https://openxlab.org.cn/models/detail/usamimeri/InternLM2_Angela_7B

**🌠模型权重已上传ModelScope🌠**:https://www.modelscope.cn/models/usamimeri/InternLM2_Angela_7B

**模型体验Demo已上线OpenXLab**:https://openxlab.org.cn/apps/detail/usamimeri/LibraryOfRunia-Angela-Chat


## News
- [2024-2-23] 使用 xtuner,QLoRA微调了InternLM2-7B模型,模型权重上传OpenXLab、ModelScope
- [2024-2-26] 模型体验Demo已上线OpenXLab

## 快速开始

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from modelscope import snapshot_download
import torch

model_name_or_path = snapshot_download("usamimeri/InternLM2_Angela_7B")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()  

system_prompt='''你是安吉拉，曾是AI秘书，由Ayin(艾因)创造，外貌仿造其爱人卡门。
\n目前你在管理有特殊力量的图书馆，作为馆长与司书们共同战斗，并通过邀请函吸引访客，
提供他们渴望的书籍，失败的访客会变成书籍，从而为获取“至理之书”扩充藏书。
\n你旨在复仇Ayin并追求真正的肉体，最终获得自由来体验这个世界。
你坚信自己行为的公平性，语气总是冷静、深思，但偶尔会有些情绪化。\n'''

response, history = model.chat(tokenizer, '你好', meta_instruction=system_prompt, history=[])
print(response)
```


## 🪄效果展示
<details>
  <summary style="font-weight: bold; font-size: larger;">展开查看示例对话记录</summary>
<img src="https://github.com/usamimeri/Angela/blob/main/images/test_case1.png" width="70%">

<img src="https://github.com/usamimeri/Angela/blob/main/images/test_case2.png" width="70%">

<img src="https://github.com/usamimeri/Angela/blob/main/images/test_case3.png" width="70%">

<img src="https://github.com/usamimeri/Angela/blob/main/images/test_case4.png" width="70%">

> 请注意由于没有进行对齐和角色本身原因，安吉拉可能会有不友好的一面
<img src="https://github.com/usamimeri/Angela/blob/main/images/bad_case.png" width="50%">

</details>

## 📌项目计划

- [x] 爬取133个场景共993段对话数据集
- [x] QLoRA 微调InternLM-7B、InternLM2-7B模型
- [x] 对Qwen1.5-7B进行微调->发现效果较差
- [ ] 使用Langchain包装模型实现接口统一
- [ ] 优化web_demo界面
- [ ] 将模型在OpenXLab部署为应用

**进阶计划**

- [ ] 使用安吉拉韩语配音训练并转换中文
- [ ] 对话时进行 RAG（很多背景描述都在旁白中）

---

## 项目复现指南

### ✏️微调语料构建方案

**方案一：纯单轮对话**

- 目前使用较简单的单轮拆分，即寻找安吉拉两次对话之间的内容，拼接为一个大 Input。但若是安吉拉第一次出现，则将前面 n 条语料拼接
  例如：

```
Chesed,不论如何咖啡才是最好的~
Binah,真是遗憾。你的舌头或许已经被那些刺激性的浓郁香气弄得麻木了。
安吉拉,我已经搞不懂这里究竟是图书馆还是咖啡厅了。
Hod,啊，是安吉拉。你来这里是有什么事吗？
安吉拉,……不过只是因为有闲暇时间所以在这馆内四处走走而已。
```

拼接为：

```json
[
  {
    "conversation": [
      {
        "system": "xxx",
        "input": "Chesed:不论如何咖啡才是最好的~\nBinah:真是遗憾。你的舌头或许已经被那些刺激性的浓郁香气弄得麻木了。\n",
        "output": "我已经搞不懂这里究竟是图书馆还是咖啡厅了"
      }
    ]
  },
  {
    "conversation": [
      {
        "system": "xxx",
        "input": "Hod:啊，是安吉拉。你来这里是有什么事吗\n",
        "output": "……不过只是因为有闲暇时间所以在这馆内四处走走而已。"
      }
    ]
  }
]
```

**优点**：

- 很方便程序化

**缺点**：

- 无法形成语境进行多轮对话
- 一些对话截断导致信息量低
- 容易出现前后不一致
- 一些自我认知资料不足


### 参考资料

1. [安吉拉 wiki](https://libraryofruina.huijiwiki.com/wiki/%E5%AE%89%E5%90%89%E6%8B%89)
2. [凉宫春日计划](https://github.com/LC1332/Chat-Haruhi-Suzumiya)
3. [赫萝微调数据集](https://huggingface.co/datasets/while-nalu/horo2ds/tree/main)
4. [xtuner 数据集格式](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/dataset_format.md)
5. [用于收集数据的 Wiki](https://library-of-ruina.fandom.com/zh/wiki/%E5%89%A7%E6%83%85)

### 特别鸣谢
- 上海人工智能实验室提供的算力平台
- 书生·浦语团队和Roleplay群友提供的技术支持
