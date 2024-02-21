<p align="center" width="100%">
  <img src="images\Angela.jpg" width="50%">
</p>

<h3 align="center">"正于此地，愿你找到想要的书"<h3>

# Angela：使用大模型进行角色扮演——安吉拉（废墟图书馆）

> 本项目是书生浦语大模型训练营角色扮演的一个大项目

## News

[2024-2-21] 爬取并格式化了共 133 个场景对话
[2024-2-22] 构造了 1066 条单轮数据（尚未精修）
[2024-2-22] 使用 xtuner,QLoRA 微调 InternLM-7B 模型

# 项目计划

## 收集数据

[用于收集数据的 Wiki](https://library-of-ruina.fandom.com/zh/wiki/%E5%89%A7%E6%83%85)

- [x] 爬取所有对话数据并保存为统一格式
- [x] 对爬取的数据进行清晰，例如替换？？？中的名字，过滤仅含安吉拉的对话
- [x] 构建符合微调数据集格式的对话（暂时单轮对话）
- [ ] （可选）收集 wiki 各种背景描述，用于增量微调或者 RAG

## 模型准备

- [x] 下载 InternLM2 1.8B、7B 模型
- [x] 跑通一个 demo，测试微调前可用

## 模型微调

- [x] 对模型进行指令微调
- [ ] 对模型进行增量微调，学习背景知识

## 模型部署

## 模型量化

## 进阶计划

- 使用安吉拉韩语配音，利用 VITS 转换中文
- 赋予安吉拉 Agent 能力
- 对话时进行 RAG（很多背景描述都在旁白中）

## 微调语料构建方案

### 方案一：纯单轮对话

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

### 参考资料

1. [安吉拉 wiki](https://libraryofruina.huijiwiki.com/wiki/%E5%AE%89%E5%90%89%E6%8B%89)
2. [凉宫春日计划](https://github.com/LC1332/Chat-Haruhi-Suzumiya)
3. [赫萝微调数据集](https://huggingface.co/datasets/while-nalu/horo2ds/tree/main)
4. [xtuner 数据集格式](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/dataset_format.md)

### 特别鸣谢
- 上海人工智能实验室提供的算力平台
- 书生浦语大模型角色扮演群里大佬的指点
