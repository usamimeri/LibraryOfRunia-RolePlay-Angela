<p align="center" width="100%">
  <img src="images\Angela.jpg" width="50%">
</p>

<h3 align="center">"正于此地，愿你找到想要的书"<h3>

# Angela：使用大模型进行角色扮演——安吉拉（废墟图书馆）
> 本项目是书生浦语大模型训练营角色扮演的一个大项目

## News
[2024-2-21] 爬取并格式化了共133个场景对话

# 项目计划

## 收集数据
[用于收集数据的Wiki](https://library-of-ruina.fandom.com/zh/wiki/%E5%89%A7%E6%83%85)

- [x] 爬取所有对话数据并保存为统一格式 
- [ ] 构建符合微调数据集格式的对话（暂时单轮对话）
- [ ] （可选）收集wiki各种背景描述，用于增量微调或者RAG

## 模型准备
- [ ] 下载InternLM2 1.8B、7B模型
- [ ] 跑通一个demo，测试微调前可用

## 模型微调
- [ ] 对模型进行指令微调
- [ ] 对模型进行增量微调，学习背景知识

## 模型部署

## 模型量化

## 进阶计划
- 使用安吉拉韩语配音，利用VITS转换中文
- 赋予安吉拉Agent能力
- 对话时进行RAG（很多背景描述都在旁白中）

### 参考资料

1. [安吉拉wiki](https://libraryofruina.huijiwiki.com/wiki/%E5%AE%89%E5%90%89%E6%8B%89)
2. [凉宫春日计划](https://github.com/LC1332/Chat-Haruhi-Suzumiya)
3. [赫萝微调数据集](https://huggingface.co/datasets/while-nalu/horo2ds/tree/main)

