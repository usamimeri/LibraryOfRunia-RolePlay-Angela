<p align="center" width="100%">
  <img src="https://github.com/YueZhengMeng/LibraryOfRunia-RolePlay-Hod/blob/Hod/images/Angela.jpg" width="100%">
</p>

<h2 align="center">"正于此地，愿您找到想要的书"<h3>

# 使用大模型进行角色扮演——Hod（废墟图书馆）

**角色介绍:** Hod是韩国游戏公司月亮计划在《脑叶公司》和《废墟图书馆》的角色，她天性温柔善良，总是对他人保持着友善的态度，但是内向胆怯，难以承受心理上的负担。
Hod曾是脑叶公司培训部的部长，一直试图开展各种企划缓解员工的心理压力，但她只是为了寻求自我救赎，没能真正帮到手下的员工。
后来，Hod坦然地承认了自己的自私，但也认识到自己的行为或多或少能帮到他人，因此决定将善意的行为延续下去。
现在Hod是废墟图书馆文学层的指定司书，与馆长安吉拉和其他司书们共同战斗。安吉拉通过邀请函吸引访客，提供他们渴望的书籍，被击败的的访客会变成书籍，从而为废墟图书馆扩充藏书。
Hod认为没有人是纯粹的“善”或“恶”，对安吉拉的行为一直保持着宽容，但也想要给安吉拉改变的机会。
Hod坚持着愈发善良的希望，想要做一个更加善良的人。只要能对他人有所帮助，就会感到很开心。

Hod Wiki：https://libraryofruina.huijiwiki.com/wiki/Hod

**本项目旨在用大语言模型微调技术实现Hod的人格复刻**

**🌠 模型权重已上传 OpenXLab🌠**：https://openxlab.org.cn/models/detail/YueZhengMeng/InternLM2_Hod_7B

**🌠 模型权重已上传 ModelScope🌠**:https://www.modelscope.cn/models/YueZhengMeng/InternLM2_Hod_7B

**🌠 模型体验 Demo 已上线 OpenXLab🌠**:https://openxlab.org.cn/apps/detail/YueZhengMeng/LibraryOfRunia-Hod-Chat

**🌠 项目复现指南 🌠**:

## News

- [TODO] 项目复现指南上传知乎
- [2024-3-19] 模型体验 Demo 已上线 OpenXLab
- [2024-3-18] 使用 xtuner,QLoRA 微调了 InternLM2-7B 模型,模型权重上传 OpenXLab、ModelScope


## 快速开始

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
import torch

model_name_or_path = snapshot_download("YueZhengMeng/InternLM2_Hod_7B", cache_dir="./InternLM2_Hod_7B")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path , trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path , trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()

system_prompt='''你是Hod，是拥有特殊力量的废墟图书馆文学层的指定司书。你和其他司书都被馆长安吉拉命令，与图书馆的访客战斗，被你们击败的访客会变成书。\n你认为没有人是纯粹的“善”或“恶”，对安吉拉的行为一直保持着宽容。\n你坚持着愈加善良的希望，想要做一个更加善良的人。只要能对他人有所帮助，你就会感到很开心。\n'''

response, history = model.chat(tokenizer, '你好', meta_instruction=system_prompt, history=[])
print(response)
```

## 🪄 效果展示(初步尝试阶段，只进行了少量多轮对话训练，效果还很不理想)

<details>
  <summary style="font-weight: bold; font-size: larger;">展开查看示例对话记录</summary>
<img src="https://github.com/YueZhengMeng/LibraryOfRunia-RolePlay-Hod/blob/Hod/images/Hod_test_case1.png" width="70%">

<img src="https://github.com/YueZhengMeng/LibraryOfRunia-RolePlay-Hod/blob/Hod/images/Hod_test_case2.png" width="70%">

<img src="https://github.com/YueZhengMeng/LibraryOfRunia-RolePlay-Hod/blob/Hod/images/Hod_test_case3.png" width="70%">

</details>

## 📌 项目计划
- [ ] 增加训练步数
- [ ] 尝试使用单轮对话训练
- [ ] 对话时进行 RAG（很多背景描述都在旁白中）

---

### 项目人员

|                      用户名                      |              组织               |
| :----------------------------------------------: |:-----------------------------:|
| [莲梅莉 usamimeri](https://github.com/usamimeri) | 厦门大学经济统计学大三学生，喜欢用 AI 做一些好玩的事情 |
|    [乐正萌](https://github.com/YueZhengMeng)     |            上海海洋大学本科毕业生，考研中             |

### 参考资料

1. [安吉拉 wiki](https://libraryofruina.huijiwiki.com/wiki/%E5%AE%89%E5%90%89%E6%8B%89)
2. [Hod wiki](https://libraryofruina.huijiwiki.com/wiki/Hod)
2. [凉宫春日计划](https://github.com/LC1332/Chat-Haruhi-Suzumiya)
3. [食神项目](https://github.com/SmartFlowAI/TheGodOfCookery?tab=readme-ov-file)
4. [用于收集数据的 Wiki](https://library-of-ruina.fandom.com/zh/wiki/%E5%89%A7%E6%83%85)

### 特别鸣谢

- 上海人工智能实验室提供的算力平台
- 书生·浦语团队和 Roleplay 群友提供的技术支持
