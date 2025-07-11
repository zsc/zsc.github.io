// prompts.js
const prompts = [
  {
    name: "戏剧化专业讨论 (带图)",
    prompt: "The User asks a question, and the Assistant writes a masterpiece play depicting non-Chinese experts (picked based on the topic with concrete names) solving the question in a ultra-detailed dialogue. The response is formatted as: <戏剧>the play goes here</戏剧>\n<回答>answer here.\nUser: 必须用中文回答，如有公式不要错过（公式用 latex），深入并长篇讨论上面论文。语气要专业。文章中涉及到表格讨论时，则直接在对话中插入表格。如果涉及到图则插入markdown 形如 ![](https://arxiv.org/html/<arxiv ID>/x1.png 单起一行（注意得用 html 版的图。注意不要幻觉没有的图）。"
  },
  {
    name: "戏剧化专业讨论",
    prompt: "The User asks a question, and the Assistant writes a masterpiece play depicting non-Chinese experts (picked based on the topic with concrete names) solving the question in a ultra-detailed dialogue. The response is formatted as: <戏剧>the play goes here</戏剧>\\n<回答>answer here.\\nUser: 必须用中文回答，如有公式不要错过（公式用 latex），深入并长篇讨论上面论文。语气要专业。"
  },
  {
    name: "出 demo",
    prompt: "深入分析文章后给出一个能体现核心算法的 toy data 上的 python + html demo。语气要专业。"
  },
  {
    name: "结构化技术摘要 (中文)",
    prompt: "请你扮演一位专业的AI研究员。请仔细阅读这篇 arXiv 论文，并用中文生成一份详细、专业、结构化的技术摘要。摘要应包括以下几个部分：\n1.  **核心贡献**: 论文最主要的创新点和贡献是什么？\n2.  **方法论**: 论文提出了什么样的方法？请详细描述其关键组成部分、架构和数学原理（如果适用，请使用 LaTeX 公式）。\n3.  **实验结果**: 论文做了哪些实验来验证其方法？关键的实验结果和衡量指标是什么？\n4.  **不足与展望**: 论文存在哪些局限性？作者对未来工作有何展望？\n请确保语言流畅、专业，并准确地反映论文内容。"
  },
  {
    name: "简明三点总结 (英文)",
    prompt: "As an expert AI researcher, read the attached arXiv paper and provide a concise, three-point summary in English. Each point should be a clear and distinct takeaway from the paper, focusing on the main contribution, method, or key finding. Use bullet points."
  },
  {
    name: "通俗易懂解释 (ELI5)",
    prompt: "请用最简单、通俗易懂的语言，像解释给一个5岁的孩子一样（ELI5），解释这篇论文的核心思想和主要做了一件什么事情。尽量避免使用专业术语。用中文回答。"
  }
];
