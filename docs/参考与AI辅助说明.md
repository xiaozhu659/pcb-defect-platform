# 参考与 AI 辅助说明

## 1. 参考资料

- 第二阶段项目:
  https://github.com/xiaozhu659/pcb-defect-comparison
- FastAPI 文档: https://fastapi.tiangolo.com/
- SQLAlchemy 文档: https://docs.sqlalchemy.org/
- Gradio 文档: https://www.gradio.app/docs
- Ultralytics YOLO 与 SAM 文档: https://docs.ultralytics.com/
- Redis 文档: https://redis.io/docs/
- RQ 文档: https://python-rq.org/
- Docker Compose 文档: https://docs.docker.com/compose/
- nginx 文档: https://nginx.org/en/docs/
- DeepSeek API 文档: https://api-docs.deepseek.com/

## 2. 项目创新点

1. **双方案同图对比**:同一张 PCB 图片同时经过传统检测模型和视觉大模型,
   并在网页中并排展示,差异比单独汇报一个准确率更直观。
2. **检测与分割组合**:YOLO 负责定位,SAM 负责生成精细掩膜,可以计算缺陷面积,
   补足大模型只能给出文字描述的不足。
3. **从脚本到平台**:把第二阶段的离线实验改造成上传、异步任务、状态查询和结果下载
   的完整 Web 流程。
4. **资源受限部署**:针对 2 GiB CPU 云服务器限制线程、SAM 输入分辨率和
   Worker 内存,并在失败时保持网页可访问。

## 3. AI 辅助的范围

本项目在开发过程中使用了 AI 编程助手,主要用于:

- 梳理第三阶段任务与第二阶段成果的复用关系;
- 生成 FastAPI、SQLAlchemy、Gradio、Redis 和 Docker 的基础代码;
- 协助定位 Docker 构建、nginx 502、云服务器网络和内存问题;
- 生成测试草稿、文档初稿和演示流程;
- 检查代码结构与交付材料是否完整。

以下内容经过实际运行和人工确认:

- 在 Ubuntu 22.04 服务器上启动四个容器;
- 登录、上传图片、查询任务并下载两种方案的结果图;
- 自动冒烟测试覆盖 JWT、Redis 队列、上传校验和结果图;
- YOLO 与 SAM 模型和 DeepSeek API 使用真实输入执行,没有伪造实验输出。

项目中的代码和结论仍由本人负责理解、运行和解释;未经验证的指标不会写入文档。
