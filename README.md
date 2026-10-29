# PCB 缺陷检测平台

> **在线演示:** http://39.107.27.28/ui/
>
> **接口文档:** http://39.107.27.28/docs

灵境竞赛组 · 后端 AI 方向 · **第三阶段**考核项目

把第二阶段的"双方案对比"做成了一个可以在浏览器里操作的应用:
上传一张 PCB 照片,后端同时运行**传统检测模型(YOLO + SAM)**与
**视觉大模型(DeepSeek)**,并排展示两种方案的结果。

## 在线演示

当前部署页面:

```text
http://39.107.27.28/ui/
```

该地址只在云服务器租用期内有效。需要在自己的电脑上运行时,按下面的“本机启动”步骤启动服务。

## 功能

- 图片上传与任务提交(FastAPI);
- 上传大小、扩展名、文件内容和图片像素校验;
- 后台异步任务(Redis + RQ;本机没有 Redis 时自动退化为后台线程);
- 每次检测写入 SQLite 数据库(SQLAlchemy),可查询历史任务;
- 两种方案的可视化结果:左边 YOLO 检测框 + SAM 掩膜,右边大模型报出的位置;
- Gradio 页面,上传即用;
- Docker Compose 一键部署(web + worker + redis + nginx)。

## 架构

```
浏览器 ──> nginx ──> FastAPI ──> SQLite(任务记录)
                        │
                        ├── Redis 队列 ──> worker ──> YOLO + SAM
                        │                        └──> DeepSeek API
                        └── Gradio 页面(/ui)
```

## 本机启动(Windows 也能运行)

下面步骤会在当前电脑启动服务,执行后需要保持命令窗口运行。

```bash
uv sync

# 1. 准备模型权重(从第二阶段复制,或用环境变量指定路径)
#    models/yolo_best.pt   —— 第二阶段训练好的 YOLO 权重
#    models/sam2.1_t.pt    —— SAM 2.1 权重(ultralytics 也可自动下载)

# 2. 设置 API key(只放环境变量)
$env:DEEPSEEK_API_KEY = "你的 key"      # PowerShell

# 3. 启动服务
uv run uvicorn pcb_defect_platform.main:app --host 127.0.0.1 --port 8000
```

本机服务启动后,打开 http://127.0.0.1:8000/ui 使用页面;
接口文档在 http://127.0.0.1:8000/docs。

如果提示“无法访问”或“连接被拒绝”,通常是没有执行上面的启动命令,而不是公网页面故障。
云服务器页面请使用本文开头的在线演示地址。

部署后可运行自动验收脚本:

```bash
uv run python scripts/smoke_test.py \
  --base-url http://127.0.0.1 \
  --image path/to/pcb.jpg
```

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务状态,以及当前用的是 Redis 队列还是本地线程(公开) |
| POST | `/api/login` | 账号密码换取 JWT token(公开) |
| POST | `/api/jobs` | 上传图片,创建检测任务(需 token) |
| GET | `/api/jobs/{id}` | 查询任务状态与两种方案的结果(需 token) |
| GET | `/api/jobs` | 最近的任务列表(需 token) |
| GET | `/api/files/{id}/yolo` | 传统方案的可视化图(需 token) |
| GET | `/api/files/{id}/llm` | 大模型方案的可视化图(需 token) |

## 身份认证(JWT)

除 `/health`、`/api/login` 和页面本身外,所有接口都要求
`Authorization: Bearer <token>`。

```bash
# 1. 登录拿 token
curl -X POST http://localhost/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# 2. 带 token 提交任务
curl -X POST http://localhost/api/jobs \
  -H "Authorization: Bearer <token>" \
  -F "file=@test.jpg"
```

账号密码来自环境变量 `APP_USER` / `APP_PASSWORD`,签名密钥来自 `JWT_SECRET`,
默认值仅供本地测试,**部署前务必修改**(见 `.env.example`)。

## Docker 部署

```bash
# 1. 准备 .env(照 .env.example 填 key)
copy .env.example .env

# 2. 启动 nginx + web + redis
docker compose -f deploy/docker-compose.yml up -d --build
```

启动后访问 `http://服务器IP/`。

> 模型权重不打进镜像,而是以只读卷挂载(`deploy/docker-compose.yml`),
> 这样镜像更小,换模型也不用重新构建。

2 GiB 内存的 CPU 服务器上建议先增加 2 GiB Swap。Compose 已把 Worker
限制在 1150 MiB,并将 SAM 分割输入设为 512,避免首次推理拖慢整个网页服务。
部署和验收的完整步骤见 [docs/部署与验收.md](docs/部署与验收.md)。
实际运行结果见 [docs/验收记录.md](docs/验收记录.md)。

## 目录结构

```
.
├── src/pcb_defect_platform/
│   ├── config.py    # 路径、模型、数据库、队列配置
│   ├── db.py        # SQLAlchemy 任务表
│   ├── pipeline.py  # YOLO + SAM 与 DeepSeek 推理
│   ├── worker.py    # 队列任务处理
│   ├── ui.py        # Gradio 页面
│   └── main.py      # FastAPI 接口
├── deploy/          # Dockerfile / compose / nginx
├── docs/            # 项目说明与部署文档
├── models/          # 模型权重(不入库)
├── data/            # SQLite 数据库(不入库)
└── uploads/         # 上传的图片(不入库)
```

## 与第二阶段的关系

模型与推理逻辑来自第二阶段仓库
[pcb-defect-comparison](https://github.com/xiaozhu659/pcb-defect-comparison):
其中 YOLO 在验证集上 mAP50 为 0.956,SAM 用于把检测框细化成可量面积的掩膜。
本阶段新增的是 Web 接口、异步队列、数据库记录与容器化部署。

## 交付文档

- [项目说明](docs/项目说明.md) / [PDF](docs/pdf/PCB缺陷检测平台-项目说明.pdf)
- [部署与验收](docs/部署与验收.md) / [PDF](docs/pdf/PCB缺陷检测平台-部署与验收.pdf)
- [验收记录](docs/验收记录.md)
- [参考与 AI 辅助说明](docs/参考与AI辅助说明.md)
- [示例输入与参考结果](examples/README.md)

## 当前进度

- [x] FastAPI 后端与任务接口
- [x] SQLite 任务记录(SQLAlchemy)
- [x] Redis + RQ 队列(本地自动退化为线程)
- [x] 两种方案的推理封装与可视化
- [x] Gradio 上传页面
- [x] Docker Compose 与 nginx 配置
- [x] Docker 本机实测(web + worker + redis + nginx 四容器,队列走 Redis)
- [x] JWT 登录与接口鉴权
- [x] Linux 云服务器部署与端到端实测
- [x] 上传安全校验、单元测试与端到端自动验收
- [x] 演示视频录制清单、逐段旁白稿与个人总结
- [x] 源码、模型权重、说明文档与交付说明整理完成

视频文件由本人按照旁白稿录制后放入交付目录;技术开发、部署、测试和交付材料均已准备完成。
