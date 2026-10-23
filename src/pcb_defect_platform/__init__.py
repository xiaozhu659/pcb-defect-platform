"""PCB 缺陷检测平台:把第二阶段的双方案对比做成本地可操作的 Web 应用。

模块分工:
    config    路径、模型位置、数据库与队列配置
    db        SQLAlchemy 任务记录
    pipeline  YOLO + SAM 与 DeepSeek 视觉模型的推理封装
    worker    队列任务(Redis + RQ)
    ui        Gradio 可视化页面
    main      FastAPI 接口与页面挂载
"""


def main() -> None:
    """启动 Web 服务(命令行入口)。"""
    from .main import main as run

    run()


if __name__ == "__main__":
    main()
