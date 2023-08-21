from fastapi import FastAPI
from starlette.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from speakers.server.utils import MakeFastAPIOffline
from speakers.server.model.result import BaseResponse
from speakers.server.servlet.document import page_index, document
from speakers.server.servlet.runner import (submit_async,
                                            get_task_async,
                                            post_task_update_async,
                                            result_async)
from speakers.server.bootstrap.bootstrap_register import bootstrap_register
from speakers.server.bootstrap.base import Bootstrap
import uvicorn
import threading


@bootstrap_register.register_bootstrap("runner_bootstrap_web")
class RunnerBootstrapBaseWeb(Bootstrap):
    """
    Bootstrap Server Lifecycle
    """
    app: FastAPI
    server_thread: threading

    def __init__(self, host: str, port: int):
        super().__init__()

        self.host = host
        self.port = port

    @classmethod
    def from_config(cls, cfg=None):
        host = cfg.get("host")
        port = cfg.get("port")
        return cls(host=host, port=port)

    async def run(self):
        self.app = FastAPI(
            title="API Server",
            version=self.version
        )
        MakeFastAPIOffline(self.app)
        # Add CORS middleware to allow all origins
        # 在config.py中设置OPEN_DOMAIN=True，允许跨域
        # set OPEN_DOMAIN=True in config.py to allow cross-domain
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.get("/",
                     response_model=BaseResponse,
                     summary="演示首页")(page_index)
        self.app.get("/docs",
                     response_model=BaseResponse,
                     summary="swagger 文档")(document)
        self.app.post("/runner/submit",
                      tags=["Runner"],
                      summary="提交调度Runner")(submit_async)
        self.app.get("/runner/task-internal",
                     tags=["Runner"],
                     summary="内部获取调度Runner")(get_task_async)
        self.app.post("/runner/task-update-internal",
                      tags=["Runner"],
                      summary="内部同步调度RunnerStat")(post_task_update_async)
        self.app.get("/runner/result",
                     tags=["Runner"],
                     summary="获取任务结果")(result_async)
        app = self.app

        def run_server():
            uvicorn.run(app, host=self.host, port=self.port)

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.start()

    async def destroy(self):
        server_thread = self.server_thread
        app = self.app

        @app.on_event("shutdown")
        def shutdown_event():
            server_thread.join()  # 等待服务器线程结束
