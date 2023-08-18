from collections import deque
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from speakers.server.utils import MakeFastAPIOffline
from speakers.server.model.result import BaseResponse
from speakers.server.servlet.document import document
from speakers.server.servlet.runner import submit_async
from speakers.server.model.flow_data import BaseFlowData
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
    """任务队列"""
    _QUEUE: deque = deque()
    """进行的任务数据"""
    _TASK_DATA: Dict[str, BaseFlowData] = {}
    """进行的任务状态"""
    _TASK_STATES = {}
    """正在进行的任务"""
    _ONGOING_TASKS = {}

    def __init__(self, host: str, port: int):
        super().__init__()

        self.host = host
        self.port = port

    @classmethod
    def from_config(cls,  cfg=None):

        host = cfg.get("host")
        port = cfg.get("port")
        return cls(host=host, port=port)

    @property
    def version(self):
        return self._version

    @property
    def deque(self) -> deque:
        return self._QUEUE

    @property
    def task_data(self) -> Dict[str, BaseFlowData]:
        return self._TASK_DATA

    @property
    def task_states(self) -> dict:
        return self._TASK_STATES

    @property
    def ongoing_tasks(self) -> dict:
        return self._ONGOING_TASKS

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
                     summary="swagger 文档")(document)
        self.app.post("/runner/submit",
                      tags=["Runner"],
                      summary="调度Runner")(submit_async)
        app = self.app

        def run_server():
            uvicorn.run(app, host=self.host, port=self.port)

        server_thread = threading.Thread(target=run_server)
        server_thread.start()

    async def destroy(self):
        server_thread = self.server_thread
        app = self.app

        @app.on_event("shutdown")
        def shutdown_event():
            server_thread.join()  # 等待服务器线程结束



