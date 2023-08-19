from pydantic import BaseModel, Field

from speakers.server.model.flow_data import PayLoad


class BaseResponse(BaseModel):
    code: int = Field(200, description="HTTP status code")
    msg: str = Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class TaskRunnerResponse(BaseResponse):
    data: dict


class TaskVoiceFlowInfo(BaseModel):
    task_id: str
    data: PayLoad


class TaskInfoResponse(BaseResponse):
    data: TaskVoiceFlowInfo

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": None,
            }
        }


class RunnerState(BaseModel):
    """RunnerState"""
    task_id: str
    runner_stat: str
    nonce: str
    state: str
    finished: bool = Field(default=False)
