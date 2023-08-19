from pydantic import BaseModel
import pydantic

from speakers.server.model.flow_data import VoiceFlowData


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class TaskVoiceFlowInfo(BaseModel):
    task_id: str
    data: VoiceFlowData


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
