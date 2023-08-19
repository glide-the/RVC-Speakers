from speakers.server.model.flow_data import VoiceFlowData
from speakers.server.model.result import BaseResponse, TaskInfoResponse, TaskVoiceFlowInfo
from speakers.server.bootstrap.bootstrap_register import get_bootstrap
from speakers.common.utils import get_abs_path
from fastapi import File, Form, Body, Query, UploadFile
import hashlib
import os
import time


def constant_compare(a, b):
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    if not isinstance(a, bytes) or not isinstance(b, bytes):
        return False
    if len(a) != len(b):
        return False

    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0


def calculate_md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()


async def submit_async(flowData: VoiceFlowData):
    """Adds new task to the queue"""

    runner_bootstrap_web = get_bootstrap("runner_bootstrap_web")

    task_id = f'{calculate_md5(flowData.vits.text)}-{flowData.vits.speaker_id}-{flowData.vits.language}' \
              f'-{flowData.rvc.model_index}-{flowData.rvc.f0_up_key}'
    now = time.time()
    flowData.created_at = now
    flowData.requested_at = now
    print(f'New `submit` task {task_id}')
    if os.path.exists(get_abs_path(f'result/{task_id}.wav')):
        runner_bootstrap_web.task_states[task_id] = {
            'info': 'saved',
            'finished': True,
        }
        runner_bootstrap_web.task_data[task_id] = flowData
    elif task_id not in runner_bootstrap_web.task_data or task_id not in runner_bootstrap_web.task_states:
        os.makedirs(get_abs_path('result'), exist_ok=True)

        runner_bootstrap_web.task_data[task_id] = flowData
        runner_bootstrap_web.queue.append(task_id)
        runner_bootstrap_web.task_states[task_id] = {
            'info': 'pending',
            'finished': False,
        }

    return BaseResponse(code=200, msg="提交任务成功")


async def get_task_async(nonce: str = Query(..., examples=["samples"])):
    """
    Called by the translator to get a translation task.
    """

    runner_bootstrap_web = get_bootstrap("runner_bootstrap_web")

    if constant_compare(nonce, runner_bootstrap_web.nonce):
        if len(runner_bootstrap_web.queue) > 0 and len(runner_bootstrap_web.ongoing_tasks) < runner_bootstrap_web.max_ongoing_tasks:
            task_id = runner_bootstrap_web.queue.popleft()
            if task_id in runner_bootstrap_web.task_data:
                data = runner_bootstrap_web.task_data[task_id]
                runner_bootstrap_web.ongoing_tasks.append(task_id)
                info = TaskVoiceFlowInfo(task_id=task_id, data=data)
                return TaskInfoResponse(code=200, msg="成功", data=info)

            else:
                return BaseResponse(code=200, msg="成功")
        else:
            return BaseResponse(code=200, msg="max_ongoing_tasks")
    return BaseResponse(code=401, msg="无法获取任务")
