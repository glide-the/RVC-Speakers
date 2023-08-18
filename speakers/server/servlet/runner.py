from speakers.server.model.flow_data import VoiceFlowData
from speakers.server.model.result import BaseResponse
from speakers.server.bootstrap.bootstrap_register import bootstrap_register
from speakers.common.utils import get_abs_path
import hashlib
import os
import time


def calculate_md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()


async def submit_async(flowData: VoiceFlowData):
    """Adds new task to the queue"""
    runner_bootstrap_web = bootstrap_register.get_bootstrap_class("runner_bootstrap_web")

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
        runner_bootstrap_web.deque.append(task_id)
        runner_bootstrap_web.task_states[task_id] = {
            'info': 'pending',
            'finished': False,
        }
        runner_bootstrap_web.task_data[task_id] = flowData

    return BaseResponse(code=200, msg="提交任务成功")
