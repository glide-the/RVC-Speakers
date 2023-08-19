from speakers.server.model.flow_data import VoiceFlowData
from speakers.server.model.result import BaseResponse
from speakers.server.bootstrap.bootstrap_register import get_bootstrap
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
        runner_bootstrap_web.deque.append(task_id)
        runner_bootstrap_web.task_states[task_id] = {
            'info': 'pending',
            'finished': False,
        }


    return BaseResponse(code=200, msg="提交任务成功")


async def get_task_async(request):
    """
    Called by the translator to get a translation task.
    """
    global NONCE, ONGOING_TASKS, DEFAULT_TRANSLATION_PARAMS
    if constant_compare(request.rel_url.query.get('nonce'), NONCE):
        if len(QUEUE) > 0 and len(ONGOING_TASKS) < MAX_ONGOING_TASKS:
            task_id = QUEUE.popleft()
            if task_id in TASK_DATA:
                data = TASK_DATA[task_id]
                for p, default_value in DEFAULT_TRANSLATION_PARAMS.items():
                    current_value = data.get(p)
                    data[p] = current_value if current_value is not None else default_value
                if not TASK_DATA[task_id].get('manual', False):
                    ONGOING_TASKS.append(task_id)
                return web.json_response({'task_id': task_id, 'data': data})
            else:
                return web.json_response({})
        else:
            return web.json_response({})
    return web.json_response({})
