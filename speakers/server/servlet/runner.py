from speakers.server.model.flow_data import PayLoad
from speakers.server.model.result import (BaseResponse,
                                          TaskInfoResponse,
                                          TaskVoiceFlowInfo,
                                          RunnerState,
                                          TaskRunnerResponse)
from speakers.server.bootstrap.bootstrap_register import get_bootstrap
from speakers.common.utils import get_tmp_path
from fastapi import File, Form, Body, Query
from fastapi.responses import FileResponse
from speakers.common.registry import registry
import os
import time
import logging

logger = logging.getLogger('server_runner')


def set_server_runner_logger(l):
    global logger
    logger = l


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


async def submit_async(payload: PayLoad):
    """
        Adds new task to the queue
        task_id = f'{calculate_md5(flowData.vits.text)}-{flowData.vits.speaker_id}-{flowData.vits.language}' \
              f'-{flowData.vits.noise_scale}-{flowData.vits.speed}-{flowData.vits.noise_scale_w}' \
              f'-{flowData.rvc.model_index}-{flowData.rvc.f0_up_key}'
    """

    runner_bootstrap_web = get_bootstrap("runner_bootstrap_web")
    task = registry.get_task_class(payload.parameter.task_name)

    runner = task.prepare(payload=payload)
    task_id = runner.task_id
    now = time.time()
    payload.created_at = now
    payload.requested_at = now

    task_state = {}
    if os.path.exists(get_tmp_path(f'result/{task_id}.wav')):
        task_state = {
            'task_id': task_id,
            'info': 'saved',
            'finished': True,
        }
        if task_id not in runner_bootstrap_web.task_data or task_id not in runner_bootstrap_web.task_states:

            logger.info(f'New `submit` task {task_id}')
            runner_bootstrap_web.task_data[task_id] = payload
            runner_bootstrap_web.queue.append(task_id)
            runner_bootstrap_web.task_states[task_id] = task_state

    elif task_id not in runner_bootstrap_web.task_data or task_id not in runner_bootstrap_web.task_states:
        os.makedirs(get_tmp_path('result'), exist_ok=True)
        task_state = {
            'task_id': task_id,
            'info': 'pending',
            'finished': False,
        }

        logger.info(f'New `submit` task {task_id}')
        runner_bootstrap_web.task_data[task_id] = payload
        runner_bootstrap_web.queue.append(task_id)

        runner_bootstrap_web.task_states[task_id] = task_state
    else:
        task_state = runner_bootstrap_web.task_states[task_id]

    return TaskRunnerResponse(code=200, msg="提交任务成功", data=task_state)


async def get_task_async(nonce: str = Query(..., examples=["samples"])):
    """
    Called by the translator to get a translation task.
    """

    runner_bootstrap_web = get_bootstrap("runner_bootstrap_web")

    if constant_compare(nonce, runner_bootstrap_web.nonce):
        if len(runner_bootstrap_web.ongoing_tasks) < runner_bootstrap_web.max_ongoing_tasks:
            if len(runner_bootstrap_web.queue) > 0:
                task_id = runner_bootstrap_web.queue.popleft()
                if task_id in runner_bootstrap_web.task_data:
                    data = runner_bootstrap_web.task_data[task_id]
                    runner_bootstrap_web.ongoing_tasks.append(task_id)
                    info = TaskVoiceFlowInfo(task_id=task_id, data=data)
                    return TaskInfoResponse(code=200, msg="成功", data=info)

            return BaseResponse(code=200, msg="成功")

        else:
            return BaseResponse(code=200, msg="max_ongoing_tasks")
    return BaseResponse(code=401, msg="无法获取任务")


async def post_task_update_async(runner_state: RunnerState):
    """
    Lets the translator update the task state it is working on.
    """

    runner_bootstrap_web = get_bootstrap("runner_bootstrap_web")

    if constant_compare(runner_state.nonce, runner_bootstrap_web.nonce):
        task_id = runner_state.task_id
        if task_id in runner_bootstrap_web.task_states and task_id in runner_bootstrap_web.task_data:
            runner_bootstrap_web.task_states[task_id] = {
                'info': runner_state.state,
                'finished': runner_state.finished,
            }
            if runner_state.finished:
                try:
                    i = runner_bootstrap_web.ongoing_tasks.index(task_id)
                    runner_bootstrap_web.ongoing_tasks.pop(i)
                except ValueError:
                    pass

            logger.info(f'Task state {task_id} to {runner_bootstrap_web.task_states[task_id]}')

    return BaseResponse(code=200, msg="成功")


async def result_async(task_id: str = Query(..., examples=["task_id"])):
    filepath = get_tmp_path(f'result/{task_id}.wav')
    logger.info(f'Task  {task_id} result_async {filepath}')
    if os.path.exists(filepath):
        return FileResponse(
            path=filepath,
            filename=f"{task_id}.wav",
            media_type="multipart/form-data")
    else:
        return BaseResponse(code=500, msg=f"{task_id}.wav 读取文件失败")
