from speakers.common.registry import registry
from speakers.server.bootstrap import load_bootstrap, get_bootstrap

from omegaconf import OmegaConf

from speakers.common.utils import get_abs_path

import asyncio
import time
import os
import sys

import subprocess

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("server_library_root", root_dir)
# Time to wait for web client to send a request to /task-state request
# before that web clients task gets removed from the queue
WEB_CLIENT_TIMEOUT = -1
# Time before finished tasks get removed from memory
FINISHED_TASK_REMOVE_TIMEOUT = 1800


def start_translator_client_proc(speakers_config_file: str):
    cmds = [
        sys.executable,
        '-m', 'speakers',
        '--mode', 'web_runner',
        '--host', speakers_config_file,
        '--port', str(port)
    ]

    proc = subprocess.Popen(cmds, cwd=registry.get_path("library_root"))
    return proc


async def start_async_app(speakers_config_file: str):
    config = OmegaConf.load(get_abs_path(speakers_config_file))
    load_bootstrap(config=config.get("bootstrap"))

    runner_bootstrap_web = get_bootstrap("runner_bootstrap_web")
    await runner_bootstrap_web.run()
    return runner_bootstrap_web


async def dispatch(speakers_config_file: str):
    runner = await start_async_app(speakers_config_file=speakers_config_file)
    # Create client process that will execute translation tasks
    print()
    client_process = start_translator_client_proc(speakers_config_file)

    try:
        while True:
            await asyncio.sleep(1)

            # Restart client if OOM or similar errors occured
            if client_process.poll() is not None:
                print('Restarting translator process')
                if len(runner.ongoing_tasks) > 0:
                    task_id = runner.ongoing_tasks.pop(0)
                    state = runner.task_states[task_id]
                    state['info'] = 'error'
                    state['finished'] = True
                client_process = start_translator_client_proc(host, port)

            # Filter queued and finished tasks
            now = time.time()
            to_del_task_ids = set()
            for tid, s in runner.task_states.items():
                flowData = runner.task_data[tid]
                # Remove finished tasks after 30 minutes
                if s['finished'] and now - flowData.created_at > FINISHED_TASK_REMOVE_TIMEOUT:
                    to_del_task_ids.add(tid)

                # Remove queued tasks without web client
                elif WEB_CLIENT_TIMEOUT >= 0:
                    if tid not in runner.ongoing_tasks and not s['finished'] \
                            and now - d['requested_at'] > WEB_CLIENT_TIMEOUT:
                        print('REMOVING TASK', tid)
                        to_del_task_ids.add(tid)
                        try:
                            runner.deque.remove(tid)
                        except Exception:
                            pass

            for tid in to_del_task_ids:
                del runner.task_states[tid]
                del runner.task_data[tid]

    except:
        if client_process.poll() is None:
            # client_process.terminate()
            client_process.kill()
        await runner.destroy()
        raise

    finally:

        if client_process.poll() is None:
            # client_process.terminate()
            client_process.kill()
        await runner.destroy()
