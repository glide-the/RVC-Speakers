import os
import asyncio
import logging
from argparse import Namespace
from speakers.common.log import get_logger, set_log_level
from speakers import set_main_logger, Speaker, WebSpeaker
from speakers.processors.rvc_speakers_processor import set_rvc_speakers_logger
from speakers.processors.vits_to_voice import set_vits_to_voice_logger
from speakers.server.model.flow_data import PayLoad
from speakers.server import dispatch as dispatch_web
import argparse

logger = logging.getLogger('start_logger')


def set_start_logger(l):
    global logger
    logger = l


async def dispatch(args: Namespace):
    args_dict = vars(args)

    logger.info(f'Running in {args.mode} mode')

    if args.mode in 'demo':

        speaker = Speaker(speakers_config_file=args.speakers_config_file, verbose=args.verbose)
        await speaker.preparation_runner(task_id="0", payload=PayLoad())
    elif args.mode in 'web':

        await dispatch_web(speakers_config_file=args.speakers_config_file)

    elif args.mode in 'web_runner':

        translator = WebSpeaker(speakers_config_file=args.speakers_config_file, verbose=args.verbose, nonce=args.nonce)
        await translator.listen()


def main():
    parser = argparse.ArgumentParser(prog='speakers',
                                     description='S')
    parser.add_argument('-m', '--mode', default='demo', type=str, choices=['demo', 'web', 'web_runner'],
                        help='Run ')

    parser.add_argument('-v', '--verbose', action='store_true', help='Print debug info in result folder')
    parser.add_argument("--speakers-config-file", type=str, default="speakers.yaml")
    parser.add_argument('--nonce', default='', type=str, help='Used by web module to decide which secret for securing '
                                                              'internal web server communication')

    args = None
    try:
        args = parser.parse_args()
        set_log_level(level=logging.DEBUG if args.verbose else logging.INFO)
        set_start_logger(get_logger(args.mode))
        set_main_logger(logger)
        set_rvc_speakers_logger(logger)
        set_vits_to_voice_logger(logger)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dispatch(args))
    except KeyboardInterrupt:
        if not args or args.mode != 'web':
            print()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if args and args.verbose else None)


if __name__ == '__main__':
    main()
