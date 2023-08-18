
import os
import asyncio
import logging
from argparse import Namespace
from speakers.common.log import get_logger, set_log_level
from speakers import set_main_logger, Speaker
import argparse


async def dispatch(args: Namespace):
    args_dict = vars(args)

    logger.info(f'Running in {args.mode} mode')

    if args.mode in 'demo':
        speaker = Speaker(speakers_config_file='speakers.yaml', verbose=args.verbose)
        await speaker.preparation_runner( params=args_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='speakers',
                                     description='S')
    parser.add_argument('-m', '--mode', default='demo', type=str, choices=['demo', 'batch', 'web', 'web_client', 'ws'],
                        help='Run demo in either single image demo mode (demo), web service mode (web), web client '
                             'which executes translation tasks for a webserver (web_client) or batch translation mode '
                             '(batch)')

    parser.add_argument('-v', '--verbose', action='store_true', help='Print debug')


    args = None
    try:
        args = parser.parse_args()
        set_log_level(level=logging.DEBUG if args.verbose else logging.INFO)
        logger = get_logger(args.mode)
        set_main_logger(logger)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dispatch(args))
    except KeyboardInterrupt:
        if not args or args.mode != 'web':
            print()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if args and args.verbose else None)
