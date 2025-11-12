# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import sys
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    (case-insensitive)\n"  # noqa: E501
    "For a flag:               vllm {subcmd} --help=max-model-len  (_ or - accepted)\n"  # noqa: E501
    "Documentation:            https://docs.vllm.ai\n"
)


def main():
    import vllm.entrypoints.cli.benchmark.main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        vllm.entrypoints.cli.benchmark.main,
    ]

    # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        #from vllm import platforms

        #if platforms.current_platform.is_unspecified():
            #from vllm.platforms.cpu import CpuPlatform

            #platforms.current_platform = CpuPlatform()
            #logger.info(
            #    "Unspecified platform detected, switching to CPU Platform instead."
            #)

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="workshop-london",
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.argv = ["vllm", "bench", "serve", "--model", "Qwen/Qwen2.5-Coder-3B", "--host", "127.0.0.1", "--port", "8080", "--dataset-name", "random", "--request-rate", "10", "--num-prompts", "100"]
    main()
