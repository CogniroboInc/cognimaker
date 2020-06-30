#! /usr/bin/env python

import sys


def serve():
    from cognimaker.runner.sagemaker import start_server

    start_server()


def train():
    from cognimaker.runner.sagemaker import SagemakerRunner
    from __COGNIMAKER_MODEL_MODULE__ import config  # import ModelConfig object

    runner = SagemakerRunner(config)
    runner.train_model()


def main(args):
    if not args:
        print(
            'Run this container passing either "train"'
            ' or "serve" as the command.'
        )
        return

    cmd = args[0]
    if cmd == "train":
        train()
    if cmd == "serve":
        serve()


if __name__ == "__main__":
    main(sys.argv[1:])
