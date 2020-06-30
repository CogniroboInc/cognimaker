from pathlib import Path
import json
from typing import Optional

import pandas as pd

from ..runner import ModelConfig, Runner
from ...evaluation.evaluator import Evaluator, EvaluationResult
from ...model import Model
from ...model.prediction import Prediction
from ...util._logger import get_logger
from ...util._encoder import DefaultEncoder


INDICATOR_FILE = 'indicator.json'
SCORE_FORMAT = "MODEL_SCORE={};"
METRIC_FORMAT = "{name}={value};"


def merge_params_with_types(params: dict, default_params: dict) -> dict:
    """
    Merge model params with default values converting types if necessary.
    """

    def get_with_type(key, obj, default):
        if key in obj:
            t = type(default)
            return t(obj[key])
        else:
            return default

    return {k: get_with_type(k, params, v) for k, v in default_params.items()}


class _Paths:
    prefix = Path("/opt/ml/")

    input_dir = prefix / "input" / "data"
    model_dir = prefix / "model"
    param_path = prefix / "input" / "config" / "hyperparameters.json"

    channel_train = "training"
    channel_pretrain_model = "pretrain_model"
    train_dir = input_dir / channel_train
    pretrain_model_dir = input_dir / channel_pretrain_model


class SagemakerRunner(Runner):
    def __init__(self, config: ModelConfig):
        self.config = config

        with open(_Paths.param_path, "r") as tc:
            params = json.load(tc)
        self.process_id = params.get("process_id", "xxxxxxxx")
        self.is_finetune = params.get("is_finetune", False)

        self.model_params = merge_params_with_types(params, config.get_default_params())

        self.trained_model: Optional[Model] = None

        self.logger = get_logger(self.__class__.__name__, self.process_id)

    def train_model(self):
        self.logger.info("start training")
        self.logger.info(json.dumps(self.model_params))

        data = self.load_input_data(_Paths.train_dir)
        self.logger.info(f"data size: {data.shape[0]}")

        evaluator: Evaluator = self.config.get_evaluator(data)
        result: EvaluationResult = evaluator.run(data, self.model_params)
        result.model.save(_Paths.model_dir)

        ind = result.indicators
        score = ind[self.config.get_score_metric()]
        self.logger.info(SCORE_FORMAT.format(score))
        # We need to output the score metric in the indicators file:
        ind['score'] = score

        # Write the indicators file
        indicator_file = _Paths.model_dir / INDICATOR_FILE
        with indicator_file.open('w') as f:
            json.dump(ind, f, cls=DefaultEncoder, indent=2)
        # TODO: log indicators to stdout?
        for m in ind:
            self.logger.info(f'{m}={ind[m]};')

    def load_input_data(self, input_dir: Path) -> pd.DataFrame:
        input_files = [p for p in input_dir.iterdir() if p.is_file()]
        if len(input_files) == 0:
            raise ValueError(
                (
                    "There are no files in {}.\n"
                    "the data specification in S3 was incorrectly specified or\n"
                    "the role specified does not have permission to access the data."
                ).format(input_dir)
            )
        dfs = [pd.read_csv(file, header=0) for file in input_files]
        return pd.concat(dfs)

    def get_trained_model(self) -> Optional[Model]:
        if self.trained_model is None:
            try:
                self.trained_model = self.config.load_model(_Paths.model_dir)
            except Exception:
                return None

        return self.trained_model

    def create_flask_app(self, name):
        import flask

        # The flask app for serving predictions
        app = flask.Flask(name)

        @app.route("/ping", methods=["GET"])
        def ping():
            """
            Determine if the container is working and healthy.
            In this sample container, we declare
            it healthy if we can load the model successfully.
            """
            # You can insert a health check here
            health = self.get_trained_model() is not None

            status = 200 if health else 404
            return flask.Response(
                response="\n", status=status, mimetype="application/json"
            )

        @app.route("/invocations", methods=["POST"])
        def transformation():
            """
            Do an inference on a single batch of data.
            In this sample server, we take data as CSV,
            convert it to a pandas data frame for internal use and
            then convert the predictions back to CSV (which really
            just means one prediction per line, since there's a single column.
            """
            from io import StringIO

            data = None

            # Convert from CSV to pandas
            if flask.request.content_type == "text/csv":
                data = flask.request.data.decode("utf-8")
                s = StringIO(data)
                data = pd.read_csv(s, header=None)
            else:
                self.logger.error("input not csv data")
                return flask.Response(
                    response="This predictor only supports CSV data",
                    status=415,
                    mimetype="text/plain",
                )

            try:
                self.logger.info("Invoked with {} records".format(data.shape[0]))

                # Do the prediction
                prediction: Prediction = self.get_trained_model().predict(data)

                # Convert from numpy back to CSV
                out = StringIO()
                prediction.prediction.to_csv(out, header=False, index=False)
                result = out.getvalue()

                return flask.Response(response=result, status=200, mimetype="text/csv")
            except Exception as e:
                self.logger.error(str(e))
                return flask.Response(
                    response="Internal Server Error", status=500, mimetype="text/csv"
                )

        return app


def start_server():
    """
    Start nginx and gunicorn processes to serve the model prediction Flask app
    defined in wsgi.py in the current directory. This is expected to be in the
    working directory of the Docker container which also contains a cognimaker
    model implementation.
    """
    import multiprocessing
    import os
    import signal
    import subprocess
    import sys

    from pathlib import Path

    nginx_conf = Path.cwd() / 'nginx.conf'

    cpu_count = multiprocessing.cpu_count()

    model_server_timeout = int(os.environ.get("MODEL_SERVER_TIMEOUT", 60))
    model_server_workers = int(os.environ.get("MODEL_SERVER_WORKERS", cpu_count))

    def sigterm_handler(nginx_pid, gunicorn_pid):
        try:
            os.kill(nginx_pid, signal.SIGQUIT)
        except OSError:
            pass
        try:
            os.kill(gunicorn_pid, signal.SIGTERM)
        except OSError:
            pass

        sys.exit(0)

    # link the log streams to stdout/err so they will be logged to the container logs
    subprocess.check_call(["ln", "-sf", "/dev/stdout", "/var/log/nginx/access.log"])
    subprocess.check_call(["ln", "-sf", "/dev/stderr", "/var/log/nginx/error.log"])

    # nginx = subprocess.Popen(["nginx", "-c", "/opt/program/nginx.conf"])
    nginx = subprocess.Popen(["nginx", "-c", nginx_conf])
    gunicorn = subprocess.Popen(
        [
            "gunicorn",
            "--timeout", str(model_server_timeout),
            "-k", "gevent",
            "-b", "unix:/tmp/gunicorn.sock",
            "-w", str(model_server_workers),
            "--worker-tmp-dir", "/dev/shm",
            "wsgi:app",
        ]
    )

    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print("Inference server exiting")
