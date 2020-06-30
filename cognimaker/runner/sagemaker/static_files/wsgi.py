from cognimaker.runner.sagemaker import SagemakerRunner
from __COGNIMAKER_MODEL_MODULE__ import config  # import ModelConfig object

app = SagemakerRunner(config).create_flask_app('__COGNIMAKER_MODEL_MODULE__')
