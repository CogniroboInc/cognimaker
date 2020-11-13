from cognimaker.model.runner.sagemaker import SagemakerRunner
from __COGNIMAKER_MODEL_MODULE__ import __COGNIMAKER_MODEL_CLASS__ as Model

# Define the Flask app using SagemakerRunner.create_flask_app
app = SagemakerRunner(Model).create_flask_app(__name__)
