import os
from logging import getLogger, StreamHandler, Formatter, INFO


def get_logger(name: str = None, process_id: str = None):
    logger = getLogger(name)
    logger.setLevel(INFO)
    cogni_maker_log_handler = CogniMakerLogHandler(process_id=process_id)
    logger.addHandler(cogni_maker_log_handler)
    return logger


class CogniMakerLogHandler(StreamHandler):

    def __init__(self, process_id=None):
        StreamHandler.__init__(self)
        self.setFormatter(self._get_formatter())
        if process_id is None:
            self.process_id = os.environ.get('PROCESS_ID', 'xxxxxxxx')
        else:
            self.process_id = process_id

    def _get_formatter(self):
        log_format = "%(asctime)s %(filename)s %(funcName)s "  \
                     "[%(levelname)s] %(process_id)s %(message)s"

        date_format = "%Y-%m-%dT%H:%M:%S%z"
        return Formatter(log_format, date_format)

    def emit(self, record):
        record.__dict__['process_id'] = self.process_id
        msg = self.format(record)
        print(msg)
