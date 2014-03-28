# coding:utf-8
LOG_HOME = 'log'
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s %(process)d %('
                      'threadName)s %(message)s'
        }
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'verbose',
            'filename': '%s/extractor.log' % LOG_HOME,
            'when': 'midnight',
            'interval': 1
        },
        'error_file': {
            'level': 'WARNING',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'verbose',
            'filename': '%s/error.log' % LOG_HOME,
            'when': 'midnight',
            'interval': 1
        }
    },
    'loggers': {
        'extractor': {
            'handlers': ['file', 'error_file'],
            'propagate': False,
            'level': 'INFO'
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['error_file']
    }
}
