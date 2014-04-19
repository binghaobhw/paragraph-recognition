# coding:utf-8
LOG_HOME = 'log'
LOG_PROJECT_NAME = 'paragraph-recognition'
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s %(process)d %('
                      'threadName)s %(funcName)s: %(message)s'
        }
    },
    'handlers': {
        'info': {
            'level': 'DEBUG',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'verbose',
            'filename': '%s/info.log' % LOG_HOME,
            'when': 'midnight',
            'interval': 1
        },
        'other': {
            'level': 'WARNING',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'verbose',
            'filename': '%s/other.log' % LOG_HOME,
            'when': 'midnight',
            'interval': 1
        }
    },
    'loggers': {
        LOG_PROJECT_NAME: {
            'handlers': ['info'],
            'propagate': False,
            'level': 'DEBUG'
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['other']
    }
}
