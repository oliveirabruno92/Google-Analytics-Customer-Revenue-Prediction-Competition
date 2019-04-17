import json


class Configurations:

    def __init__(self):
        self.config_file = 'configurations/google-competition-config.json'

    def get_params(self):

        print('Get parameters from configuration file..')
        with open(self.config_file) as json_file:
            configurations = json.load(json_file)

        config_params = {
            "flatten": configurations['app']['flatten'],
            "num_cols": configurations['model']['num_cols'],
            "no_use": configurations['model']['no_use'],
            "xgb_params": configurations['model']['xgb_params'],
            "n_folds": configurations['model']['n_folds'],
            "target": configurations['model']['target']
        }

        return config_params
