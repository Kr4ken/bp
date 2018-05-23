import json

class BPConfig(object):
    def __init__(self,
        epochs = 400,
        x_windows_size = 200,
        filter_cols = ["Close"],
        data_filepath='/home/kraken/projects/bitcoin_project/data/',
        source_filename='coinbaseUSD',
        start_date='10.01.2017',
        end_date='01.01.2018',
        batch_size=200,
        data_refresh = False,
        model_filepath='/home/kraken/projects/bitcoin_project/model/',
        model_refresh = False,
        config_filename = '/home/kraken/projects/bitcoin_project/code/config.json'):
        self.epochs = epochs
        self.x_windows_size = x_windows_size
        self.filter_cols = filter_cols
        self.data_filepath=data_filepath
        self.source_filename=source_filename
        self.start_date=start_date
        self.end_date=end_date
        self.batch_size=batch_size
        self.data_refresh = data_refresh
        self.model_filepath=model_filepath
        self.model_refresh = model_refresh
        self.config_filename = config_filename


    def save(self,filename = None):
        if(not filename):
            filename = self.config_filename
        file = open(filename,'w')
        file.write(json.dumps(self))
        file.close()

    def load(self,filename=None):
        if(not filename):
            filename = self.config_filename
        file = open(filename,'r')
        self  = json.load(file.read())
        file.close()
