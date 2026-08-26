from models import BOLDCast, TimeSeriesTransformer, DLinear, ForecastGrapher, FourierGNN, GPT4TS, iTransformer, LightTS, MSGNet, PatchTST, SimMTM, TSMixer


class Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'BOLDCast': BOLDCast,
            'BrainTransformer': TimeSeriesTransformer,
            'DLinear': DLinear,
            'ForecastGrapher': ForecastGrapher,
            'FourierGNN': FourierGNN,
            'One_Fit_All': GPT4TS,
            'iTransformer': iTransformer,
            'LightTS': LightTS,
            'MSGNet': MSGNet,
            'PatchTST': PatchTST,
            'SimMTM': SimMTM,
            'TSMixer': TSMixer
        }
        self.model= self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
