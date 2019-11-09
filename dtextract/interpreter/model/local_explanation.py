import numpy as np
import pandas as pd


class LocalExplanation:

    def __init__(self, data, output, rf_output, contributions):
        self.data = data
        self.output = output
        self.rf_output = rf_output
        self.contributions = contributions
        self.contrib_df = pd.DataFrame(contributions[:, 1:].astype('float'), index=contributions[:, 0])

    def get_description(self, k, non_zero_features=False, headers=None):
        if non_zero_features:
            assert len(headers) == len(self.data)
            w = np.where(self.data != 0)[0]
            data = np.column_stack((headers[w], self.data[w]))
            input = '\n\nInput:\n'
            for i in range(len(data)):
                input += str(data[i]) + '\n'
        else:
            input = 'Input:\n' + str(self.data)
        return input + 'Output:\n' + str(self.output) \
               + '\nRF Output:\n' + str(self.rf_output) + '\n' \
               + str(k) + ' important contributions:\n' + str(self.contributions[:k, :])

    def __str__(self):
        return self.get_description(5)
