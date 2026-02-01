import numpy as np
import pandas as pd

########My Model
class MyModel:
    def __init__(self):
        pass

    def reset(self):
        pass


    def online_predict(self, E_row, sector_rows):
        """
        Predict stock E's future 5-min return. [FIXED]

        Args:
            E_row: dict, stock E data (no 'Return5min' key)
            sector_rows: list[dict], [A_row, B_row, C_row, D_row]

        Returns:
            float: predicted return
    	"""
        [A_row, B_row, C_row, D_row] = sector_rows
        ##pre[row] = E_row['AskVolume1'] + E_row['BidVolume1']
        pred = E_row['AskVolume1'] / (A_row['AskVolume1'] + B_row['AskVolume1'] + C_row['AskVolume1'] + D_row['AskVolume1'])
        return pred


    def save_data(self):
        pass
