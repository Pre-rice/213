import pandas as pd
import numpy as np
from utils import *
from MyModel import MyModel

def run_test():
    #########Load Model
    model = MyModel()

    #########Load Day Data
    days = get_day_folders("./data")

    #########Online Predict
    for day in days:
        day_data = load_day_data("./data", day)
        n_ticks = len(day_data['E'])

        ticktimes = day_data['E'].values.T[0, :]
        my_preds = np.zeros((n_ticks))

        for tick_index in range(n_ticks):
            ###########Get Tick Data(E and Sector)
            E_row_data = day_data['E'].iloc[tick_index]
            sector_row_datas = [
                day_data['A'].iloc[tick_index],
                day_data['B'].iloc[tick_index],
                day_data['C'].iloc[tick_index],
                day_data['D'].iloc[tick_index]
            ]

            ###########Predict
            my_preds[tick_index] = model.online_predict(E_row_data, sector_row_datas)

        ###########Save Data
        if os.path.exists("./output/"+day) is not True:
            os.makedirs("./output/"+day)
        out_frame = pd.DataFrame(np.vstack(([ticktimes, my_preds])).T)
        columns = ['Time', 'Predict']
        out_frame.columns = columns
        out_frame.to_csv("./output/"+day+"/E.csv", index=False)
        print ("Submit Day", day)

    
if __name__ == '__main__':
    run_test()
