import os
import numpy as np
import pandas as pd

from src.data import RootDirectory, FloorPlan
from src.model_autoencoder import FingerprintModel

AUGMENT_DATA = True
TEST_SIZE = 0.25
BATCH_SIZE = 64
USE_WIFI_FEATURES = True
USE_IBEACON_FEATURES = True

if __name__ == '__main__':
    # get train/test dataset
    rd = RootDirectory()
    available_data = rd._AVAILABLE_DATA
    print(available_data)
    
    # for each site-floor, generate dataset with features
    # perform train-test split for dataset
    # train autoencoder
    available_data = {'site2' : ['F8']}
    results = []
    for site, floors in available_data.items():
        for floor in floors:
            site_num = int(site[-1])
            fp = FloorPlan(site_num, floor)
            data_train, data_test, (wifi_bssid_idx, ibeacon_uid_idx) \
                = fp.get_train_test_splits(test_size = TEST_SIZE,
                                                augment_data = AUGMENT_DATA)
            model_id = '_'.join([site, floor])
            model = FingerprintModel(device='cuda', model_id=model_id)  
            model.set_data(train=data_train, test=data_test, batch_size=BATCH_SIZE)
            model.initialize_model(len(wifi_bssid_idx), len(ibeacon_uid_idx), AUGMENT_DATA,
                                        USE_WIFI_FEATURES, USE_IBEACON_FEATURES)
            min_val_error, stop_epoch, loss_error = model.train(300, reduce_lr_epoch=15, startlr=0.01, verbose=False)
            #self.loss_error = [[], [], []] # training loss, training error, test  error
            loss_error_new = []
            for i in range(1, len(loss_error[0])+1):
                loss_error_new.append([i, loss_error[0][i-1], loss_error[1][i-1], loss_error[2][i-1]])
            df = pd.DataFrame(loss_error_new, columns=['epoch', 'training_loss', 'training_error', 'test_error'])
            df.to_csv(os.path.join(os.path.join(os.getcwd(), 'output'), '_'.join([site, floor, 'details.csv'])))
            results.append([site, floor, min_val_error, stop_epoch])
            model.plotloss()

    df = pd.DataFrame(results, columns = ['site', 'floor', 'min_val_error', 'stop_epoch'])    
    df.to_csv(os.path.join(os.path.join(os.getcwd(), 'output'), 'results.csv'))
    avg_err = np.mean([result[2] for result in results])
            
