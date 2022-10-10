import numpy as np

from src.data import RootDirectory, FloorPlan
from src.model import FingerprintModel

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
    available_data = {'site1' : ['B1']}
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
            min_val_error, stop_epoch = model.train(300, reduce_lr_epoch=15, startlr=0.01, verbose=False)
            results.append([site, floor, min_val_error, stop_epoch])
            model.plotloss()
    
    avg_err = np.mean([result[2] for result in results])
            