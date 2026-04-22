import os


class Config:
    DATA_FOLDER = r" "
    START_YEAR = #Start year
    END_YEAR = #End year
    MONTHS = [5, 6, 7, 8, 9]
    SAVE_FOLDER = r" "

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    RS_FEATURE_COLS = [
        'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04',
        'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'GNDVI', 'SIF', 'NIRv'
    ]
    M_FEATURE_COLS = ['Tmax', 'Pdsi', 'Pet', 'Pre']
    FEATURE_COLS = RS_FEATURE_COLS + M_FEATURE_COLS
    LABEL_COL = "Yield"

    SEED = #Randomly set
    NORMALIZE = True
    TEST_YEAR = #Randomly set

    USE_KAN_IN_PRIMARY_CAPS = True
    FASTKAN_USE_BASE_UPDATE = True