LABEL_COLUMN = "isFraud"

NUM_CLIENTS = 5

NUMERICAL_FEATURES = [
    "TransactionAmt",
    "addr1", "addr2",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "C8", "C9", "C10", "C11",
    "D1", "D2", "D3", "D4", "D5",
    "D10", "D11", "D15",
]

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M4", "M5", "M6",
    "DeviceType",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES