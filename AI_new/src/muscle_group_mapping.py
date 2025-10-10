# muscle_group_mapping.py
# Central mapping for 7 muscle groups for use in muscle group classifier and data processing

MUSCLE_GROUPS_7 = {
    0: 0,  # barbell bicep curl -> biceps
    1: 1,  # bench press -> chest
    2: 1,  # chest fly machine -> chest
    3: 2,  # deadlift -> back
    4: 0,  # decline bench press -> biceps
    5: 0,  # hammer curl -> biceps
    6: 3,  # hip thrust -> legs
    7: 1,  # incline bench press -> chest
    8: 2,  # lat pulldown -> back
    9: 4,  # lateral raises -> shoulders
    10: 3, # leg extensions -> legs
    11: 5, # leg raises -> core
    12: 5, # plank -> core
    13: 2, # pull up -> back
    14: 1, # push ups -> chest
    15: 3, # romanian deadlift -> legs
    16: 5, # russian twist -> core
    17: 4, # shoulder press -> shoulders
    18: 3, # squat -> legs
    19: 2, # t bar row -> back
    20: 6, # tricep dips -> triceps
    21: 6, # tricep pushdown -> triceps
}

MUSCLE_GROUP_LABELS_7 = [
    "biceps",    # 0
    "chest",     # 1
    "back",      # 2
    "legs",      # 3
    "shoulders", # 4
    "core",      # 5
    "triceps"    # 6
]
