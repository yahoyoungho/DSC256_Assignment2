"""Explicit, reviewable normalization of the supplied multilingual genre IDs.

These IDs belong to this CSV export, not Steam's public genre ID scheme.
Indie, commercial/status/content labels and software/video categories are
excluded from the nine-genre recommendation vocabulary.
"""

GENRE_IDS = {
    'Action': [6, 8, 13, 14, 22, 64, 80, 120, 122, 125, 133, 136, 142, 145, 154, 160],
    'Adventure': [1, 18, 26, 39, 42, 43, 54, 57, 61, 85, 94, 101, 108, 113, 128, 132, 150],
    'Casual': [29, 32, 34, 35, 52, 90, 92, 110, 127, 135],
    'Massively Multiplayer': [2, 16, 19, 25, 55, 58, 59, 148],
    'RPG': [40, 51, 65, 69, 100, 119, 124, 144],
    'Racing': [15, 63, 66, 73, 74, 121, 149],
    'Simulation': [10, 11, 17, 20, 37, 41, 67, 71, 87, 91, 99, 103, 109, 114, 137, 138, 141, 143, 147],
    'Sports': [60, 62, 77, 89, 97, 104, 111, 129, 140],
    'Strategy': [4, 5, 30, 46, 56, 70, 78, 84, 88, 102, 105, 112, 118, 139],
}

EXCLUDED_IDS = {
    'production category (Indie), not gameplay genre': [3, 7, 24, 31, 48, 72, 76, 96, 115, 126, 146],
    'business model or release status': [9, 21, 28, 36, 44, 75, 93, 95, 107, 117, 167],
    'content descriptor': [12, 49, 79, 106],
    'software, education or video category': [23, 27, 33, 38, 45, 47, 50, 53, 68, 81, 82, 83, 86, 98, 116, 123, 130, 131, 134, 151],
}

ID_TO_GENRE = {i: name for name, ids in GENRE_IDS.items() for i in ids}
ID_TO_EXCLUSION = {i: reason for reason, ids in EXCLUDED_IDS.items() for i in ids}
