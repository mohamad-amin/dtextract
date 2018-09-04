from consts import *

# Circle constants
CIRCLE_PATH = TMP_PATH + "/circle.csv"
CIRCLE_HAS_HEADER = False
CIRCLE_DATA_TYPES = [NUM, NUM] + [CAT_RES]

CIRCLE_IS_CLASSIFY = True
CIRCLE_N_DATA_MATRIX_COLS = 3 # TODO: check this

CIRCLE_OUTPUT = 'circle.log'

# Checkerboard constants
CHECKERBOARD_PATH = TMP_PATH + "/checkerboard.csv"
CHECKERBOARD_HAS_HEADER = False
CHECKERBOARD_DATA_TYPES = [NUM, NUM] + [CAT_RES]

CHECKERBOARD_IS_CLASSIFY = True
CHECKERBOARD_N_DATA_MATRIX_COLS = 3 # TODO: check this

CHECKERBOARD_OUTPUT = 'checkerboard.log'

# Short aclimdb constants
ACLIMDB_PATH = TMP_PATH + '/aclimdb_short.csv'
ACLIMDB_HAS_HEADER = False
ACLIMDB_DATA_TYPES = [ID] + [NUM] * 28349 + [CAT_RES]

ACLIMDB_IS_CLASSIFY = True
ACLIMDB_N_DATA_MATRIX_COLS = 28350 # TODO: check this

ACLIMDB_OUTPUT = 'aclimdb.log'

# Youtube Shakira's video's comments spam classification
YOUTUBE_SHAKIRA_PATH = TMP_PATH + '/new_youtube_shakira.csv'
YOUTUBE_SHAKIRA_HAS_HEADER = False
YOUTUBE_SHAKIRA_DATA_TYPES = [ID] * 2 + [NUM] * 1195 + [CAT_RES]

YOUTUBE_SHAKIRA_IS_CLASSIFY = True
YOUTUBE_SHAKIRA_N_DATA_MATRIX_COLS = len(YOUTUBE_SHAKIRA_DATA_TYPES) - 1 # TODO: check this

YOUTUBE_SHAKIRA_OUTPUT = 'youtube_shakira.log'

# Simple 50*50 samples' face detection
FACES_PATH = TMP_PATH + '/faces_50_50.csv'
FACES_HAS_HEADER = True
FACES_DATA_TYPES = [NUM] * 2500 + [CAT_RES]

FACES_IS_CLASSIFY = True
FACES_N_DATA_MATRIX_COLS = 2501 # TODO: check this

FACES_OUTPUT = 'faces.log'

# Simple 50*50 samples' face detection
SMS_SPAM_PATH = TMP_PATH + '/sms_spams.csv'
SMS_SPAM_HAS_HEADER = True
SMS_SPAM_DATA_TYPES = [NUM] * 8444 + [CAT_RES]

SMS_SPAM_IS_CLASSIFY = True
SMS_SPAM_N_DATA_MATRIX_COLS = 8446 # TODO: check this

SMS_SPAM_OUTPUT = 'sms_spam.log'

# Test
TEST_PATH = TMP_PATH + '/test.csv'  # Path to the input dataset csv
TEST_HAS_HEADER = False  # If the first row is a header row
TEST_DATA_TYPES = [ID] + [NUM] * 10 + [CAT_RES]  # First column is id (not important), 10 numeric feature columns, categorical label column

TEST_IS_CLASSIFY = True  # If it's a classification problem

TEST_OUTPUT = 'test.log'  # The output file to write log of running code
