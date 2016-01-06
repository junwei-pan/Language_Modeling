import build_lm
import logging
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
#logging.basicConfig(filename='log/language_modeling.log',level=logging.INFO)
logging.basicConfig(level=logging.INFO)
#def __init__(self, path_input, path_model = 'lm.json', path_test, path_output = 'output.p', flag_normalization = True, n = 2):
#path_input = '/home/jwpan/Labs/Kaggle/Allen_AI_Science_Challenge_JunweiPan/data/validation_set.tsv'
#path_test = '/home/jwpan/Labs/Kaggle/Allen_AI_Science_Challenge_JunweiPan/data/validation_set.tsv'
path_input = 'data/wikipedia_content_based_on_ck_12_keyword_one_file_per_keyword'
#path_test = '/home/jwpan/Labs/Kaggle/Allen_AI_Science_Challenge_JunweiPan/data/validation_set.tsv'
path_test = 'data/data_for_lm_training.txt'
lm = build_lm.LM(path_input = path_input, path_test = path_test, path_model = 'model/lmA02.json', path_output = 'output_p/lmA03.p', n = 4)
lm.build_model()
#lm.load_model('model/lmA01.json')
lm.predict()
