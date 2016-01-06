from nltk.stem.lancaster import LancasterStemmer
import json
import logging
import math
import os
import string

class LM:
    def __init__(self, path_input, path_test, path_model = 'lm.json', path_output = 'output.p', flag_normalization = True, n = 2):
        self.d_norm = {}
        self.V = 0
        self.path_input = path_input
        self.path_model = path_model
        self.path_test = path_test
        self.path_output = path_output
        self.flag_normalization = flag_normalization
        self.n = int(n)

    def normalize(self, lst):
        '''
        For a list of words, normalize each word(stemming), and tokenize(split to N-gram)
        '''
        st = LancasterStemmer()
        lst_st = map(st.stem, lst)
        lst_del_punc = self.del_punc(lst_st)
        return lst_del_punc

    def f_del_punc(self, s):
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        return s
    
    def del_punc(self, lst):
        return map(self.f_del_punc, lst)

    def update_d_gram_from_lst(self, lst):
        if len(lst) >= self.n:
            # Update n Gram
            #logging.debug("lst: %s" % '\t'.join(lst))
            logging.debug("len(lst) - self.n: %d" % (len(lst) - self.n))
            for i in range(len(lst) - self.n):
                logging.debug("i: %d" % i)
                norm = '\t'.join(lst[i:i+self.n])
                logging.debug("norm: %s" % norm)
                if not self.d_norm.has_key(norm):
                    self.V += 1
                self.d_norm.setdefault(norm, 0)
                self.d_norm[norm] += 1
            # Update n-1 Gram
            for i in range(len(lst) - self.n + 1):
                norm = '\t'.join(lst[i:i+self.n - 1])
                self.d_norm.setdefault(norm, 0)
                self.d_norm[norm] += 1
    
    def save_model(self):
        with open(self.path_model, 'w') as outfile:
            json.dump(self.d_norm, outfile)
        n_result = len(self.d_norm.keys())
        file = open(self.path_model + '.txt', 'w')
        for key in self.d_norm.keys():
            file.write("%s : %d \n" % (key, self.d_norm[key]))
        file.close()
        logging.info("Successfully store %d norm counts to the file." % n_result)

    def predict_p_for_lst(self, lst):
        if len(lst) >= self.n:
            prob_log = 0
            for i in range(len(lst) - self.n):
                norm_n = '\t'.join(lst[i:i+self.n])
                norm_n_minus_1 = '\t'.join(lst[i:i+self.n - 1])
                cnt_norm_n = self.d_norm.get(norm_n, 0) + 1
                cnt_norm_n_minus_1 = self.d_norm.get(norm_n_minus_1, 0) + self.V
                try:
                    prob_log += math.log(cnt_norm_n * 1.0 / cnt_norm_n_minus_1)
                except:
                    print cnt_norm_n, cnt_norm_n_minus_1
            return prob_log
        else:
            return -1

    def build_model(self):
        logging.info("=> Build Model")
        if os.path.isdir(self.path_input):
            cnt_file = 0
            lst_path = os.listdir(self.path_input)
            cnt_file_total = len(lst_path)
            for path_r in lst_path:
                cnt_file += 1
                logging.info("Processing %d th file, totally %d files." % (cnt_file, cnt_file_total))
                path = self.path_input + '/' + path_r
                for line in open(path):
                    lst = line.strip('\n').split()
                    if self.flag_normalization:
                        lst = self.normalize(lst)
                    self.update_d_gram_from_lst(lst)
        else:
            for line in open(self.path_input):
                lst = line.strip('\n').split()
                if self.flag_normalization:
                    lst = self.normalize(lst)
                self.update_d_gram_from_lst(lst)
        self.save_model()
    
    def load_model(self, path_model):
        with open(path_model) as json_file:
            self.d_norm = json.load(json_file)
        self.V = 0
        for norm in self.d_norm.keys():
            leng = len(norm.split('\t'))
            if leng == self.n:
                self.V += 1
        print self.V
        #print self.d_norm
        #with open(path_model) as modelfile:
        #    json.load(modelfile, self.d_norm)

    def predict(self):
        file = open(self.path_output, 'w')
        for line in open(self.path_test):
            lst = line.strip('\n').split()
            if self.flag_normalization:
                lst = self.normalize(lst)
            prob_log = self.predict_p_for_lst(lst)
            file.write("%f\n" % prob_log)
        file.close()
            

            

