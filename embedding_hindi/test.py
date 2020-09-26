from time import time
from wsim.wsim import wsimdict as wd

a = wd('../res/mapping_hindi.txt', '../data/dict_hindi')

start_time = time()
# r = a.top_similar('SIT', 10, wd.BIGRAM | wd.INSERT_BEG_END, 1)
r = a.top_similar('समान', 20, wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 0.5)
# r = a.similarity('WONDER', 'ASUNDER', wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 1)
# r = a.random_scores(5, wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 0.4)
# r = a.get_word(1)
# r = a.get_index('उन्नाव')
# r = [a.get_index(s.upper()) for s in words]
print(f'time taken: {time() - start_time}')
print(r)
