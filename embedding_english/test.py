from time import time
from wsim.wsim import wsimdict as wd

a = wd('../data/cmudict-0.7b-with-vitz-nonce')

# words = ['plant','screech','bricks','crowd','waves','harmed','limps','drink','smart','blamed','parks','grams','cramp','flushed','blond','pierced','stand','plots','split','prank','grant','plast','plint','plans','prant','slant','sit','wage','rule','keys','end','take','tall','dose','tass','toss','mess','imp','feet','chin','but','live','song','tin','this','miss','its','sat','set','sick','pit','hit','wonder','sickles','tossing','locket','raising','rapid','chinning','bucket','cradle','willows','member','colors','handle','butler','danger','widows','bundle','tender','windle','welder','winter','wuzder','wundle','wander','sunder','wunter']

s = time()
# r = a.top_similar('SIT', 10, wd.BIGRAM | wd.INSERT_BEG_END, 1)
r = a.top_similar('WONDER', 20, wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 1)
# r = a.similarity('WONDER', 'ASUNDER', wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 1)
# r = a.random_scores(5, wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 1)
# r = [a.get_index(s.upper()) for s in words]
print(f'time taken: {time() - s}')
print(r)