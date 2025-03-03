import datasets
from collections import Counter
from gensim.utils import tokenize
import string
import json



newsdata=datasets.load_dataset('ag_news')
#Создаём словарь количества вхождений слов
words=['<unk>','<bos>','<eos>','<pad>']
#str.maketrans('','',string.punctuation)-возврящяет словарь для замены
#.translate ждёт словарь слов для замены
for sent in newsdata['train']['text']:
    proced_sent=sent.lower().translate(
        str.maketrans('','',string.punctuation)
    )
    for word in tokenize(proced_sent):
        words.append(word)
vocab = set(['<unk>','<bos>','<eos>','<pad>'])#Токеный неизвестногого слова, начала,конца последовательности и токен пустного пропуска для батчей
treshold=25 #Порог для включения в словарь word2ind
words=Counter(words)
for word,cnt in words.items():
    if cnt>treshold:
        vocab.add(word)
print(f'Размер словаря {len(vocab)}')

word2ind={i:char for char,i in enumerate(vocab)}
ind2word={char:i for char,i in enumerate(vocab)}

splited={}
splited['word2ind']=word2ind
splited['ind2word']=ind2word


with open ('words_dict.json','w') as out:
    json.dump(splited,out)



   