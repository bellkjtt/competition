from konlpy.tag import Okt

okt = Okt()

text = '안녕하세요. 오래간만이네요~~. 어제 재미있었어요.'

print(okt.nouns(text))
