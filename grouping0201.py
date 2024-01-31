from gensim.models import Word2Vec
import numpy as np

# 텍스트 데이터
sentences = [
    ["apple", "banana", "orange", "grape"],
    ["dog", "cat", "rabbit", "hamster"],
    ["car", "bus", "train", "bike"]
]

# Word2Vec 모델 학습
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
word_vectors = model.wv

# 모든 단어 벡터를 가져와 배열로 변환
vectors = word_vectors.vectors

# 단어별 대표 벡터 계산 (각 단어 벡터의 평균)
representative_vectors = {}
for word in word_vectors.index_to_key:
    vectors_for_word = [word_vectors[word]]
    mean_vector = np.mean(vectors_for_word, axis=0)
    representative_vectors[word] = mean_vector

# 대표 벡터들 간의 거리 계산하여 비슷한 대표 벡터들을 그룹화
threshold = 0.1  # 비슷한 대표 벡터를 그룹화하기 위한 거리 임계값
groups = {}
for word, vector in representative_vectors.items():
    grouped = False
    for group, group_vector in groups.items():
        if np.linalg.norm(vector - group_vector) < threshold:
            groups[group].append(word)
            grouped = True
            break
    if not grouped:
        groups[word] = [word]

# 그룹에 속한 단어들 출력
for i, (group, words) in enumerate(groups.items()):
    print(f"그룹 {i + 1}: {words}")