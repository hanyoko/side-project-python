from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# 텍스트 데이터
texts = [
    "과일",
    "딸기",
    "수박",
    "연필",
    "볼펜",
    "공책",
    "컴퓨터",
    "노트북",
    "마우스",
]

# TfidfVectorizer를 사용하여 텍스트 데이터를 벡터화
vectorizer = TfidfVectorizer(use_idf=False, norm='l1')  # l1 정규화를 사용하여 단어의 존재 여부만을 고려
X = vectorizer.fit_transform(texts)

# K-means 클러스터링을 사용하여 데이터 그룹화
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 클러스터에 속한 텍스트들 출력
for i in range(3):
    cluster_text_indices = np.where(kmeans.labels_ == i)[0]
    cluster_texts = [texts[idx] for idx in cluster_text_indices]
    print(f"Cluster {i + 1}: {cluster_texts}")