# scikit-learn에서 제공하는 텍스트 데이터를 벡터화하는 도구
from sklearn.feature_extraction.text import CountVectorizer

# scikit-learn에서 제공하는 KMeans 클러스터링 알고리즘
from sklearn.cluster import KMeans

# NumPy는 수치 계산을 위한 파이썬 라이브러리
import numpy as np

# 텍스트 데이터
texts = [
    "물리학", "화학", "생물학", "실험",
    "코딩", "프로그래밍", "알고리즘", "개발",
    "미술", "음악", "창작", "예술",
]

# CountVectorizer를 사용하여 텍스트 데이터를 단어 빈도로 벡터화
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# K-means 클러스터링을 사용하여 데이터 그룹화
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 클러스터에 속한 텍스트들 출력
for i in range(3):
    cluster_text_indices = np.where(kmeans.labels_ == i)[0]
    cluster_texts = [texts[idx] for idx in cluster_text_indices]
    print(f"그룹 {i + 1}: {cluster_texts}")


# CountVectorizer: 주어진 텍스트 데이터를 단어 빈도로 벡터화하는데 사용되는 도구. 각 문서를 단어의 등장 빈도로 표현.
# KMeans: 주어진 데이터를 KMeans 클러스터링 알고리즘을 사용하여 그룹화. 클러스터의 개수는 n_clusters 매개변수로 설정.
# NumPy (np): 수치 계산을 위한 파이썬 라이브러리. 여기서는 NumPy의 배열을 사용하여 데이터를 처리하고 클러스터에 속한 텍스트를 출력.