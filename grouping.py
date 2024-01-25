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
X = vectorizer.fit_transform(texts) # fit_transform 메서드를 사용하여 텍스트 데이터를 벡터로 변환

# K-means 클러스터링을 사용하여 데이터 그룹화
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 클러스터에 속한 텍스트들 출력
for i in range(3):
    cluster_text_indices = np.where(kmeans.labels_ == i)[0]
    cluster_texts = [texts[idx] for idx in cluster_text_indices] # 추출된 인덱스를 사용하여 원본 텍스트 데이터에서 해당 클러스터에 속한 텍스트를 추출. cluster_texts 리스트에 저장
    print(f"그룹 {i + 1}: {cluster_texts}")


# KMeans: 주어진 데이터를 KMeans 클러스터링 알고리즘을 사용하여 그룹화. 클러스터의 개수는 n_clusters 매개변수로 설정.
# NumPy (np): 수치 계산을 위한 파이썬 라이브러리. 여기서는 NumPy의 배열을 사용하여 데이터를 처리하고 클러스터에 속한 텍스트를 출력.

# CountVectorizer: 주어진 텍스트 데이터를 단어 빈도로 벡터화하는데 사용되는 도구. 각 문서를 단어의 등장 빈도로 표현.
# fit_transform 메서드는 CountVectorizer 객체에 데이터를 적합화(fit)하고, 동시에 변환(transform)하여 텍스트 데이터를 단어 등장 빈도의 행렬로 변환합니다.
# n_clusters: 클러스터(그룹화 시킬 주제) 개수
# random_state: 머신러닝 모델에서 사용되는 난수 생성 시드(Seed)를 설정하는 매개변수
# => random_state를 설정하면 초기 클러스터 중심을 결정하는데 사용되는 난수의 시드를 제어할 수 있습니다. 이렇게 하면 동일한 초기 중심을 얻어, 모델을 동일한 조건에서 여러 번 실행할 때 결과가 일관되게 나올 수 있다.
# 난수: 예측할 수 없는 값으로, 무작위성을 나타내는 수
# 난수 생성: 많은 머신러닝 알고리즘에서 초기화, 데이터 분할, 가중치 초기화 등에서 사용
# fit 메서드를 사용하여 K-means 모델을 입력 데이터 X에 적합화. 여기서 X는 앞서 CountVectorizer를 통해 변환된 문서-단어 행렬. K-means 알고리즘은 이 데이터를 기반으로 클러스터링을 수행하고, 각 문서를 적절한 클러스터에 할당.
# kmeans.labels_는 각 데이터 포인트가 어떤 클러스터에 속하는지를 나타내는 배열이며, 여기서 i는 현재 반복 중인 클러스터의 인덱스입니다. np.where(kmeans.labels_ == i)[0]는 클러스터 i에 속한 데이터 포인트의 인덱스를 추출