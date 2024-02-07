# 1. **라이브러리 임포트**:
    
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from gensim.models import Word2Vec
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    
#     - 필요한 라이브러리를 임포트합니다.
#     - **`TSNE`**는 t-SNE 시각화를 위해, **`matplotlib.pyplot`**는 그래프를 그리기 위해, **`Word2Vec`**은 단어 벡터를 학습하기 위해, **`KMeans`**는 클러스터링을 위해, **`CountVectorizer`**는 단어를 벡터로 변환하기 위해 사용됩니다.
# 2. **텍스트 데이터 준비**:
    
    texts = [
        "computer", "mouse", "coding",
        "korea", "usa", "china",
        "music", "singer", "dance",
    ]
    
#     - 단어를 포함한 텍스트 데이터를 리스트에 저장합니다.
# 3. **Word2Vec 모델 학습**:
    
    model = Word2Vec([text.split() for text in texts], vector_size=100, window=5, min_count=1, workers=4)
    
#     - 입력 텍스트를 단어 단위로 분할하고, Word2Vec 모델을 학습합니다.
#     - **`vector_size`**는 단어 벡터의 차원 수를 지정하고, **`window`**는 주변 단어의 범위를 나타냅니다.
#     - **`min_count`**는 최소 단어 등장 횟수를 의미하며, 이 값보다 적게 등장한 단어는 무시됩니다.
#     - **`workers`**는 학습에 사용되는 CPU 코어의 수를 의미합니다.
# 4. **K-means 클러스터링**:
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(word_vectors)
    
#     - 학습된 단어 벡터를 기반으로 K-means 클러스터링을 수행합니다.
#     - **`n_clusters`**는 클러스터의 개수를 지정하며, 여기서는 3으로 설정되었습니다.
#     - **`random_state`**는 재현 가능한 결과를 얻기 위한 랜덤 시드 값입니다.
#     - **`kmeans.fit(word_vectors)`**는 KMeans 클러스터링 알고리즘을 사용하여 주어진 단어 벡터 데이터를 클러스터링하는 과정을 수행하는 부분입니다.
# 5. **t-SNE를 사용한 시각화**:
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=1)
    word_vectors_tsne = tsne.fit_transform(word_vectors)
    
#     - 학습된 단어 벡터를 2차원으로 축소하기 위해 t-SNE를 사용합니다.
#     - **`n_components`**는 축소된 차원의 수를 의미하며, 여기서는 2로 설정되었습니다.
#     - **`perplexity`**는 t-SNE 알고리즘의 하이퍼파라미터로, 값이 작을수록 군집이 더 세분화됩니다.
#     - **`tsne.fit_transform(word_vectors)`**은 t-SNE (t-distributed Stochastic Neighbor Embedding) 알고리즘을 사용하여 고차원의 단어 벡터를 2차원 공간으로 차원 축소하는 과정을 수행하는 부분입니다.
# 6. **시각화 및 클러스터 결과 출력**:
    
    for i in range(len(texts)):
        plt.scatter(word_vectors_tsne[i, 0], word_vectors_tsne[i, 1], c=kmeans.labels_[i], cmap='viridis')
        plt.text(word_vectors_tsne[i, 0], word_vectors_tsne[i, 1], texts[i])
    
        if i == len(texts) - 1:
            for cluster_num in range(max(kmeans.labels_) + 1):
                cluster_indices = np.where(kmeans.labels_ == cluster_num)[0]
                cluster_words = [texts[idx] for idx in cluster_indices]
                print(f"클러스터 {cluster_num + 1}: {cluster_words}")
    
    # - t-SNE를 통해 축소된 단어 벡터를 산점도로 시각화합니다.
    # - 각 데이터 포인트에 해당하는 단어를 텍스트로 표시합니다.
    # - 각 클러스터에 속한 단어들을 출력합니다.
    
    # - 2차원으로 차원 축소된 단어 벡터를 시각화하는 과정과 함께 각 클러스터에 속한 단어들을 출력하는 부분입니다.
    # 1. **`for i in range(len(texts)):`**: 단어의 개수만큼 반복하며 각 단어에 대한 산점도를 그리고 해당 단어를 텍스트로 표시합니다.
    # 2. **`plt.scatter(word_vectors_tsne[i, 0], word_vectors_tsne[i, 1], c=kmeans.labels_[i], cmap='viridis')`**: t-SNE를 통해 2차원으로 축소된 단어 벡터를 산점도로 시각화합니다. 각 점의 색상은 해당 단어가 속한 K-means 클러스터를 나타냅니다.
    # 3. **`plt.text(word_vectors_tsne[i, 0], word_vectors_tsne[i, 1], texts[i])`**: 각 점에 해당하는 단어를 텍스트로 표시합니다.
    # 4. **`if i == len(texts) - 1:`**: 모든 단어를 처리한 이후에 각 클러스터에 속한 단어들을 출력하는 부분입니다.
    # 5. **`for cluster_num in range(max(kmeans.labels_) + 1):`**: 클러스터 번호를 반복합니다.
    # 6. **`cluster_indices = np.where(kmeans.labels_ == cluster_num)[0]`**: 현재 클러스터에 속한 단어의 인덱스를 가져옵니다.
    # 7. **`cluster_words = [texts[idx] for idx in cluster_indices]`**: 해당 클러스터에 속한 단어들을 가져옵니다.
    # 8. **`print(f"클러스터 {cluster_num + 1}: {cluster_words}")`**: 현재 클러스터에 속한 단어들을 출력합니다.
# 7. **그래프 출력**:
    
    plt.title('Word Embeddings Clustering')
    plt.show()
    
    # - 시각화된 산점도 그래프를 출력합니다.