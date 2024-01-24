# 텍스트 데이터
texts = [
    "물리학", "화학", "생물학", "실험",
    "코딩", "프로그래밍", "알고리즘", "개발",
    "미술", "음악", "창작", "예술",
]

# 각 단어의 길이를 계산하여 그룹 형성
groups = {}
for text in texts:
    length = len(text)
    if length not in groups:
        groups[length] = [text]
    else:
        groups[length].append(text)

# 그룹 출력
for key, values in groups.items():
    print(f"글자 수 {key}: {values}")
