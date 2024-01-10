# 필요한 라이브러리를 가져옵니다.
import requests
import re
from bs4 import BeautifulSoup

# HTTP 요청을 위한 사용자 에이전트 및 언어 헤더를 설정
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36", "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3"}

# 1부터 5까지의 페이지를 반복
for i in range(1, 6):
    # print("페이지 :", i)

    # 각 페이지에 대한 URL을 구성
    url = "https://www.coupang.com/np/search?q=%EB%85%B8%ED%8A%B8%EB%B6%81&channel=user&component=&eventCategory=SRP&trcid=&traid=&sorter=scoreDesc&minPrice=&maxPrice=&priceRange=&filterType=&listSize=36&filter=&isPriceRange=false&brand=&offerCondition=&rating=0&page={}&rocketAll=false&searchIndexingToken=1=6&backgroundColor=".format(i)

    # URL로 GET 요청을 보냄
    res = requests.get(url, headers=headers)
    res.raise_for_status() # 응답이 실패할 경우 예외를 발생.

    # BeautifulSoup을 사용하여 페이지의 HTML 내용을 파싱
    soup = BeautifulSoup(res.text, "lxml")

    # BeautifulSoup의 find_all 메서드를 사용하여 페이지의 모든 제품 항목을 찾습니다.
    items = soup.find_all("li", attrs={"class":re.compile("^search-product")})
    # print(items[0].find("div", attrs={"class":"name"}).get_text())

    for item in items:  # 각 제품 항목을 반복

        # 제품이 광고로 표시되어 있는지 확인하고 제외
        ad_badge = item.find("span", attrs={"class":"ad-badge-text"})
        if ad_badge:
            # print("  <광고 상품 제외합니다>")
            continue

        # 제품 정보를 추출 (제품명, 가격, 평점, 평점 수, 링크).
        name = item.find("div", attrs={"class":"name"}).get_text() # 제품명

        # 애플 제품 제외
        # if "Apple" in name:
        #     #print("  <Apple 상품 제외합니다")
        #     continue

        price = item.find("strong", attrs={"class":"price-value"}).get_text() # 가격

        # 리뷰 100개 이상, 평점 4.5 이상 되는 것만 조회
        rate = item.find("em", attrs={"class":"rating"}) # 평점
        if rate:
            rate = rate.get_text()
        else:
            # rate = "평점 없음"
            # print("  <평점 없는 상품 제외합니다>")
            continue

        rate_cnt = item.find("span", attrs={"class":"rating-total-count"}) # 평점 수
        if rate_cnt:
            rate_cnt = rate_cnt.get_text()[1:-1] # 예 : (26), 괄호 없애기

        else:
            # rate_cnt = "평점 수 없음"
            # print("  <평점 수 없는 상품 제외합니다>")
            continue

        link = item.find("a", attrs={"class":"search-product-link"})["href"] # 링크

        # 평점이 4.5 이상이고 평점 수가 100 이상인 제품만 필터링
        if float(rate) >= 4.5 and int(rate_cnt) >= 100:
            # print(name, price, rate, rate_cnt)
            print(f"제품명 : {name}")
            print(f"가격 : {price}")
            print(f"평점 : {rate}점 ({rate_cnt})개")
            print("바로가기 : {}".format("https://www.coupang.com/"+link))
            print("-"*100)