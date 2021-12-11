# EmotractionServer
스마일게이트 제안 프로젝트 '감성대화챗봇'을 기반으로 하는 프로젝트의 Flask 기반 모델 Api.   
#### 기본 목표 
음성대화로 부터 기쁨, 슬픔, 화남과 같은 감정을 인식하는 기술을 개발   
#### 확장 목표 
감정이 시각화 된 1대1 채팅 서비스의 개발   
<br/><br/>

## 접근1: <한국어 데이터를 통한 학습>
#### 1) 7개의 감정으로 분류하는 모델 - TestEmotion.py
감정 : 공포 / 놀람 / 분노 / 슬픔 / 중립 / 행복 / 혐오  
학습 코퍼스로 사용한 Aihub의 데이터 자체가 제대로 라벨링되어 있지 않다는 문제 -> 다소 낮은 정확도  



#### 2) 3개의 감정으로 분류하는 모델 - NewEmotion.py
- 부정 : 공포 / 놀람 / 분노 / 슬픔 / 혐오
- 중립
- 긍정 : 행복   

위의 3개의 감정으로 데이터를 다시 라벨링하고 네이버 영화 평점 데이터 3만개를 추가해 학습을 진행하여 정확도 향상

그러나 코퍼스가 부족하다는 문제가 여전히 존재.

<I>모델 학습 결과 모델은 크기 문제로 업로드 불가<I/>
<br/><br/>
## 접근2 : <영어 데이터를 통한 학습> - server.py
구글의 emotion 관련 영어데이터인 goEmotions로 학습된 모델을 이용.  
[monologg/GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch)  
위의 깃허브에서 학습된 모델 이용.  

iOS 클라이언트에서 영어로 데이터를 전달받고 이에서 감정을 추출하여 반환하는 방식 선택.
- 27가지 감정을 확인 가능한 모델
- 긍정, 부정, 중립을 확인 가능한 모델
- 7가지 감정을 확인 가능한 모델

#### <사용 방법>
1. 위의 깃허브를 clone
```text
git clone https://github.com/monologg/GoEmotions-pytorch.git
```
3. 가상환경을 만들어 실행 
```text
virtualenv [가상환경 이름]
cd [가상환경 이름]
source bin/activate
cd ..
```
4. clone한 레파지토리에 필요한 것 설치 (requirements.txt는 clone한 파일에 함께 있음)
```text
pip3 install -r requirements.txt
```
5. Flask 설치
```text
pip3 install Flask
```
6. server.py 파일을 폴더에 추가 후 실행
```text
python server.py
```
<br/><br/>
url은 아래와 같음
- 27가지 : /original
- 긍정/부정/중립 : /group
- 7가지 : /ekman
