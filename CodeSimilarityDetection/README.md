# CodeSimilarityDetection
> 코드 유사성 판단 AI 경진대회 (Monthly Dacon 21)

## @Subject
- 주최: Dacon
- 주제: 코드 유사성 판단 AI 경진대회 (https://dacon.io/competitions/official/235900/overview/description)
- Dataset
  - sample_train.csv: 17,970개의 python code 쌍
  - test.csv: 179,700개의 python code 쌍
  - code: 추가로 제공되는 sample code

## @Update log
- 2022.05.23 (mon) Baseline source code update - Using CountVectorizer and CosineSimilarity
- 2022.05.25 (wed) Update ver.0.1 source code - Using Graphcodebert, negative sampling
- 2022.06.01 (wed) Add .ipynb file that create data set for each level
  1. Random pairs data
  2. Middle similar positive, negative pairs
  3. Low similar positive pairs, high similar negative pairs
- 2022.06.02 (thu) Update create data set file and EDA, Add train file
  - Upgrade preprocessing - 다중 주석에 대한 전처리 추가, 필요없는 url (https...) 삭제
- 2022.06.07 (tue) Update create dataset file, add file (graphcodebert by using k-fold, ensemble)
  - Level 2, Level 3 데이터에 대한 성능이 기대보다 저조해 사용하지 않는 것으로 수정
  - microsoft/graphcodebert-base 모델을 사용하여 k-fold 방식으로 학습 및 검증, 최종 예측은 ensemble을 통해 추론
