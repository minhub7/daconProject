# ConsumerDataAnalysis
> 소비자 데이터 기반 소비 예측 경진대회 (Dacon Basic)  
> 주최: Dacon  
> 주제: 소비자 데이터 기반 소비 예측 경진대회 (https://dacon.io/competitions/official/235893/overview/description)  
> 분야: NLP, 수치해석


## @Dataset 
- id : 샘플 아이디
- Year_Birth : 고객 생년월일
- Education : 고객 학력
- Marital_status : 고객 결혼 상태
- Income : 고객 연간 가구 소득
- Kidhome : 고객 가구의 자녀 수
- Teenhome : 고객 가구의 청소년 수
- Dt_Customer : 고객이 회사에 등록한 날짜
- Recency : 고객의 마지막 구매 이후 일수
- NumDealsPurchases : 할인된 구매 횟수
- NumWebPurchases : 회사 웹사이트를 통한 구매 건수
- NumCatalogPurchases : 카탈로그를 사용한 구매 수
- NumStorePuchases : 매장에서 직접 구매한 횟수
- NumWebVisitsMonth : 지난 달 회사 웹사이트 방문 횟수
- AcceptedCmp1: 고객이 첫 번째 캠페인에서 제안을 수락한 경우 1, 그렇지 않은 경우 0
- AcceptedCmp2: 고객이 두 번째 캠페인에서 제안을 수락한 경우 1, 그렇지 않은 경우 0
- AcceptedCmp3: 고객이 세 번째 캠페인에서 제안을 수락한 경우 1, 그렇지 않은 경우 0
- AcceptedCmp4: 고객이 네 번째 캠페인에서 제안을 수락한 경우 1, 그렇지 않은 경우 0
- AcceptedCmp5: 고객이 5번째 캠페인에서 제안을 수락한 경우 1, 그렇지 않은 경우 0
- Complain : 고객이 지난 2년 동안 불만을 제기한 경우 1, 그렇지 않은 경우 0
- Response : 고객이 마지막 캠페인에서 제안을 수락한 경우 1, 그렇지 않은 경우 0
- target : 고객의 제품 총 소비량


## @EDA
  |Train data describe|Categorical features|quantitative features|
  |-------------------|--------------------|---------------------|
|![image](https://user-images.githubusercontent.com/84756721/175192072-48bfadab-698c-485b-8e15-32c4a55ef230.png)|![image](https://user-images.githubusercontent.com/84756721/175192930-1b826aa8-1e84-4bc7-838a-1df5f5053f67.png)|![image](https://user-images.githubusercontent.com/84756721/175192945-019ca876-861d-4754-ad83-0eece4931118.png)|

  |Target 값 distribution|Year_Birth|Income distribution|
  |----------------------|----------|-------------------|
  |![image](https://user-images.githubusercontent.com/84756721/175192277-085b5f65-b6ba-4cfe-a921-fb7559aee78d.png)|![image](https://user-images.githubusercontent.com/84756721/175192912-f457c5c2-0092-48f3-a1fd-f8bf9cfd740b.png)|![image](https://user-images.githubusercontent.com/84756721/175193032-c49e6a1f-839e-49fc-bde9-1263d5b01ec9.png)|


## @Preprocessing
- **target 값 범위를 나타내는 range 특성 추가**
  - target 분포가 100~400 사이의 값이 많으므로 범위를 나누어 데이터를 추출해 고른 분포에서 학습할 수 있게 만듦.  
- **Year_Birth**
  - 1900년대 이전에 태어난 사람 존재 ➜ 이상치로 간주
  - 값을 제거하는 것이 아니라 빈도수가 가장 많은 값에 포함시켜 학습 시 변동이 적게 만듬.
- **Dt_customer**
  - 가입일을 year, month, day로 분리
  - year 값의 경우 target 값과 음의 상관관계가 크므로 유의미한 값을 가짐
- **Education, Marital_Status**
  - 학력 및 결혼 상태의 특성이 불필요하게 세분화되어있어 각각의 특성을 합쳐서 feature를 줄임
  - Education은 세 그룹, Marital_Status는 두 그룹으로 분류
- **Year_Birth를 Age로 변환**  
  - <code>train['Age'] = 2022 - train['Year_Birth']</code>
- **N번째 캠페인 참여 여부를 캠페인 참여 수로 통합**
  - <code>train["TotalAcceptedCmp"] = train["AcceptedCmp1"] + train["AcceptedCmp2"] + train["AcceptedCmp3"] + train["AcceptedCmp4"] + train["AcceptedCmp5"] + train["Response"]</code>
- **Purchases 횟수 통합**  
  - <code>train['TotalPurchases'] = train['NumCatalogPurchases']+train['NumStorePurchases']+train['NumWebPurchases']</code>
- **자녀 수 통합**  
  - <code>train['Dependents'] = train['Kidhome'] + train['Teenhome']</code>
- **전처리 후 필요없는 특성 제거**


## @Model
- RandomForestRegressor(n_estimators=500, max_depth=16, random_state=42, criterion='mae')
- XGBRegressor(n_estimators=200, max_depth=8, random_state=42)
- CatBoostRegressor(n_estimators=1200, max_depth=8, random_state=42, verbose=0)
- **GridSearch**를 통해 최적의 하이퍼파라미터를 찾아 규제를 걸어줌.  

  ### Hyperparameter for each model
  | |n_estimators|max_depth|random_state|criterion|verbose|
  |-|:----------:|:-------:|:----------:|:-------:|:-----:|  
  |**RandomForestRegressor**|500|16|42|'mae'|
  |**XGBRegressor**|200|8|42|||
  |**CatBoostRegressor**|1200|8|42||0|
  
  ### Validation
  - K-Fold와 Ensemble(Soft-Voting)을 통해 최종 inference를 진행
    

## @Result
|Model|Public Accuracy|Private Accuracy|
|-----|:-------------:|:--------------:|
|10-Fold<br>Soft-voting|0.16197|0.18788|


## @Update log
