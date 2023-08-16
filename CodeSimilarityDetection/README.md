# CodeSimilarityDetection
> 코드 유사성 판단 AI 경진대회 (Monthly Dacon 21)

<br />
<a href="RELEASE.md"><strong>Explore the release notes »</strong></a>
<br />

## @Subject
- 주최: Dacon
- 주제: 코드 유사성 판단 AI 경진대회 ([from here!](https://dacon.io/competitions/official/235900/overview/description))
- 분야: Code NLP
- Dataset
  - sample_train.csv: 17,970개의 python code 쌍
  - test.csv: 179,700개의 python code 쌍
  - code: 추가로 제공되는 sample code


## @Preprocessing
- '#' ➜ '' : 주석 제거
- '\n' ➜ '' : 여러 번의 개행 문자 삭제
- '    ' ➜ '\t' : 다중 공백을 tab 문자로 변환
- """ """, ''' ''' : 다중 문자열로 주석처리된 구문 제거
- 'https/http...' : 코드 분석과 상관없는 url 제거


## @Make personal data set
- 제공되는 300개의 문제에 대한 sample code를 활용하여 dataset 생성
- 같은 문제를 푸는 **Positive pairs**와 다른 문제를 푸는 **Negative pairs**를 random 추출하여 구성
  - 총 600,000개의 train_data set으로 학습
  - 총 60,000개의 valid_data set으로 검증


## @Model
- microsoft 사에서 개발한 **graphcodebert-base** 모델 사용
- huggingface의 datasets, transformers 라이브러리 활용하여 코드 토큰화 및 학습 진행

  ### Hyperparameter for TrainingArguments
  - **CodeSimilarityDetection_Graphcodebert_Kfold_Ensemble.ipynb**
    |**parameter**|**value**|  
    |-------------|:---------:|  
    |model|microsoft/graphcodebert-base|
    |per_device_train_batch_size|4|  
    |per_device_eval_batch_size|32|  
    |gradient_accumulation_steps|4|  
    |warmup_steps|250|
    |eval_steps|500|
    |learning_rate|2e-5|  
    |optim|'adamw_torch'|
    |early_stopping_patience|15|
    
  - **graphcodebert5folds1epoch.ipynb for NH**
    |**parameter**|**value**|  
    |-------------|:---------:|  
    |model|microsoft/graphcodebert-base|
    |optim|'AdamW'|
    |scheduler|'linear'|
    |warmup_steps|500|
    |cycle_mult|1.2|
    |batch_size|16|  
    |patience|5|
    |learning_rate|2e-5|  
    |criterion|'cross'|

## @Result
|Model|Public Accuracy|Private Accuracy|
|---|:---:|:---:|
|graphcodebert|0.94057||
|K-fold and Ensemble|0.96463||
|5-fold and hard-voting for NH|0.9795|0.97946|
