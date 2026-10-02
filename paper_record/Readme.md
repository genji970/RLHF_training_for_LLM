Paper : "self rewarding language models"

EFT : evaluation fine tuning data
  - few chat prompting
  - sampling prompts

Training
  - seed, IFT, EFT
  - Ai self feedback
      - data 추가(augmented)
  - Instruction prompt
      - positive <- highest score response
      - negative <- lowest
M_0 -> M_1 -> M_2 -> M_3 -> M_2 -> M_3 -> ...
M_0 : Based pretrained LLM with no fine tuning
M_1 : Fine tuned on IFT + EFT seed data using SFT
M_2 : trained with AIFT data using DPO.
M_3 : .. 

Eval : ARCiEasy, ARC-Challenge , HellaWag, SiQA , PiQA,...

frozen reward model은 LLM training 도중 학습할 수 없다.
DPO든 reward model을 쓰든 human preference data의 size와 quality가 bottleneck의 원인일 수 있다. 

그래서 이 논문에서 제시하는 방안은,

-> prediction을 generate하는 llm model과 데이터에 reward를 메기는 reward model이 동일 Model이다. 
