# OpenFlamingo 이미지 질문-답변 데모

이 프로젝트는 OpenFlamingo 모델을 사용하여 이미지에 대해 질문하고 답변을 받는 예제 코드들을 제공합니다.

## 파일 구성

- `flamingo_demo.py`: 완전한 기능을 가진 OpenFlamingo QA 클래스
- `simple_flamingo_qa.py`: 간단한 대화형 스크립트
- `flamingo_demo.ipynb`: Jupyter 노트북 형태의 데모
- `demo_requirements.txt`: 필요한 Python 패키지 목록

## 설치 방법

### 1. 필수 패키지 설치

```bash
pip install -r demo_requirements.txt
```

또는 개별적으로:

```bash
pip install torch torchvision torchaudio
pip install open-flamingo
pip install huggingface-hub
pip install transformers
pip install requests pillow matplotlib
```

### 2. 모델 다운로드

첫 실행 시 모델이 자동으로 다운로드됩니다 (약 3GB). 안정적인 인터넷 연결이 필요합니다.

## 사용 방법

### 방법 1: 클래스 기반 사용 (추천)

```python
from flamingo_demo import FlamingoQA

# 모델 초기화 (첫 실행 시 시간이 걸림)
flamingo_qa = FlamingoQA()

# 이미지 로드
image = flamingo_qa.load_image_from_url("https://example.com/image.jpg")
# 또는 로컬 파일: flamingo_qa.load_image_from_path("path/to/image.jpg")

# 질문하기
answer = flamingo_qa.ask_question(image, "What do you see in this image?")
print(f"답변: {answer}")
```

### 방법 2: 대화형 스크립트

```bash
python simple_flamingo_qa.py
```

이 스크립트는 대화형 인터페이스를 제공하여:
- 예시 이미지 선택
- 커스텀 URL 이미지 사용
- 로컬 이미지 파일 사용
- 실시간 질문-답변

### 방법 3: Jupyter 노트북

```bash
jupyter notebook flamingo_demo.ipynb
```

노트북에서는 시각적으로 이미지를 확인하면서 단계별로 실험할 수 있습니다.

## 주요 기능

### 1. 이미지 질문-답변

```python
# 영어 질문
answer = flamingo_qa.ask_question(image, "How many cats are in the image?")

# 다양한 질문 타입
questions = [
    "What animals are in this image?",
    "What color is the cat?",
    "Where is the cat sitting?",
    "What is the cat doing?"
]
```

### 2. 자동 캡션 생성

```python
# 단순 캡션 생성
from simple_flamingo_qa import generate_caption
caption = generate_caption(model, image_processor, tokenizer, image)
```

### 3. Few-shot 캡셔닝

```python
# 예시 이미지와 캡션으로 학습 후 새 이미지 캡셔닝
example_images = [cat_image, dog_image]
example_captions = ["A cat sitting", "A dog running"]
new_caption = flamingo_qa.few_shot_caption(example_images, example_captions, new_image)
```

## 예시 이미지 URL

테스트용으로 다음 COCO 데이터셋 이미지들을 사용할 수 있습니다:

- 고양이: `http://images.cocodataset.org/val2017/000000039769.jpg`
- 해변: `http://images.cocodataset.org/val2017/000000281929.jpg`
- 음식: `http://images.cocodataset.org/val2017/000000060623.jpg`

## 성능 최적화

### GPU 사용

CUDA가 사용 가능한 경우 GPU를 사용하여 성능을 향상시킬 수 있습니다:

```python
if torch.cuda.is_available():
    model = model.cuda()
    print("GPU 사용 중")
```

### 메모리 관리

- 큰 이미지는 리사이즈하여 사용
- 배치 처리 시 메모리 사용량 주의
- 필요없는 변수는 `del`로 제거

## 문제해결

### 일반적인 오류

1. **모델 다운로드 실패**
   - 인터넷 연결 확인
   - 충분한 디스크 공간 확보 (최소 5GB)

2. **메모리 부족**
   - 이미지 크기 줄이기
   - `max_new_tokens` 값 줄이기
   - 배치 크기 줄이기

3. **느린 생성 속도**
   - GPU 사용 확인
   - `num_beams` 값 줄이기
   - `max_new_tokens` 값 줄이기

### 의존성 문제

```bash
# PyTorch 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OpenFlamingo 재설치
pip uninstall open-flamingo
pip install open-flamingo
```

## 모델 정보

- **사용 모델**: OpenFlamingo-3B-vitl-mpt1b
- **언어 모델**: MPT-1B
- **비전 인코더**: OpenAI CLIP ViT-L/14
- **모델 크기**: 약 3GB

## 라이선스

이 데모 코드는 원본 OpenFlamingo 프로젝트의 라이선스를 따릅니다.

## 참고 자료

- [OpenFlamingo GitHub](https://github.com/mlfoundations/open_flamingo)
- [OpenFlamingo Paper](https://arxiv.org/abs/2308.01390)
- [Hugging Face 모델 페이지](https://huggingface.co/openflamingo/OpenFlamingo-3B-vitl-mpt1b)
