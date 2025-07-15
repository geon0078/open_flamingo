#!/usr/bin/env python3
"""
OpenFlamingo 모델을 사용한 이미지 질문-답변 데모
이 스크립트는 OpenFlamingo 모델을 로드하고 이미지에 대해 질문할 수 있는 간단한 인터페이스를 제공합니다.
"""

import torch
import requests
from PIL import Image
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms


class FlamingoQA:
    def __init__(self, model_name="openflamingo/OpenFlamingo-3B-vitl-mpt1b", cache_dir=None, device="cpu", low_memory_mode=True):
        """
        OpenFlamingo 모델을 초기화합니다.
        
        Args:
            model_name (str): 사용할 OpenFlamingo 모델 이름
            cache_dir (str): 캐시 디렉토리 경로 (None이면 ~/.cache 사용)
            device (str): 사용할 디바이스 ("cpu" 권장)
            low_memory_mode (bool): 메모리 절약 모드 사용 여부
        """
        print("OpenFlamingo 모델을 로딩 중... (CPU 모드)")
        
        # 디바이스를 강제로 CPU로 설정
        self.device = "cpu"
        self.low_memory_mode = low_memory_mode
        print(f"사용 디바이스: {self.device}")
        print(f"메모리 절약 모드: {'활성화' if low_memory_mode else '비활성화'}")
        
        # CPU 최적화 설정
        torch.set_num_threads(min(4, torch.get_num_threads()))  # CPU 스레드 제한
        
        # 모델 설정 매핑
        model_configs = {
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b": {
                "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b",
                "tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b",
                "cross_attn_every_n_layers": 1
            },
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct": {
                "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
                "tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
                "cross_attn_every_n_layers": 1
            },
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b": {
                "lang_encoder_path": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
                "tokenizer_path": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
                "cross_attn_every_n_layers": 2
            },
            "openflamingo/OpenFlamingo-9B-vitl-mpt7b": {
                "lang_encoder_path": "anas-awadalla/mpt-7b",
                "tokenizer_path": "anas-awadalla/mpt-7b",
                "cross_attn_every_n_layers": 4
            }
        }
        
        # 모델 설정 가져오기
        if model_name in model_configs:
            config = model_configs[model_name]
        else:
            print(f"경고: 알 수 없는 모델 '{model_name}'. 기본 설정을 사용합니다.")
            config = model_configs["openflamingo/OpenFlamingo-3B-vitl-mpt1b"]
        
        try:
            # 모델과 변환기 생성 (CPU 최적화)
            self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path=config["lang_encoder_path"],
                tokenizer_path=config["tokenizer_path"],
                cross_attn_every_n_layers=config["cross_attn_every_n_layers"],
                cache_dir=cache_dir
            )
            
            # 모델을 CPU로 이동
            self.model = self.model.to(self.device)
            
            # 메모리 절약을 위한 설정
            if self.low_memory_mode:
                # 모델을 반정밀도로 변환하여 메모리 절약
                self.model = self.model.half()
                print("반정밀도(FP16) 모드로 메모리 사용량을 절약합니다.")
            
            # 사전 훈련된 가중치 다운로드 및 로드
            print("사전 훈련된 가중치를 다운로드 중...")
            checkpoint_path = hf_hub_download(model_name, "checkpoint.pt", cache_dir=cache_dir)
            
            # CPU에서 로드하고 메모리 최적화
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 메모리 절약을 위해 strict=False로 로드
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                print(f"누락된 키: {len(missing_keys)}개")
            if unexpected_keys:
                print(f"예상치 못한 키: {len(unexpected_keys)}개")
            
            # 체크포인트 메모리 해제
            del checkpoint
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            # 토크나이저 설정
            self.tokenizer.padding_side = "left"
            
            print("모델 로딩 완료! (CPU 최적화됨)")
            
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}")
            print("메모리 부족이 발생한 경우, 더 작은 모델을 사용하거나 시스템을 재시작해보세요.")
            raise

    def load_image_from_url(self, url):
        """URL에서 이미지를 로드합니다."""
        try:
            response = requests.get(url, stream=True)
            return Image.open(response.raw)
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            return None

    def load_image_from_path(self, path):
        """파일 경로에서 이미지를 로드합니다."""
        try:
            return Image.open(path)
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            return None

    def preprocess_image(self, image):
        """이미지를 모델 입력 형식으로 전처리합니다."""
        if image is None:
            return None
        
        # 이미지 크기 제한 (메모리 절약)
        max_size = 224
        if max(image.size) > max_size:
            image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
        
        vision_x = self.image_processor(image).unsqueeze(0)
        
        # 메모리 절약 모드에서는 FP16 사용
        if self.low_memory_mode:
            vision_x = vision_x.half()
        
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)  # batch_size x num_media x num_frames x channels x height x width
        return vision_x.to(self.device)

    def ask_question(self, image, question, max_tokens=30, temperature=0.7, num_beams=1, do_sample=False):
        """
        이미지에 대해 질문하고 답변을 생성합니다. (CPU 최적화)
        
        Args:
            image: PIL Image 객체
            question (str): 이미지에 대한 질문
            max_tokens (int): 생성할 최대 토큰 수 (CPU에서는 작게)
            temperature (float): 생성 시 temperature 값
            num_beams (int): beam search에 사용할 beam 수 (CPU에서는 1)
            do_sample (bool): 샘플링 사용 여부 (CPU에서는 False)
            
        Returns:
            str: 생성된 답변
        """
        if image is None:
            return "이미지가 제공되지 않았습니다."
        
        try:
            print(f"질문 처리 중: {question[:50]}...")
            
            # 이미지 전처리
            vision_x = self.preprocess_image(image)
            if vision_x is None:
                return "이미지 전처리에 실패했습니다."
            
            # 텍스트 전처리 - 질문 형식으로 구성
            prompt = f"<image>{question} Answer:"
            lang_x = self.tokenizer(
                [prompt],
                return_tensors="pt",
                max_length=512,  # 토큰 길이 제한
                truncation=True
            )
            
            # 토큰을 디바이스로 이동
            lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
            
            # 메모리 절약 모드에서는 FP16 사용
            if self.low_memory_mode and hasattr(lang_x["input_ids"], "half"):
                for key in lang_x:
                    if lang_x[key].dtype == torch.float32:
                        lang_x[key] = lang_x[key].half()
            
            # 메모리 정리
            import gc
            gc.collect()
            
            # 답변 생성 (CPU 최적화 설정)
            with torch.no_grad():
                generated_text = self.model.generate(
                    vision_x=vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    max_new_tokens=max_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,  # 조기 종료로 속도 향상
                    use_cache=False  # 메모리 절약
                )
            
            # 메모리 정리
            del vision_x, lang_x
            gc.collect()
            
            # 생성된 텍스트 디코딩 및 정리
            full_text = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)
            
            # 프롬프트 부분을 제거하고 답변만 반환
            answer_start = full_text.find("Answer:") + len("Answer:")
            answer = full_text[answer_start:].strip()
            
            return answer if answer else "답변을 생성할 수 없습니다."
            
        except Exception as e:
            print(f"질문 처리 중 오류 발생: {e}")
            # 메모리 정리
            import gc
            gc.collect()
            return f"오류: CPU 메모리 부족. 더 간단한 질문을 시도해보세요."

    def few_shot_caption(self, images, captions, query_image, max_tokens=15):
        """
        Few-shot 학습을 사용하여 이미지 캡션을 생성합니다. (CPU 최적화)
        
        Args:
            images: 예시 이미지들의 리스트
            captions: 예시 캡션들의 리스트
            query_image: 캡션을 생성할 이미지
            max_tokens (int): 생성할 최대 토큰 수 (CPU에서는 작게)
            
        Returns:
            str: 생성된 캡션
        """
        try:
            print("Few-shot 캡션 생성 중... (CPU 모드)")
            
            # 이미지 개수 제한 (메모리 절약)
            if len(images) > 2:
                images = images[:2]
                captions = captions[:2]
                print("메모리 절약을 위해 예시 이미지를 2개로 제한합니다.")
            
            # 모든 이미지 전처리
            all_images = images + [query_image]
            vision_x_list = []
            
            for i, img in enumerate(all_images):
                if img is None:
                    return "이미지가 제공되지 않았습니다."
                
                # 이미지 크기 제한
                max_size = 224
                if max(img.size) > max_size:
                    img = img.resize((max_size, max_size), Image.Resampling.LANCZOS)
                
                processed = self.image_processor(img).unsqueeze(0)
                if self.low_memory_mode:
                    processed = processed.half()
                vision_x_list.append(processed)
                print(f"이미지 {i+1}/{len(all_images)} 전처리 완료")
            
            vision_x = torch.cat(vision_x_list, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(self.device)
            
            # Few-shot 프롬프트 구성 (간단하게)
            prompt_parts = []
            for i, caption in enumerate(captions):
                prompt_parts.append(f"<image>{caption}<|endofchunk|>")
            prompt_parts.append("<image>")
            
            prompt = "".join(prompt_parts)
            
            lang_x = self.tokenizer(
                [prompt],
                return_tensors="pt",
                max_length=256,  # 토큰 길이 제한
                truncation=True
            )
            
            # 토큰을 디바이스로 이동
            lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
            
            # 메모리 정리
            import gc
            gc.collect()
            
            # 캡션 생성 (CPU 최적화)
            with torch.no_grad():
                generated_text = self.model.generate(
                    vision_x=vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    max_new_tokens=max_tokens,
                    num_beams=1,  # CPU에서는 beam search 사용 안함
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    use_cache=False
                )
            
            # 메모리 정리
            del vision_x, lang_x
            gc.collect()
            
            # 생성된 텍스트에서 새로운 캡션 추출
            full_text = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)
            print(f"생성된 전체 텍스트: {full_text}")
            
            # 마지막 <image> 태그 이후의 텍스트가 생성된 캡션
            last_image_pos = full_text.rfind("<image>")
            if last_image_pos != -1:
                caption = full_text[last_image_pos + len("<image>"):].strip()
                # <|endofchunk|> 태그가 있다면 제거
                caption = caption.replace("<|endofchunk|>", "").strip()
                # 불필요한 반복 제거
                if caption.startswith("An image of"):
                    return caption
                elif caption:
                    return f"An image of {caption}"
                else:
                    # 간단한 캡션 생성으로 폴백
                    simple_caption = self.ask_question(query_image, "What is this?", max_tokens=10)
                    return f"An image of {simple_caption}" if simple_caption else "이미지 캡션"
            
            return "이미지 캡션"
            
        except Exception as e:
            print(f"캡션 생성 중 오류 발생: {e}")
            import gc
            gc.collect()
            return f"오류: CPU 메모리 부족으로 캡션 생성 실패"
    
    def batch_ask_questions(self, image, questions, max_tokens=30):
        """
        하나의 이미지에 대해 여러 질문을 처리합니다. (CPU 최적화)
        
        Args:
            image: PIL Image 객체
            questions: 질문들의 리스트
            max_tokens (int): 생성할 최대 토큰 수
            
        Returns:
            dict: 질문-답변 쌍의 딕셔너리
        """
        results = {}
        
        # 질문 개수 제한 (메모리 절약)
        if len(questions) > 4:
            questions = questions[:4]
            print("메모리 절약을 위해 질문을 4개로 제한합니다.")
        
        print(f"배치 처리: {len(questions)}개 질문")
        
        for i, question in enumerate(questions, 1):
            print(f"질문 {i}/{len(questions)} 처리 중...")
            answer = self.ask_question(image, question, max_tokens)
            results[question] = answer
            
            # 메모리 정리 (각 질문 처리 후)
            import gc
            gc.collect()
            
        return results
    
    def get_model_info(self):
        """모델 정보를 반환합니다."""
        return {
            "device": self.device,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def main():
    """데모 실행 함수"""
    print("OpenFlamingo 이미지 질문-답변 데모")
    print("=" * 50)
    
    # 사용 가능한 모델들
    available_models = [
        "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
        "openflamingo/OpenFlamingo-4B-vitl-rpj3b",
        "openflamingo/OpenFlamingo-9B-vitl-mpt7b"
    ]
    
    print(f"사용 가능한 모델들:")
    for i, model in enumerate(available_models):
        print(f"  {i+1}. {model}")
    
    # 모델 초기화 (CPU 모드)
    try:
        print("CPU 모드로 모델 초기화 중...")
        print("주의: CPU 모드는 속도가 느릴 수 있습니다.")
        flamingo_qa = FlamingoQA(device="cpu", low_memory_mode=True)
        
        # 모델 정보 출력
        model_info = flamingo_qa.get_model_info()
        print(f"\n모델 정보:")
        print(f"  디바이스: {model_info['device']}")
        print(f"  총 파라미터 수: {model_info['model_parameters']:,}")
        print(f"  훈련 가능 파라미터 수: {model_info['model_trainable_parameters']:,}")
        
    except Exception as e:
        print(f"모델 초기화 실패: {e}")
        print("메모리 부족이 발생한 경우:")
        print("1. 시스템을 재시작해보세요")
        print("2. 다른 프로그램을 종료해보세요")
        print("3. 더 많은 RAM이 있는 시스템을 사용해보세요")
        return
    
    # 예시 이미지들 (COCO 데이터셋에서)
    print("\n예시 1: 단일 이미지 질문-답변")
    print("-" * 30)
    
    # 고양이 이미지
    cat_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(f"이미지 로딩 중: {cat_image_url}")
    cat_image = flamingo_qa.load_image_from_url(cat_image_url)
    
    if cat_image:
        questions = [
            "What animals are in this image?",
            "How many cats are there?",
            "What color are the cats?",
            "Where are the cats sitting?"
        ]
        
        # 배치 질문 처리
        print("질문들을 처리 중...")
        results = flamingo_qa.batch_ask_questions(cat_image, questions)
        
        for question, answer in results.items():
            print(f"질문: {question}")
            print(f"답변: {answer}")
            print()
    else:
        print("고양이 이미지 로드에 실패했습니다.")
    
    print("\n예시 2: Few-shot 이미지 캡셔닝")
    print("-" * 30)
    
    # Few-shot 예시를 위한 이미지들
    demo_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg"
    ]
    
    print("Few-shot 예시 이미지들을 로딩 중...")
    demo_images = []
    for i, url in enumerate(demo_urls):
        print(f"  이미지 {i+1} 로딩 중...")
        image = flamingo_qa.load_image_from_url(url)
        demo_images.append(image)
    
    if all(img is not None for img in demo_images):
        example_images = demo_images[:2]
        query_image = demo_images[2]
        example_captions = [
            "An image of two cats.",
            "An image of a bathroom sink."
        ]
        
        print("Few-shot 캡션 생성 중...")
        generated_caption = flamingo_qa.few_shot_caption(
            example_images, example_captions, query_image
        )
        
        print("Few-shot 캡셔닝 결과:")
        print(f"생성된 캡션: {generated_caption}")
    else:
        print("일부 이미지 로드에 실패했습니다.")
    
    print("\n예시 3: 로컬 이미지 처리 (있는 경우)")
    print("-" * 30)
    
    # 현재 디렉토리에서 이미지 파일 찾기
    import glob
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    local_images = []
    for ext in image_extensions:
        local_images.extend(glob.glob(ext))
        local_images.extend(glob.glob(ext.upper()))
    
    if local_images:
        local_image_path = local_images[0]
        print(f"로컬 이미지 발견: {local_image_path}")
        local_image = flamingo_qa.load_image_from_path(local_image_path)
        
        if local_image:
            test_questions = [
                "What is in this image?",
                "Describe what you see.",
                "What colors are prominent in this image?"
            ]
            
            print("로컬 이미지에 대한 질문 처리 중...")
            for question in test_questions:
                answer = flamingo_qa.ask_question(local_image, question)
                print(f"질문: {question}")
                print(f"답변: {answer}")
                print()
        else:
            print("로컬 이미지 로드에 실패했습니다.")
    else:
        print("현재 디렉토리에 이미지 파일이 없습니다.")
    
    print("\n데모 완료!")
    print("다른 이미지나 질문을 시도하려면 코드를 수정하여 실행하세요.")


if __name__ == "__main__":
    main()
