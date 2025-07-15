#!/usr/bin/env python3
"""
OpenFlamingo CPU 모드 간단 테스트
"""

from flamingo_demo import FlamingoQA

def simple_test():
    """간단한 CPU 테스트"""
    print("OpenFlamingo CPU 모드 간단 테스트")
    print("=" * 40)
    
    try:
        # 모델 초기화
        print("모델 로딩 중...")
        flamingo = FlamingoQA(device="cpu", low_memory_mode=True)
        
        # 테스트 이미지 URL
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        print(f"\n이미지 로딩: {image_url}")
        
        image = flamingo.load_image_from_url(image_url)
        if not image:
            print("이미지 로드 실패")
            return
        
        # 간단한 질문들
        questions = [
            "What do you see?",
            "How many animals?",
            "What are they doing?"
        ]
        
        print("\n=== 질문-답변 테스트 ===")
        for question in questions:
            print(f"\nQ: {question}")
            answer = flamingo.ask_question(image, question, max_tokens=20)
            print(f"A: {answer}")
        
        # 캡셔닝 테스트
        print("\n=== 단순 캡셔닝 테스트 ===")
        caption_question = "Describe this image in one sentence."
        caption = flamingo.ask_question(image, caption_question, max_tokens=25)
        print(f"캡션: {caption}")
        
        print("\n테스트 완료!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")

if __name__ == "__main__":
    simple_test()
