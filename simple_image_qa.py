#!/usr/bin/env python3
"""
OpenFlamingo 간단한 이미지 질문 도구
이미지 파일을 현재 폴더에 복사하고 질문하는 간단한 도구
"""

import os
import shutil
import glob
from flamingo_demo import FlamingoQA


def copy_image_to_current_folder():
    """사용자가 이미지 파일을 현재 폴더로 복사하도록 안내"""
    print("📁 이미지 파일을 현재 폴더에 복사해주세요!")
    print(f"현재 위치: {os.getcwd()}")
    print("\n지원하는 형식: .jpg, .jpeg, .png, .bmp, .gif, .webp")
    print("\n방법:")
    print("1. 파일 탐색기에서 이미지 파일을 찾으세요")
    print("2. 이미지 파일을 이 폴더로 드래그 앤 드롭하거나 복사하세요")
    print("3. 준비되면 Enter를 누르세요")
    
    input("\n준비 완료 후 Enter를 누르세요...")
    
    # 이미지 파일 찾기
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    found_images = []
    
    for ext in image_extensions:
        found_images.extend(glob.glob(ext, recursive=False))
        found_images.extend(glob.glob(ext.upper(), recursive=False))
    
    if not found_images:
        print("❌ 이미지 파일을 찾을 수 없습니다.")
        print("이미지 파일이 현재 폴더에 있는지 확인하세요.")
        return None
    
    print(f"\n✅ {len(found_images)}개의 이미지 파일을 찾았습니다:")
    for i, img in enumerate(found_images, 1):
        print(f"  {i}. {img}")
    
    if len(found_images) == 1:
        return found_images[0]
    
    # 여러 이미지가 있으면 선택
    while True:
        try:
            choice = int(input(f"\n사용할 이미지를 선택하세요 (1-{len(found_images)}): ")) - 1
            if 0 <= choice < len(found_images):
                return found_images[choice]
            else:
                print("올바른 번호를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")


def quick_questions(flamingo, image, image_name):
    """빠른 질문들을 제공"""
    print(f"\n🔍 '{image_name}' 분석 중...")
    
    quick_questions_list = [
        ("기본 설명", "What do you see in this image?"),
        ("상세 설명", "Describe this image in detail."),
        ("주요 객체", "What are the main objects in this image?"),
        ("색상", "What colors are prominent in this image?"),
        ("개수 세기", "How many people or animals are in this image?"),
        ("위치/장소", "Where is this image taken? What is the setting?"),
        ("활동/상황", "What is happening in this image?"),
        ("감정/분위기", "What is the mood or atmosphere of this image?")
    ]
    
    print("\n🚀 빠른 질문들:")
    for i, (title, question) in enumerate(quick_questions_list, 1):
        print(f"  {i}. {title}")
    
    print(f"  0. 직접 질문 입력")
    
    while True:
        try:
            choice = input(f"\n선택하세요 (0-{len(quick_questions_list)}): ").strip()
            
            if choice == "0":
                custom_question = input("질문을 입력하세요: ").strip()
                if custom_question:
                    print(f"\n질문: {custom_question}")
                    answer = flamingo.ask_question(image, custom_question, max_tokens=60)
                    print(f"답변: {answer}")
                    return custom_question, answer
                else:
                    print("질문을 입력하세요.")
                    continue
            
            elif choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(quick_questions_list):
                    title, question = quick_questions_list[choice_idx]
                    print(f"\n{title}: {question}")
                    answer = flamingo.ask_question(image, question, max_tokens=60)
                    print(f"답변: {answer}")
                    return question, answer
                else:
                    print("올바른 번호를 입력하세요.")
                    continue
            else:
                print("올바른 번호를 입력하세요.")
                continue
                
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            return None, None


def main():
    """메인 함수"""
    print("🦩 OpenFlamingo 간단 이미지 질문 도구")
    print("=" * 50)
    
    # 모델 로딩
    print("🔄 모델 로딩 중... (잠시만 기다려주세요)")
    try:
        flamingo = FlamingoQA(device="cpu", low_memory_mode=True)
        print("✅ 모델 준비 완료!")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return 1
    
    while True:
        try:
            # 이미지 가져오기
            image_path = copy_image_to_current_folder()
            if not image_path:
                print("\n다시 시도하시겠습니까? (y/n)")
                if input().lower().startswith('n'):
                    break
                continue
            
            # 이미지 로드
            print(f"\n📷 이미지 로딩: {image_path}")
            image = flamingo.load_image_from_path(image_path)
            
            if image is None:
                print("❌ 이미지 로드 실패")
                continue
            
            print(f"✅ 이미지 로드 성공! 크기: {image.size}")
            
            # 질문-답변
            question_count = 0
            while True:
                question, answer = quick_questions(flamingo, image, image_path)
                if question is None:
                    break
                
                question_count += 1
                
                print(f"\n{'='*50}")
                print("다음 선택:")
                print("1. 같은 이미지에 다른 질문하기")
                print("2. 다른 이미지 사용하기") 
                print("3. 종료")
                
                next_choice = input("선택하세요 (1-3): ").strip()
                
                if next_choice == "1":
                    continue
                elif next_choice == "2":
                    break
                elif next_choice == "3":
                    print("\n프로그램을 종료합니다.")
                    return 0
                else:
                    print("1, 2, 또는 3을 입력하세요.")
                    continue
            
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            print("계속하시겠습니까? (y/n)")
            if input().lower().startswith('n'):
                break
    
    print("감사합니다! 🦩")
    return 0


if __name__ == "__main__":
    exit(main())
