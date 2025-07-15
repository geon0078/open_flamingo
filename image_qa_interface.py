#!/usr/bin/env python3
"""
OpenFlamingo 이미지 업로드 및 질문-답변 인터페이스
사용자가 직접 이미지를 업로드하고 질문할 수 있는 대화형 프로그램
"""

import os
import sys
import glob
from flamingo_demo import FlamingoQA


def show_available_images():
    """현재 디렉토리의 사용 가능한 이미지 파일들을 보여줍니다."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    local_images = []
    
    for ext in image_extensions:
        local_images.extend(glob.glob(ext, recursive=False))
        local_images.extend(glob.glob(ext.upper(), recursive=False))
    
    return local_images


def get_image_from_user():
    """사용자로부터 이미지를 가져옵니다."""
    print("\n=== 이미지 선택 ===")
    print("1. 로컬 파일 경로 입력")
    print("2. 웹 URL 입력")
    print("3. 현재 디렉토리의 이미지 파일 선택")
    print("4. 예시 이미지 사용")
    
    while True:
        try:
            choice = input("\n선택하세요 (1-4): ").strip()
            
            if choice == "1":
                # 로컬 파일 경로
                file_path = input("이미지 파일 경로를 입력하세요: ").strip()
                if os.path.exists(file_path):
                    return ("file", file_path)
                else:
                    print(f"파일을 찾을 수 없습니다: {file_path}")
                    continue
            
            elif choice == "2":
                # 웹 URL
                url = input("이미지 URL을 입력하세요: ").strip()
                if url.startswith(("http://", "https://")):
                    return ("url", url)
                else:
                    print("올바른 URL을 입력하세요 (http:// 또는 https://로 시작)")
                    continue
            
            elif choice == "3":
                # 현재 디렉토리의 이미지
                local_images = show_available_images()
                if not local_images:
                    print("현재 디렉토리에 이미지 파일이 없습니다.")
                    continue
                
                print("\n현재 디렉토리의 이미지 파일들:")
                for i, img in enumerate(local_images, 1):
                    print(f"  {i}. {img}")
                
                try:
                    img_choice = int(input(f"이미지를 선택하세요 (1-{len(local_images)}): ")) - 1
                    if 0 <= img_choice < len(local_images):
                        return ("file", local_images[img_choice])
                    else:
                        print("잘못된 선택입니다.")
                        continue
                except ValueError:
                    print("숫자를 입력하세요.")
                    continue
            
            elif choice == "4":
                # 예시 이미지
                example_images = {
                    "1": ("고양이 이미지", "http://images.cocodataset.org/val2017/000000039769.jpg"),
                    "2": ("오토바이 이미지", "http://images.cocodataset.org/val2017/000000000139.jpg"),
                    "3": ("테디베어 이미지", "http://images.cocodataset.org/val2017/000000000285.jpg"),
                    "4": ("욕실 이미지", "http://images.cocodataset.org/test-stuff2017/000000028137.jpg")
                }
                
                print("\n예시 이미지들:")
                for key, (desc, url) in example_images.items():
                    print(f"  {key}. {desc}")
                
                example_choice = input("예시 이미지를 선택하세요 (1-4): ").strip()
                if example_choice in example_images:
                    desc, url = example_images[example_choice]
                    print(f"선택됨: {desc}")
                    return ("url", url)
                else:
                    print("잘못된 선택입니다.")
                    continue
            
            else:
                print("1-4 중에서 선택하세요.")
                continue
                
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            return None


def ask_questions_interactive(flamingo, image):
    """대화형으로 질문을 받고 답변합니다."""
    print("\n=== 질문-답변 세션 ===")
    print("이미지에 대해 질문하세요.")
    print("명령어:")
    print("  'quit' 또는 'exit' - 종료")
    print("  'help' - 도움말")
    print("  'suggest' - 추천 질문들 보기")
    print("  'caption' - 이미지 설명 생성")
    print("-" * 50)
    
    question_count = 0
    
    while True:
        try:
            user_input = input(f"\n질문 {question_count + 1}: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("질문-답변 세션을 종료합니다.")
                break
            
            elif user_input.lower() == 'help':
                print("\n도움말:")
                print("- 이미지에 대한 자연어 질문을 입력하세요")
                print("- 예: 'What do you see?', '이 이미지에 무엇이 있나요?'")
                print("- 'suggest'로 추천 질문들을 볼 수 있습니다")
                print("- 'caption'으로 이미지 설명을 생성할 수 있습니다")
                continue
            
            elif user_input.lower() == 'suggest':
                suggestions = [
                    "What do you see in this image?",
                    "Describe the main objects in the image.",
                    "What colors are prominent?",
                    "How many people/animals are there?",
                    "What is the setting or location?",
                    "What is happening in this image?",
                    "What is the mood or atmosphere?",
                    "Are there any text or signs visible?"
                ]
                
                print("\n추천 질문들:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
                
                try:
                    choice = input("\n추천 질문을 선택하시겠습니까? (1-8, 또는 Enter로 건너뛰기): ").strip()
                    if choice and choice.isdigit():
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(suggestions):
                            user_input = suggestions[choice_idx]
                            print(f"선택된 질문: {user_input}")
                        else:
                            print("잘못된 선택입니다.")
                            continue
                    else:
                        continue
                except ValueError:
                    continue
            
            elif user_input.lower() == 'caption':
                user_input = "Describe this image in detail."
            
            # 질문 처리
            print(f"\n처리 중... '{user_input}'")
            answer = flamingo.ask_question(image, user_input, max_tokens=50)
            
            print(f"답변: {answer}")
            question_count += 1
            
            # 매 5개 질문마다 메모리 정리 제안
            if question_count % 5 == 0:
                print(f"\n{question_count}개 질문 완료. 계속하시겠습니까? (y/n)")
                if input().lower().startswith('n'):
                    break
            
        except KeyboardInterrupt:
            print("\n질문-답변 세션을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            print("다른 질문을 시도해보세요.")


def main():
    """메인 함수"""
    print("🦩 OpenFlamingo 이미지 질문-답변 인터페이스")
    print("=" * 60)
    print("이미지를 업로드하고 질문해보세요!")
    
    # 모델 초기화
    print("\n모델 초기화 중...")
    try:
        flamingo = FlamingoQA(device="cpu", low_memory_mode=True)
        print("✅ 모델 로딩 완료!")
        
        # 모델 정보
        model_info = flamingo.get_model_info()
        print(f"디바이스: {model_info['device']}")
        print(f"파라미터 수: {model_info['model_parameters']:,}")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        print("시스템 메모리가 부족하거나 네트워크 연결을 확인하세요.")
        return 1
    
    # 메인 루프
    while True:
        try:
            # 이미지 선택
            image_info = get_image_from_user()
            if image_info is None:
                break
            
            image_type, image_source = image_info
            
            # 이미지 로드
            print(f"\n이미지 로딩 중: {os.path.basename(image_source) if image_type == 'file' else image_source}")
            
            if image_type == "file":
                image = flamingo.load_image_from_path(image_source)
            else:  # url
                image = flamingo.load_image_from_url(image_source)
            
            if image is None:
                print("❌ 이미지 로드에 실패했습니다.")
                continue
            
            print("✅ 이미지 로드 성공!")
            print(f"이미지 크기: {image.size}")
            
            # 질문-답변 세션
            ask_questions_interactive(flamingo, image)
            
            # 계속할지 물어보기
            print("\n다른 이미지로 계속하시겠습니까? (y/n)")
            if input().lower().startswith('n'):
                break
                
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            print("다시 시도하시겠습니까? (y/n)")
            if input().lower().startswith('n'):
                break
    
    print("\n감사합니다! 🦩")
    return 0


if __name__ == "__main__":
    exit(main())
