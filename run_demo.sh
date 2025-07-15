#!/bin/bash
# OpenFlamingo 데모 실행 스크립트

echo "OpenFlamingo 데모 선택:"
echo "1. 기본 데모 (flamingo_demo.py)"
echo "2. 대화형 데모 (interactive_demo.py)"
echo "3. 평가 데모 (evaluation_demo.py)"
echo "4. 간단한 QA (simple_flamingo_qa.py)"

read -p "실행할 데모 번호를 선택하세요 (1-4): " choice

case $choice in
    1)
        echo "기본 데모 실행 중..."
        python flamingo_demo.py
        ;;
    2)
        echo "대화형 데모 실행 중..."
        python interactive_demo.py
        ;;
    3)
        echo "평가 데모 실행 중..."
        python evaluation_demo.py
        ;;
    4)
        echo "간단한 QA 실행 중..."
        python simple_flamingo_qa.py
        ;;
    *)
        echo "잘못된 선택입니다."
        ;;
esac
