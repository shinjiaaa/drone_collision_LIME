import os, json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_lime_mask(pos_mask: np.ndarray, neg_mask: np.ndarray) -> str:
    """
    LIME 마스크 요약: 가장 큰 영향력을 가진 영역을 분석.
    pos_mask: 충돌 확률을 높이는 영역
    neg_mask: 충돌 확률을 낮추는 영역
    """
    # 간단한 수치 기반 특징 추출
    pos_intensity = float(np.mean(pos_mask) * 100)
    neg_intensity = float(np.mean(neg_mask) * 100 if neg_mask is not None else 0.0)
    dominant = "positive" if pos_intensity > neg_intensity else "negative"

    description = {
        "dominant": dominant,
        "pos_intensity": pos_intensity,
        "neg_intensity": neg_intensity,
        "analysis": "충돌 확률에 가장 크게 기여한 시각적 영역의 평균 강도값입니다.",
    }
    return json.dumps(description, ensure_ascii=False, indent=2)


def generate_lime_explanation(
    pos_mask: np.ndarray, neg_mask: np.ndarray, class_name: str, collision_prob: float
):
    lime_summary = summarize_lime_mask(pos_mask, neg_mask)
    prompt = f"""
당신은 영상 기반 충돌 탐지 해석 전문가입니다.
LIME이 제공한 시각적 분석 결과를 사람이 바로 이해할 수 있는 자연어로 해석해야 합니다.

### 입력 정보
- LIME 데이터: {lime_summary}
- 대상 객체: {class_name}
- 충돌 확률: {collision_prob:.2f}

### 목표
사용자가 "아, 이래서 위험하다고 판단했구나"라고 직관적으로 이해할 수 있도록 다음 네 항목을 작성하세요.

1. **Reason (충돌 판단 이유)**  
   - 충돌 확률이 높게 나온 구체적 이유를 서술합니다.  
   - 예: "전방 차량이 가까워지고 있어 즉시 감속이 필요합니다."

2. **Visual Focus (주요 시각적 영역)**  
   - LIME의 positive/negative 영역을 실제 장면 속 물체로 풀어서 설명합니다.  
   - 예: "차량의 앞부분", "사람의 몸통", "도로 중앙선", "하늘 영역"  
   - 단, 단순 'positive/negative' 용어는 사용하지 않고, 사용자가 직관적으로 이해할 수 있게 표현합니다.

3. **Intensity Interpretation (영향 강도 해석)**  
   - 각 영역이 충돌 판단에 얼마나 강하게 작용했는지 수치 대신 자연어로 설명합니다.  
   - 예: "이 영역은 판단에 매우 큰 영향을 미쳤습니다", "중간 정도의 영향을 미쳤습니다"

4. **Summary (간결 요약)**  
   - 위 세 항목 내용을 한 문장으로 자연스럽게 요약합니다.  
   - 반복적인 표현은 피하고, 새로운 표현을 사용하세요.

### 출력 형식 (JSON)
{{
  "reason": "...",
  "visual_focus": "...",
  "intensity_interpretation": "...",
  "summary": "..."
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text_output = response.choices[0].message.content
    try:
        return json.loads(text_output)
    except:
        return {"summary": text_output.strip()}
