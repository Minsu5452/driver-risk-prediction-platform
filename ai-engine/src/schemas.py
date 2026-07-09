from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union


class DriverInput(BaseModel):
    """운전자 예측 요청 입력 스키마."""
    Test_id: str
    TestDate: str
    Age: Union[float, str]
    PrimaryKey: str
    domain: str  # 'A' (신규검사) 또는 'B' (자격유지검사)
    features: Dict[str, Any]  # 검사 피처 데이터


class PredictionResponse(BaseModel):
    """예측 결과 응답 스키마. 새 필드 추가 시 반드시 여기에도 추가할 것."""
    Test_id: str
    PrimaryKey: Optional[str] = None
    score: float
    result: float
    riskGroup: str
    masked_name: Optional[str] = None
    masked_dob: Optional[str] = None
    gender: Optional[str] = None
    TestDate: Optional[str] = None
    Age: Optional[str] = None
    domain: Optional[str] = None
    branch: Optional[str] = None  # 지사명
    original_name: Optional[str] = None  # 검색용 실명
    original_rrn: Optional[str] = None  # 마스킹 해제용 주민번호
    original_dob: Optional[str] = None  # 마스킹 해제용 생년월일
    birth_yyyymmdd: Optional[str] = None  # 8자리 출생일 — 프론트엔드 정확 만나이 재계산용

    # 추가 메타데이터 필드
    exam_age: Optional[str] = None
    current_age: Optional[str] = None
    industry: Optional[str] = None
    industry_detail: Optional[str] = None
    masked_rrn: Optional[str] = None

    features: Optional[Dict[str, Any]] = None  # 프론트엔드 SHAP 분석용

class PredictionInput(BaseModel):
    """예측 입력 항목 스키마."""
    Test_id: str = Field(
        ..., description="검사 세션 고유 식별자", example="TEST_12345"
    )
    PrimaryKey: str = Field(
        ..., description="운전자 고유 식별자", example="USER_999"
    )
    Age: Union[str, int] = Field(
        ..., description="운전자 연령 코드 또는 값", example="30a"
    )
    TestDate: Union[str, int] = Field(
        ..., description="검사 일자 (YYYYMM 또는 YYYY-MM-DD)", example="202305"
    )

    class Config:
        extra = "allow"


class GlobalExplainRequest(BaseModel):
    """글로벌 SHAP 설명 요청 스키마."""
    domain: str = Field("A", pattern="^(A|B)$")
    detailed: bool = False
    items: List[PredictionInput]


class GlobalExplainByIdsRequest(BaseModel):
    """서버 캐시 경유 글로벌 SHAP 요청. test_ids만 전달하면 서버가 캐시에서 features 로드."""
    domain: str = Field("A", pattern="^(A|B)$")
    test_ids: List[str]


