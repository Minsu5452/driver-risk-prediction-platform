# DriverRisk Platform 설치 및 운영 가이드

본 문서는 배포된 `DriverRisk-Platform-AutoStart-v*.zip` 파일을 Windows PC에 설치·운영하는 방법을 안내합니다.

---

## 1. 사전 확인

- Windows 10 64bit 이상
- **관리자 권한** 필요 (설치 1회만)
- 디스크 여유 공간 2 GB 이상
- 파일 압축 해제 경로에 **한글/공백이 없어야 합니다** (예: `D:\temp\`)
  - ✗ 나쁜 예: `C:\Users\홍길동\바탕 화면\`
  - ✓ 좋은 예: `D:\temp\`, `C:\install\`

> 설치 후 최종 위치는 자동으로 `C:\DriverRisk-Platform` 이 됩니다.

---

## 2. 설치

1. zip 파일을 위 조건을 만족하는 폴더에 **압축 해제**
2. 생긴 `DriverRisk-Platform-AutoStart` 폴더 안의 **`install.bat`** 파일
3. **마우스 우클릭 → "관리자 권한으로 실행"**
4. 검은 창에서 자동으로 아래 작업이 진행됩니다 (2~3분 소요):
   - 기존 DriverRisk Platform 중지 (설치된 게 있다면)
   - `C:\DriverRisk-Platform` 에 파일 복사
   - Python 라이브러리 오프라인 설치
   - 방화벽 규칙 등록
   - 바탕화면 바로가기 생성 (**DriverRisk Start**, **DriverRisk Stop**, **DriverRisk Platform**)
   - Windows 로그온 시 자동 시작 등록
5. 완료되면 브라우저가 자동으로 `http://localhost:3000` 을 엽니다

---

## 3. 관리자 로그인

| 구분 | 기본값 |
|---|---|
| ID | `admin` |
| 비밀번호 | `change-this!` |

관리자 로그인: 화면 우측 상단 **관리자 로그인** 버튼 (또는 `http://localhost:3000/admin/login`)

---

## 4. 일상 운영

| 작업 | 방법 |
|---|---|
| **시작** | 바탕화면 **DriverRisk Start** 더블클릭 |
| **종료** | 바탕화면 **DriverRisk Stop** 더블클릭 |
| **접속** | 바탕화면 **DriverRisk Platform** 더블클릭 (또는 브라우저에서 `http://localhost:3000`) |
| **자동 시작** | PC 전원을 켜고 Windows에 로그인하면 자동으로 시작됩니다 |

> PC를 재부팅하면 서비스가 완전히 올라올 때까지 1~2분 걸릴 수 있습니다.

---

## 5. 데이터 / 로그 위치

설치 위치: `C:\DriverRisk-Platform`

```
C:\DriverRisk-Platform\
├── VERSION                      ← 설치된 버전 정보
├── ai-engine\data\              ← 검사 데이터 (DB 파일)
├── ai-engine\artifacts\         ← 학습된 AI 모델
└── logs\                        ← 각종 로그 파일
    ├── ai-engine.log
    ├── backend.log
    ├── nginx-access.log / nginx-error.log
    ├── install-pip.log / install-packages.log
```

---

## 6. 문제 해결

### "관리자 권한으로 실행하세요" 라는 메시지가 뜹니다
→ `install.bat` 우클릭 → **"관리자 권한으로 실행"** 선택

### 설치 도중 "Cannot remove old installation" 오류
→ `C:\DriverRisk-Platform` 폴더를 열어둔 **탐색기 창** 또는 **터미널** 이 있다면 모두 닫고 재시도

### 브라우저에서 `http://localhost:3000` 이 안 열립니다
1. 바탕화면 **DriverRisk Stop** → **DriverRisk Start** 순서로 다시 실행
2. 그래도 안 되면 명령 프롬프트에서 로그 확인:
   ```
   type C:\DriverRisk-Platform\logs\ai-engine-error.log
   type C:\DriverRisk-Platform\logs\backend-error.log
   ```

### 포트 충돌 (3000 / 8000 / 8080 이 다른 프로그램과 겹침)
→ 겹치는 프로그램을 종료한 뒤 **DriverRisk Start** 다시 실행

### 한글이 깨져 보입니다
→ 웹 브라우저 인코딩은 자동 UTF-8 입니다. 만약 깨지면 **Ctrl + F5** 로 강제 새로고침

---

## 7. 버전 업데이트

새 버전의 zip 파일을 받았을 때의 절차입니다.

> ⚠ **중요**: 기존 `C:\DriverRisk-Platform` 폴더는 **반드시 수동으로 삭제**해야 합니다.
> 자동 삭제 로직은 백신 프로그램이 "대량 파일 삭제 = 랜섬웨어"로 오인해
> 설치 터미널을 강제 종료하는 현상이 반복되어 의도적으로 제거했습니다.

**업데이트 순서**:

1. 바탕화면 **DriverRisk Stop** 실행 (현재 서비스 종료)
   - 또는 작업 관리자에서 나머지 관련 프로세스 확인
2. **Windows 탐색기**로 `C:\` 이동
3. **`DriverRisk-Platform`** 폴더 우클릭 → **삭제** (필요 시 `Shift+Delete`)
4. 삭제 완료까지 대기 (1~3분 소요. 탐색기 진행률 창이 뜹니다)
5. 새 zip 파일을 압축 해제 (`Downloads` 등 **다른** 경로)
6. 새 `install.bat` **관리자 권한으로 실행**
   - 기존 `C:\DriverRisk-Platform` 이 남아 있으면 `install.bat` 이 안내 메시지와 함께 중단됩니다
7. 설치 완료 후 자동으로 브라우저가 열립니다

> 기존 검사 데이터 / 학습된 모델 / 업로드 파일은 이 과정에서 **모두 삭제**됩니다. 새 배포본에서 다시 학습해야 합니다.

### 탐색기 삭제가 오래 걸린다면

폴더 안에 파일이 수만 개라 시간이 걸립니다. 휴지통을 거치지 않는 `Shift+Delete` 가 더 빠릅니다.

---

## 8. 제거

완전히 제거하려면:

1. 바탕화면 **DriverRisk Stop** 실행
2. 관리자 권한 명령 프롬프트에서:
   ```
   schtasks /Delete /TN "RISK_Platform" /F
   netsh advfirewall firewall delete rule name="DriverRisk Platform"
   rmdir /s /q C:\DriverRisk-Platform
   ```
3. 바탕화면의 **DriverRisk Start**, **DriverRisk Stop**, **DriverRisk Platform** 바로가기 삭제

---

## 9. 기술 지원

설치 관련 문의 또는 문제 발생 시, 해당 프로젝트 담당자에게 `C:\DriverRisk-Platform\logs\` 폴더 전체를 압축하여 전달해주세요.
