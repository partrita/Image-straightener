# Image Straightener

학회 사진을 비스듬하게 찍어도 자동으로 보정해주는 도구입니다.

> A Python-based tool for automatically detecting and correcting skew in batches of images.

## 주요 기능

- **자동화된 문서 스캔:** 이미지에서 문서 윤곽을 자동으로 감지하고 원근을 보정합니다.
- **병렬 처리:** 여러 이미지를 동시에 처리하여 대용량 작업 시간을 단축합니다.
- **상세 로깅:** 처리 과정과 오류를 로그 파일에 기록하여 디버깅을 용이하게 합니다.

# 사용 방법

1. `uv`를 설치한 후, 아래 명령어로 필요한 패키지를 설치합니다.

```bash
uv sync
```

2. 다음 명령어를 실행하여 이미지를 처리합니다.

```bash
uv run image-straightener -- --input_path data/input/ --output_path data/output/
```

- `--input_path`: 처리할 이미지가 있는 디렉토리 경로
- `--output_path`: 처리된 이미지를 저장할 디렉토리 경로
- `--workers` (선택 사항): 병렬 처리에 사용할 프로세스 수 (기본값: CPU 코어 수)
- `--log_file` (선택 사항): 로그를 저장할 파일 경로 (기본값: `batch_scan.log`)

## 결과 확인

`output_path`로 지정한 디렉토리에서 처리된 이미지를 확인할 수 있습니다.

![](./static/workflow_2.jpg)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
