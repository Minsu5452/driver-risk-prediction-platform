package egovframework.let.risk.web;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

/**
 * AI Engine 프록시 컨트롤러
 * 프론트엔드의 분석 요청을 AI Engine(FastAPI)으로 중계한다.
 */
@RestController
@RequestMapping("/api/analysis")
public class AnalysisController {

  private static final Logger LOGGER = LoggerFactory.getLogger(AnalysisController.class);

  private static final Set<String> ALLOWED_CONTENT_TYPES = Collections.unmodifiableSet(
      new HashSet<>(Arrays.asList(
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
          "application/vnd.ms-excel",                                          // .xls
          "text/csv"                                                           // .csv
      )));

  @Value("${risk.ai-engine.url:http://127.0.0.1:8000}")
  private String aiEngineBaseUrl;

  /** RestTemplate을 매 요청마다 생성하지 않고 클래스 수준에서 재사용 */
  private final RestTemplate restTemplate;
  {
    SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
    factory.setConnectTimeout(5_000);
    factory.setReadTimeout(120_000);
    restTemplate = new RestTemplate(factory);
  }

  @PostMapping("/upload")
  public ResponseEntity<?> uploadFile(@RequestParam("file") MultipartFile file) {
    try {
      if (file.isEmpty()) {
        return ResponseEntity.badRequest().body("File is empty");
      }

      String contentType = file.getContentType();
      if (contentType == null || !ALLOWED_CONTENT_TYPES.contains(contentType)) {
        return ResponseEntity.badRequest().body("허용되지 않는 파일 형식입니다. Excel(.xlsx, .xls) 또는 CSV 파일만 업로드 가능합니다.");
      }

      HttpHeaders headers = new HttpHeaders();
      headers.setContentType(MediaType.MULTIPART_FORM_DATA);

      MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
      /*
       * ByteArrayResource는 기본적으로 getFilename()이 null을 반환하는데,
       * Spring의 multipart 요청 시 파일명이 없으면 Content-Disposition 헤더에
       * filename이 누락되어 수신 측(FastAPI)에서 파일을 인식하지 못한다.
       * 따라서 원본 파일명을 반환하도록 오버라이드한다.
       */
      body.add("file", new ByteArrayResource(file.getBytes()) {
        @Override
        public String getFilename() {
          return file.getOriginalFilename();
        }
      });

      HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

      ResponseEntity<String> response = restTemplate.postForEntity(aiEngineBaseUrl + "/predict/upload", requestEntity, String.class);

      Object responseBody = response.getBody();
      return ResponseEntity.status(response.getStatusCode())
          .body(responseBody != null ? responseBody : "응답 본문이 비어 있습니다.");

    } catch (HttpClientErrorException e) {
      LOGGER.error("AI 엔진 클라이언트 오류: {}", e.getStatusCode(), e);
      return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
    } catch (ResourceAccessException e) {
      LOGGER.error("AI 엔진 연결 실패", e);
      return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body("AI 엔진에 연결할 수 없습니다.");
    } catch (Exception e) {
      LOGGER.error("파일 업로드 처리 중 오류 발생", e);
      return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("파일 처리 중 오류가 발생했습니다.");
    }
  }

  @PostMapping("/explain_global")
  public ResponseEntity<?> explainGlobal(@RequestBody Map<String, Object> payload) {
    try {
      HttpHeaders headers = new HttpHeaders();
      headers.setContentType(MediaType.APPLICATION_JSON);

      HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(payload, headers);

      String aiUrl = aiEngineBaseUrl + "/predict/explain_global";
      @SuppressWarnings("unchecked")
      ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity(aiUrl, requestEntity, (Class<Map<String, Object>>)(Class<?>)Map.class);

      Object responseBody = response.getBody();
      return ResponseEntity.status(response.getStatusCode())
          .body(responseBody != null ? responseBody : "응답 본문이 비어 있습니다.");

    } catch (HttpClientErrorException e) {
      LOGGER.error("AI 엔진 클라이언트 오류: {}", e.getStatusCode(), e);
      return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
    } catch (ResourceAccessException e) {
      LOGGER.error("AI 엔진 연결 실패", e);
      return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body("AI 엔진에 연결할 수 없습니다.");
    } catch (Exception e) {
      LOGGER.error("SHAP 글로벌 설명 처리 중 오류 발생", e);
      return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("SHAP 설명 처리 중 오류가 발생했습니다.");
    }
  }
}
