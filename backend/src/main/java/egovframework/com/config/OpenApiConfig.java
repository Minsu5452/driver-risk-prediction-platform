package egovframework.com.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.security.SecurityScheme;

/**
 * Swagger/OpenAPI 문서 설정
 */
@Configuration
public class OpenApiConfig {

	private static final String API_NAME = "교통사고 위험도 예측 플랫폼";
	private static final String API_VERSION = "4.3.0";
	private static final String API_DESCRIPTION = "교통사고 위험도 예측 플랫폼 API 명세서";

	@Bean
	public OpenAPI api() {
		return new OpenAPI()
				.info(new Info().title(API_NAME)
				.description(API_DESCRIPTION)
				.version(API_VERSION)
				.contact(new Contact().name("eGovFrame").url("https://www.egovframe.go.kr/").email("egovframesupport@gmail.com"))
				.license(new License().name("Apache 2.0").url("https://www.apache.org/licenses/LICENSE-2.0")))
				.components(new Components()
						.addSecuritySchemes("Authorization", new SecurityScheme()
								.name("Authorization")
								.type(SecurityScheme.Type.APIKEY)
								.in(SecurityScheme.In.HEADER)));
	}
}
