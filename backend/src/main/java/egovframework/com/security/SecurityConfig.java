package egovframework.com.security;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.servlet.MultipartConfigFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.access.channel.ChannelProcessingFilter;
import org.springframework.security.web.csrf.CsrfFilter;
import org.springframework.util.unit.DataSize;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CharacterEncodingFilter;
import org.springframework.web.multipart.support.MultipartFilter;

import javax.servlet.MultipartConfigElement;
import java.util.Arrays;

/**
 * Spring Security 설정
 * 현재는 인증 없이 모든 요청을 허용하는 간소화된 구성이다.
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    /** CORS 허용 출처 (쉼표 구분, 프로퍼티로 오버라이드 가능) */
    @Value("${risk.cors.allowed-origins:http://localhost:3000}")
    private String allowedOrigins;

    /** CORS 정책 설정 -- 허용 메서드, 출처, 헤더 정의 */
    @Bean
    protected CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();

        configuration.setAllowedMethods(Arrays.asList("HEAD", "POST", "GET", "DELETE", "PUT", "PATCH"));
        configuration.setAllowedOrigins(Arrays.asList(allowedOrigins.split(",")));
        configuration.setAllowedHeaders(Arrays.asList("Content-Type", "Authorization", "X-Requested-With"));
        configuration.setAllowCredentials(true);

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }

    /** 요청/응답 UTF-8 인코딩 강제 적용 필터 */
    @Bean
    public CharacterEncodingFilter characterEncodingFilter() {
        CharacterEncodingFilter characterEncodingFilter = new CharacterEncodingFilter();
        characterEncodingFilter.setEncoding("UTF-8");
        characterEncodingFilter.setForceEncoding(true);
        return characterEncodingFilter;
    }

    /** 멀티파트 요청을 Security 필터 체인 이전에 파싱하기 위한 필터 */
    @Bean
    public MultipartFilter multipartFilter() {
        return new MultipartFilter();
    }

    /** 멀티파트 업로드 최대 크기 제한 (요청/파일 각 50MB) */
    @Bean
    public MultipartConfigElement multipartConfigElement() {
        MultipartConfigFactory factory = new MultipartConfigFactory();
        factory.setMaxRequestSize(DataSize.ofMegabytes(50L));
        factory.setMaxFileSize(DataSize.ofMegabytes(50L));
        return factory.createMultipartConfig();
    }

    /** Security 필터 체인 -- CSRF 비활성화, 세션 미사용(STATELESS), CORS 활성화 */
    @Bean
    protected SecurityFilterChain filterChain(HttpSecurity http) throws Exception {

        return http
                .csrf(AbstractHttpConfigurer::disable)
                .authorizeHttpRequests(authorize -> authorize
                        /* TODO(production): 운영 환경 배포 전 인증/인가 설정 필요 */
                        .anyRequest().permitAll())
                .sessionManagement(
                        (sessionManagement) -> sessionManagement.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .cors().and()
                .addFilterBefore(characterEncodingFilter(), ChannelProcessingFilter.class)
                .addFilterBefore(multipartFilter(), CsrfFilter.class)
                .build();
    }

}
