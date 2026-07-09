package egovframework.com.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.annotation.PropertySources;

@Configuration
@Import({
		EgovConfigAppCommon.class,
		EgovConfigAppDatasource.class
})
@PropertySources({
		@PropertySource("classpath:/application.properties")
})
public class EgovConfigApp {

}
