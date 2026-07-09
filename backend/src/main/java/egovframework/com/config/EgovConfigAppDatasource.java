package egovframework.com.config;

import javax.annotation.PostConstruct;
import javax.sql.DataSource;

import org.apache.commons.dbcp2.BasicDataSource;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;
import org.springframework.jdbc.datasource.embedded.EmbeddedDatabaseBuilder;
import org.springframework.jdbc.datasource.embedded.EmbeddedDatabaseType;

/**
 * DataSource 설정
 */
@Configuration
public class EgovConfigAppDatasource {

	private final Environment env;

	public EgovConfigAppDatasource(Environment env) {
		this.env = env;
	}

	private String dbType;

	private String className;

	private String url;

	private String userName;

	private String password;

	@PostConstruct
	void init() {
		dbType = env.getProperty("Globals.DbType");
		// DbType에 따른 JDBC 프로퍼티 로드 -- 프로퍼티 누락 시 null이 되어 basicDataSource()에서 실패할 수 있음
		className = env.getProperty("Globals." + dbType + ".DriverClassName");
		url = env.getProperty("Globals." + dbType + ".Url");
		userName = env.getProperty("Globals." + dbType + ".UserName");
		password = env.getProperty("Globals." + dbType + ".Password");
	}

	/**
	 * 개발용 내장 HSQL DataSource
	 */
	private DataSource dataSourceHSQL() {
		return new EmbeddedDatabaseBuilder()
				.setType(EmbeddedDatabaseType.HSQL)
				.setScriptEncoding("UTF8")
				.addScript("classpath:/db/risk.sql")
				.build();
	}

	/**
	 * 운영용 외부 DB DataSource (MySQL 등)
	 */
	private DataSource basicDataSource() {
		BasicDataSource basicDataSource = new BasicDataSource();
		basicDataSource.setDriverClassName(className);
		basicDataSource.setUrl(url);
		basicDataSource.setUsername(userName);
		basicDataSource.setPassword(password);
		return basicDataSource;
	}

	/**
	 * DbType 프로퍼티 값에 따라 내장(HSQL) 또는 외부 DB DataSource를 반환한다.
	 */
	@Bean(name = { "dataSource", "egov.dataSource", "egovDataSource" })
	public DataSource dataSource() {
		if ("hsql".equals(dbType)) {
			return dataSourceHSQL();
		} else {
			return basicDataSource();
		}
	}
}
