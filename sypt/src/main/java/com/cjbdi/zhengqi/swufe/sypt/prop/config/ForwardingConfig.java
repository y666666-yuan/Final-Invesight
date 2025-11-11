package com.cjbdi.zhengqi.swufe.sypt.prop.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
@ConfigurationProperties(prefix = "forwarding")
@Data
public class ForwardingConfig {
    private List<Route> routes;

    @Data
    public static class Route {
        private String id;
        private String uri;
        private List<String> paths;
        private Integer stripPrefix = 0;

        // 设置读取超时时间为20分钟
        private Integer readTimeout = 60 * 1000 * 20;
        private Integer connectionTimeout = 50000;
    }
}