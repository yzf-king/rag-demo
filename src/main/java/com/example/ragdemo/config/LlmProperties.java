package com.example.ragdemo.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 绑定 application.yml 里 llm.* 配置（chat = DeepSeek，embedding = DashScope）。
 * Key 从环境变量读取，由 Spring 的 ${VAR:} 占位符注入，这里只做结构绑定。
 */
@ConfigurationProperties(prefix = "llm")
public class LlmProperties {

    private Chat chat = new Chat();
    private Embedding embedding = new Embedding();

    public Chat getChat() { return chat; }
    public void setChat(Chat chat) { this.chat = chat; }

    public Embedding getEmbedding() { return embedding; }
    public void setEmbedding(Embedding embedding) { this.embedding = embedding; }

    public static class Chat {
        private String baseUrl;
        private String model;
        private String apiKey;
        // getter/setter（省略的样板代码，可自行补全——Spring 通过 setter 注入）
        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
        public String getModel() { return model; }
        public void setModel(String model) { this.model = model; }
        public String getApiKey() { return apiKey; }
        public void setApiKey(String apiKey) { this.apiKey = apiKey; }
    }

    public static class Embedding {
        private String baseUrl;
        private String model;
        private String apiKey;
        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
        public String getModel() { return model; }
        public void setModel(String model) { this.model = model; }
        public String getApiKey() { return apiKey; }
        public void setApiKey(String apiKey) { this.apiKey = apiKey; }
    }
}
