package com.example.ragdemo.config;

import javax.sql.DataSource;

import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.OpenAiEmbeddingModel;
import org.springframework.ai.openai.OpenAiEmbeddingOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore.PgDistanceType;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore.PgIndexType;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;

/**
 * 手动装配两家服务商 + 向量库，全部代码可见、可调，不用黑盒自动配置：
 *
 * 链路（对照 W2 任务清单的 RAG 全流程）：
 *   DeepSeek OpenAiApi ──> OpenAiChatModel     （生成回答，W1 已熟）
 *   DashScope OpenAiApi ─> OpenAiEmbeddingModel（文本向量化，1024 维）
 *                                     │
 *                                     ▼
 *   PgVectorStore（pgvector 表）──> 相似度检索 top-k
 *
 * 为什么手动建 bean：DeepSeek 和 DashScope 是两个不同的 OpenAI 兼容端点，
 * Spring AI 自动配置只认一家（spring.ai.openai.*），所以 yml 里关了自动配置。
 *
 * 注意：Spring AI 1.1.8 的包结构和 1.0 不同（OpenAiApi 在 .api 子包、
 * options 直接挂在 .openai 包、EmbeddingModel 用构造器、pgvector 的
 * builder 要传 JdbcTemplate 和 EmbeddingModel 两个参数），改版本必查！
 */
@Configuration
@EnableConfigurationProperties(LlmProperties.class)
public class LlmConfig {

    @Bean
    public OpenAiChatModel chatModel(LlmProperties props) {
        OpenAiApi api = OpenAiApi.builder()
                .baseUrl(props.getChat().getBaseUrl())
                .apiKey(props.getChat().getApiKey())
                .build();
        return OpenAiChatModel.builder()
                .openAiApi(api)
                .defaultOptions(OpenAiChatOptions.builder()
                        .model(props.getChat().getModel())
                        .build())
                .build();
    }

    @Bean
    public OpenAiEmbeddingModel embeddingModel(LlmProperties props) {
        OpenAiApi api = OpenAiApi.builder()
                .baseUrl(props.getEmbedding().getBaseUrl())
                .apiKey(props.getEmbedding().getApiKey())
                .build();
        // 1.1.x 的 EmbeddingModel 没有 builder，用构造器：
        // (api, MetadataMode.EMBED, 默认选项) —— EMBED 表示 embedding 输入含文档元数据
        return new OpenAiEmbeddingModel(api, MetadataMode.EMBED,
                OpenAiEmbeddingOptions.builder()
                        .model(props.getEmbedding().getModel())
                        .build());
    }

    @Bean
    public PgVectorStore vectorStore(DataSource dataSource, EmbeddingModel embeddingModel) {
        // 维度必须和 embedding 模型输出一致：text-embedding-v3 默认 1024 维
        return PgVectorStore.builder(new JdbcTemplate(dataSource), embeddingModel)
                .dimensions(1024)
                .vectorTableName("knowledge")
                .indexType(PgIndexType.HNSW)
                .distanceType(PgDistanceType.COSINE_DISTANCE)
                .initializeSchema(true) // 启动时自动建表（CREATE TABLE IF NOT EXISTS）
                .build();
    }
}
