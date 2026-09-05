package com.example.ragdemo.service;

import java.util.ArrayList;
import java.util.List;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import com.example.ragdemo.web.RagController;

/**
 * RAG 核心服务：入库（ingest）+ 问答（ask）两条链
 *
 * 离线链：ingest —— 文本 → 切分 chunk → 向量化 → 存入 pgvector
 * 在线链：ask   —— 问题 → 向量化 → 相似度检索 top-k → 拼 prompt → LLM 生成
 *
 * 切分这里先用最朴素的按段落/字符数切（今天的最小切片只求跑通），
 * 9/4 会替换成正经的 chunking 策略并对比效果。
 */
@Service
public class RagService {

    /** 单块最大字符数（中文场景粗略值，后续调参对比用） */
    private static final int CHUNK_SIZE = 300;
    /** 相邻块重叠字符数：避免恰好把一句话从中间切开 */
    private static final int CHUNK_OVERLAP = 50;
    /** 检索返回的相似块数量 top-k */
    private static final int TOP_K = 3;

    private final VectorStore vectorStore;
    private final ChatClient chatClient;

    public RagService(VectorStore vectorStore, ChatClient.Builder chatClientBuilder) {
        this.vectorStore = vectorStore;
        // ChatClient 是 Spring AI 的对话门面（类似 RestTemplate 之于 HTTP）
        this.chatClient = chatClientBuilder.build();
    }

    /** 入库：文本 → 切分 → 向量化 → 存库，返回切出的 chunk 数 */
    public int ingest(String text) {
        List<String> chunks = split(text);
        List<Document> docs = chunks.stream()
                .map(c -> Document.builder()
                        .text(c)
                        .metadata("source", "inline-text")
                        .build())
                .toList();
        vectorStore.add(docs); // 内部完成 embedding + 插入，打印日志能看到发了多少次 embedding 请求
        return docs.size();
    }

    /** 问答：检索 + 生成，返回答案和命中的原始片段（面试展示"答案有出处"的关键） */
    public RagController.AskResult ask(String question) {
        // 1. 相似度检索：问题自动向量化，在库里找最接近的 TOP_K 块
        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder().query(question).topK(TOP_K).build());

        if (hits.isEmpty()) {
            return new RagController.AskResult("知识库还是空的，先调 POST /api/knowledge 灌入文本。", List.of());
        }

        // 2. 拼 prompt：system 限定"只依据资料回答"，检索结果带编号贴给模型
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            context.append("[资料").append(i + 1).append("] ").append(hits.get(i).getText()).append("\n\n");
        }
        String system = "你是一个知识库问答助手。只依据下面提供的资料回答问题，"
                + "可以引用资料编号如 [资料1]。如果资料中没有相关信息，直接回答\"资料中没有相关信息\"，不要编造。";
        UserMessage user = new UserMessage("资料：\n" + context + "\n问题：" + question);

        // 3. 生成（先同步，9/5 改流式 SSE）
        // 1.1.x 里 .call() 返回 CallResponseSpec，.content() 直接取纯文本回答
        String answer = chatClient.prompt(new Prompt(List.of(new SystemMessage(system), user))).call().content();

        // 4. 把命中的原文块一并返回，前端可展示出处
        List<String> sources = hits.stream().map(Document::getText).toList();
        return new RagController.AskResult(answer, sources);
    }

    /**
     * 朴素切分（今天先用它跑通全链路，勿深究）：
     * 按空行切成段落 → 超长段落再按字符数补切 → 相邻块保留 overlap。
     */
    private List<String> split(String text) {
        List<String> chunks = new ArrayList<>();
        for (String para : text.split("\\n\\s*\\n")) {
            String p = para.trim();
            if (p.isEmpty()) continue;
            while (p.length() > CHUNK_SIZE) {
                int cut = p.lastIndexOf('。', CHUNK_SIZE);
                if (cut < CHUNK_SIZE / 2) cut = CHUNK_SIZE; // 找不到句号就硬切
                chunks.add(p.substring(0, cut));
                p = p.substring(Math.max(0, cut - CHUNK_OVERLAP));
            }
            chunks.add(p);
        }
        return chunks;
    }
}
