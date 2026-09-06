package com.example.ragdemo.service;

import java.util.ArrayList;
import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import com.example.ragdemo.web.RagController;
import reactor.core.publisher.Flux;

/**
 * RAG 核心服务：入库（ingest）+ 问答（ask）两条链
 *
 * 离线链：ingest —— 文本 → 切分 chunk → 向量化 → 存入 pgvector
 * 在线链：ask   —— 问题 → 向量化 → 相似度检索 top-k → 拼 prompt → LLM 生成
 *
 * 切分目前用朴素的按段落/字符数切（最小版本先跑通），
 * 后续替换成正经 chunking 策略并对比效果（见 W2任务清单 9/6）。
 */
@Service
public class RagService {

    /**
     * 切分/检索参数全部外置到 application.yml（rag.*），改参数不用动代码——
     * 调参对比实验（9/7 评估）就是改 yml 重启，重复 N 次也不累。
     */
    @Value("${rag.chunk-size:300}")
    private int chunkSize;

    /** 相邻块重叠字符数：避免恰好把一句话从中间切开 */
    @Value("${rag.chunk-overlap:50}")
    private int chunkOverlap;

    /** 检索返回的相似块数量 top-k */
    @Value("${rag.top-k:3}")
    private int topK;

    /**
     * 单次 embedding 请求最多放几条：DashScope text-embedding-v3 限制单请求 ≤ 10，
     * 超出返回 400 InvalidParameter（9/5 实测踩坑）。留 2 条余量。
     */
    private static final int EMBED_BATCH_SIZE = 8;

    private final VectorStore vectorStore;
    private final ChatClient chatClient;

    public RagService(VectorStore vectorStore, ChatClient.Builder chatClientBuilder) {
        this.vectorStore = vectorStore;
        // ChatClient 是 Spring AI 的对话门面（类似 RestTemplate 之于 HTTP）
        this.chatClient = chatClientBuilder.build();
    }

    /** 入库：文本 → 切分 → 向量化 → 存库，返回切出的 chunk 数 */
    public int ingest(String text, String source) {
        List<String> chunks = split(text);
        List<Document> docs = chunks.stream()
                .map(c -> Document.builder()
                        .text(c)
                        .metadata("source", source) // 记录来源，方便前端展示出处
                        .build())
                .toList();
        // 分批入库：vectorStore.add() 会对这批 doc 一次性调 embedding 接口，
        // 超过 10 条就会撞上百炼的限制 → 每批最多 EMBED_BATCH_SIZE 条
        for (int i = 0; i < docs.size(); i += EMBED_BATCH_SIZE) {
            List<Document> batch = docs.subList(i, Math.min(i + EMBED_BATCH_SIZE, docs.size()));
            vectorStore.add(batch);
        }
        return docs.size();
    }

    /** 同步问答（curl 调试用；页面走下面的流式接口） */
    public RagController.AskResult ask(String question) {
        List<Document> hits = search(question);
        if (hits.isEmpty()) {
            return new RagController.AskResult("知识库还是空的，先调 POST /api/knowledge 或上传文件。", List.of());
        }
        String answer = generate(hits, question);
        return new RagController.AskResult(answer, hits.stream().map(Document::getText).toList());
    }

    /**
     * 流式问答（SSE 用）：先检索（同步，结果要拼进 prompt），
     * 生成阶段用 reactor Flux 逐 token 吐给前端逐字显示。
     * 返回结构 = 命中的原文块（引用展示）+ token 流。
     */
    public StreamAskResult streamAsk(String question) {
        List<Document> hits = search(question);
        if (hits.isEmpty()) {
            return new StreamAskResult(List.of(), Flux.just("知识库还是空的，先上传一份文档再提问吧。"));
        }
        // 拼 prompt 的公共逻辑（同步/流式一致）
        String system = "你是一个知识库问答助手。只依据下面提供的资料回答问题，"
                + "可以引用资料编号如 [资料1]。如果资料中没有相关信息，直接回答\"资料中没有相关信息\"，不要编造。";
        String userText = buildUserText(hits, question);
        UserMessage user = new UserMessage(userText);

        // .stream().content() 返回 Flux<String>，已内部过滤无文本的收尾块
        Flux<String> tokens = chatClient.prompt(new Prompt(List.of(new SystemMessage(system), user)))
                .stream()
                .content();
        return new StreamAskResult(hits.stream().map(Document::getText).toList(), tokens);
    }

    /** 检索公共步骤：问题自动向量化，在库里找最接近的 topK 块 */
    private List<Document> search(String question) {
        return vectorStore.similaritySearch(
                SearchRequest.builder().query(question).topK(topK).build());
    }

    /** 拼 prompt：检索结果带编号贴给模型（和 ask/streamAsk 共用，保证行为一致） */
    private String buildUserText(List<Document> hits, String question) {
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            context.append("[资料").append(i + 1).append("] ").append(hits.get(i).getText()).append("\n\n");
        }
        return "资料：\n" + context + "\n问题：" + question;
    }

    /** 同步生成：ChatClient .call() 一次性返回完整回答 */
    private String generate(List<Document> hits, String question) {
        String system = "你是一个知识库问答助手。只依据下面提供的资料回答问题，"
                + "可以引用资料编号如 [资料1]。如果资料中没有相关信息，直接回答\"资料中没有相关信息\"，不要编造。";
        return chatClient.prompt(new Prompt(List.of(
                new SystemMessage(system), new UserMessage(buildUserText(hits, question))))).call().content();
    }

    /**
     * 朴素切分（先跑通，勿深究）：按空行切成段落 → 超长段落再按字符数补切 → 相邻块保留 overlap。
     */
    private List<String> split(String text) {
        List<String> chunks = new ArrayList<>();
        for (String para : text.split("\\n\\s*\\n")) {
            String p = para.trim();
            if (p.isEmpty()) continue;
            while (p.length() > chunkSize) {
                int cut = p.lastIndexOf('。', chunkSize);
                if (cut < chunkSize / 2) cut = chunkSize; // 找不到句号就硬切
                chunks.add(p.substring(0, cut));
                p = p.substring(Math.max(0, cut - chunkOverlap));
            }
            chunks.add(p);
        }
        return chunks;
    }

    /** 流式问答结果：引用片段 + token 流 */
    public record StreamAskResult(List<String> sources, Flux<String> tokens) {}
}
