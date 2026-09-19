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

import org.springframework.jdbc.core.JdbcTemplate;

import com.example.ragdemo.web.RagController;
import reactor.core.publisher.Flux;

/**
 * RAG 核心服务：入库（ingest）+ 问答（ask）两条链
 *
 * 离线链：ingest —— 文本 → 切分 chunk → 向量化 → 存入 pgvector
 * 在线链：ask   —— 问题 → 向量化 → 相似度检索 top-k → 拼 prompt → LLM 生成
 *
 * 切分策略见 split()：标题跟随正文 + 段落贪心打包 + 长段按句切分。
 * 第一版是"段落超长才切"的朴素实现，实测发现 chunk-size 根本没生效（见 docs/evaluation.md）。
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

    /**
     * 低于这个字数的块语义太弱（典型是文末一行"（完）"），不再单独入库——
     * 单独成块只会占掉 top-k 名额，和第一版的"标题块污染"是同一类问题。
     */
    private static final int MIN_CHUNK_CHARS = 40;

    private final VectorStore vectorStore;
    private final ChatClient chatClient;
    private final JdbcTemplate jdbcTemplate;

    public RagService(VectorStore vectorStore, ChatClient.Builder chatClientBuilder, JdbcTemplate jdbcTemplate) {
        this.vectorStore = vectorStore;
        // ChatClient 是 Spring AI 的对话门面（类似 RestTemplate 之于 HTTP）
        this.chatClient = chatClientBuilder.build();
        this.jdbcTemplate = jdbcTemplate;
    }

    /**
     * 入库：文本 → 切分 → 向量化 → 存库，返回切出的 chunk 数。
     * chunkSizeOpt/OverlapOpt 为空时回落到 yml 默认——前端评测台选完参数随上传请求带过来，不用重启应用。
     */
    public int ingest(String text, String source, Integer chunkSizeOpt, Integer chunkOverlapOpt) {
        int cs = chunkSizeOpt != null ? chunkSizeOpt : chunkSize;
        int co = chunkOverlapOpt != null ? chunkOverlapOpt : chunkOverlap;
        // 防御：参数来自前端输入，非法值会让下方 while 死循环（如 cs=0 时切不断）
        if (cs < 10 || cs > 100000) throw new IllegalArgumentException("chunk-size 需在 10~100000 之间，当前：" + cs);
        if (co < 0 || co >= cs) throw new IllegalArgumentException("chunk-overlap 需在 0~chunk-size 之间，当前：" + co);
        List<String> chunks = split(text, cs, co);
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

    /** 同步问答（评测台/curl 调试用；页面走下面的流式接口）。topKOpt 为空时回落 yml 默认 */
    public RagController.AskResult ask(String question, Integer topKOpt) {
        List<Document> hits = search(question, topKOpt);
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
    public StreamAskResult streamAsk(String question, Integer topKOpt) {
        List<Document> hits = search(question, topKOpt);
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

    /** 检索公共步骤：问题自动向量化，在库里找最接近的 top-k 块（k 可运行时覆盖，见 ask/streamAsk） */
    private List<Document> search(String question, Integer topKOpt) {
        int k = topKOpt != null ? topKOpt : topK;
        if (k < 1 || k > 20) throw new IllegalArgumentException("top-k 需在 1~20 之间，当前：" + k);
        return vectorStore.similaritySearch(
                SearchRequest.builder().query(question).topK(k).build());
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
     * 切分策略（第二轮，替换第一版"段落永不合并"的朴素实现）。
     *
     * 第一版只在"单段长度 > chunk-size"时才切，而示例语料每段都短于最小档位（150），
     * 于是 A/B/C 三组参数切出的是同一批块——**参数等于没生效**（docs/evaluation.md 结论 1）。
     * 现在三条规则，让 cs 真正决定"一块装几段"：
     *   ① 标题跟随正文：`## 标题` 不单独成块，与紧随其后的正文合并
     *      （第一版里标题被切成独立小块，语义弱却稳定占 top-k 名额，即"标题块污染"，结论 2）
     *   ② 段落贪心打包：按顺序累积段落，直到再加一段就要超过 cs 为止
     *   ③ 单段超长：在句末标点处切（不切断句子），相邻片保留 co 字重叠
     * 段落之间不留重叠：段落边界本身就是天然语义边界，把上一段尾巴带进下一块只是噪音。
     * cs/co 由 ingest 传入：可能是 yml 默认，也可能是前端评测台选的运行时参数。
     */
    private List<String> split(String text, int cs, int co) {
        if (cs <= 0) throw new IllegalArgumentException("chunk-size 必须大于 0，当前：" + cs);
        // ingest 已校验过，这里再挡一次：co >= cs 会让下面的切分循环原地打转
        if (co < 0 || co >= cs) throw new IllegalArgumentException("chunk-overlap 需在 0~chunk-size 之间，当前：" + co);
        List<String> chunks = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (String block : toBlocks(text)) {
            if (block.length() > cs) {
                // 超长段落单独切：不跟已攒的小块混在一起，否则它的尾巴会把后面那段挤进同一个 chunk，边界变模糊
                if (cur.length() > 0) {
                    chunks.add(cur.toString());
                    cur.setLength(0);
                }
                chunks.addAll(splitLongBlock(block, cs, co));
                continue;
            }
            if (cur.length() > 0 && cur.length() + 1 + block.length() > cs) {
                chunks.add(cur.toString());   // 装上这段就超了 → 先把前面这块落盘
                cur.setLength(0);
            }
            if (cur.length() > 0) cur.append('\n');
            cur.append(block);
        }
        if (cur.length() > 0) chunks.add(cur.toString());
        return mergeTinyTail(chunks, cs);
    }

    /**
     * 段落化 + 标题归并。整段只有一行且以 # 开头 → 判为标题行，攒着；
     * 遇到正文段落就贴在它前面一起成块（标题单独成块正是"标题块污染"的来源）。
     */
    private List<String> toBlocks(String text) {
        List<String> blocks = new ArrayList<>();
        List<String> pendingHeadings = new ArrayList<>();
        for (String para : text.split("\\n\\s*\\n")) {
            String p = para.trim();
            if (p.isEmpty()) continue;
            if (p.startsWith("#") && !p.contains("\n")) {
                pendingHeadings.add(p);
                continue;
            }
            if (!pendingHeadings.isEmpty()) {
                p = String.join("\n", pendingHeadings) + "\n" + p;
                pendingHeadings.clear();
            }
            blocks.add(p);
        }
        if (!pendingHeadings.isEmpty()) {   // 文末没有正文的孤标题：并进上一块，别丢内容
            String last = String.join("\n", pendingHeadings);
            if (blocks.isEmpty()) blocks.add(last);
            else blocks.set(blocks.size() - 1, blocks.get(blocks.size() - 1) + "\n" + last);
        }
        return blocks;
    }

    /** 单段超长：从 cs 往回找句末标点当切点，找不到就硬切；相邻片重叠 co 字，避免把一句话切一半 */
    private List<String> splitLongBlock(String block, int cs, int co) {
        List<String> parts = new ArrayList<>();
        String rest = block;
        while (rest.length() > cs) {
            int cut = sentenceCut(rest, cs);
            parts.add(rest.substring(0, cut).trim());
            rest = rest.substring(Math.max(0, cut - co));   // co < cs，所以每轮至少前进 cs-co 个字，不会死循环
        }
        if (!rest.isBlank()) parts.add(rest.trim());
        return parts;
    }

    /** 在 (cs/2, cs] 区间里找最后一个句末标点（。！？；换行）当切点，整段没有就返回 cs 硬切 */
    private int sentenceCut(String s, int cs) {
        for (int i = cs; i > cs / 2; i--) {
            switch (s.charAt(i - 1)) {
                case '。', '！', '？', '；', '\n' -> {
                    return i;
                }
                default -> { }
            }
        }
        return cs;
    }

    /** 尾块过短（如文末一行"（完）"）就并进上一块——前提是并完不超 cs，绝不为了合并破坏块长上限 */
    private List<String> mergeTinyTail(List<String> chunks, int cs) {
        if (chunks.size() < 2) return chunks;
        int last = chunks.size() - 1;
        String tail = chunks.get(last);
        String prev = chunks.get(last - 1);
        if (tail.length() < MIN_CHUNK_CHARS && prev.length() + 1 + tail.length() <= cs) {
            chunks.set(last - 1, prev + "\n" + tail);
            chunks.remove(last);
        }
        return chunks;
    }

    /**
     * 只读检索：只回命中的原文片段和相似度，不调 LLM 生成。
     * 用途一：Agent（作品 2）把它当工具调——检索归 RAG 服务，编排归 Agent 服务，职责分开。
     * 用途二：调参时直接看"检索回来什么"，把检索质量和生成质量拆开判断。
     */
    public List<Hit> retrieve(String question, Integer topKOpt) {
        return search(question, topKOpt).stream()
                .map(d -> new Hit(d.getText(), d.getScore()))
                .toList();
    }

    /** 清空知识库（评测台切换切分参数/换语料前必须先清，防新旧 chunk 混检，见 docs/evaluation.md） */
    public int clear() {
        int existed = jdbcTemplate.queryForObject("SELECT count(*) FROM knowledge", Integer.class);
        jdbcTemplate.execute("TRUNCATE knowledge"); // 表名须与 LlmConfig 里 vectorTableName("knowledge") 保持一致
        return existed;
    }

    /** 流式问答结果：引用片段 + token 流 */
    public record StreamAskResult(List<String> sources, Flux<String> tokens) {}

    /** 检索命中：原文片段 + 相似度分数（score 越大越相关，余弦距离下通常 0~1） */
    public record Hit(String text, Double score) {}
}
