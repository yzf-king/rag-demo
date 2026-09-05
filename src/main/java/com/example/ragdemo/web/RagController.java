package com.example.ragdemo.web;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.ragdemo.service.RagService;

/**
 * 最小切片的两个接口（先 JSON，9/5 加流式 SSE + 前端页面）：
 *   POST /api/knowledge  {"text": "..."}  灌入知识
 *   POST /api/chat       {"question": "..."}  问答（返回答案 + 引用片段）
 */
@RestController
@RequestMapping("/api")
public class RagController {

    private final RagService ragService;

    public RagController(RagService ragService) {
        this.ragService = ragService;
    }

    @PostMapping("/knowledge")
    public IngestResult addKnowledge(@RequestBody IngestRequest request) {
        int chunks = ragService.ingest(request.text());
        return new IngestResult(chunks);
    }

    @PostMapping("/chat")
    public AskResult ask(@RequestBody AskRequest request) {
        return ragService.ask(request.question());
    }

    // ---- 请求/响应体（record：Java 16+ 的简洁写法，序列化字段名就是 JSON 字段名）----
    public record IngestRequest(String text) {}
    public record IngestResult(int chunks) {}
    public record AskRequest(String question) {}
    public record AskResult(String answer, java.util.List<String> sources) {}
}
