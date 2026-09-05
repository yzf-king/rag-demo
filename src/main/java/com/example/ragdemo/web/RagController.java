package com.example.ragdemo.web;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.example.ragdemo.service.RagService;

/**
 * RAG 接口（前端页面 static/index.html 调的就是这三个）：
 *   POST /api/knowledge          {"text": "..."}           纯文本入库（curl 调试用）
 *   POST /api/knowledge/upload   multipart 文件上传（.txt/.md）
 *   GET  /api/chat/stream        ?question=... SSE 流式问答（逐字显示 + 引用）
 *   POST /api/chat               {"question": "..."}       同步问答（保留给 curl 调试）
 */
@RestController
@RequestMapping("/api")
public class RagController {

    private final RagService ragService;

    public RagController(RagService ragService) {
        this.ragService = ragService;
    }

    /** 纯文本入库（最小切片时用 JSON 传文本，前端不调这个） */
    @PostMapping("/knowledge")
    public IngestResult addKnowledge(@RequestBody IngestRequest request) {
        int chunks = ragService.ingest(request.text(), "inline-text");
        return new IngestResult(chunks);
    }

    /** 文件上传入库：multipart/form-data，字段名必须是 file（和前端页面约好的） */
    @PostMapping("/knowledge/upload")
    public IngestResult uploadKnowledge(@RequestParam("file") MultipartFile file) throws IOException {
        String filename = file.getOriginalFilename() == null ? "unnamed" : file.getOriginalFilename();
        if (!(filename.endsWith(".txt") || filename.endsWith(".md"))) {
            throw new IllegalArgumentException("仅支持 .txt / .md 文本文件，当前：" + filename);
        }
        if (file.isEmpty()) {
            throw new IllegalArgumentException("文件内容为空");
        }
        String text = new String(file.getBytes(), StandardCharsets.UTF_8);
        int chunks = ragService.ingest(text, filename);
        return new IngestResult(chunks);
    }

    /** SSE 流式问答：先发 source 事件（命中片段，前端展示引用），再逐 token 发 token 事件 */
    @GetMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter chatStream(@RequestParam String question) {
        SseEmitter emitter = new SseEmitter(0L); // 0L = 不超时

        RagService.StreamAskResult result;
        try {
            result = ragService.streamAsk(question); // 检索是同步的，在此完成
        } catch (Exception e) {
            // 检索失败（多半是 API key 没配好），发 fatal 事件让页面显示原因
            return fatalEmitter(e.getMessage());
        }

        // 1. 先把命中的原文块推给前端（生成还没开始，引用先到）
        for (int i = 0; i < result.sources().size(); i++) {
            sendEvent(emitter, "source", "[" + (i + 1) + "] " + result.sources().get(i));
        }

        // 2. 订阅生成 token 流：逐个 token 推送；结束/出错时收尾
        result.tokens().subscribe(
                token -> sendEvent(emitter, "token", token),
                err -> {
                    sendEvent(emitter, "fatal", "生成出错：" + err.getMessage());
                    emitter.complete();
                },
                emitter::complete);
        return emitter;
    }

    /** 同步问答（curl 调试保留） */
    @PostMapping("/chat")
    public AskResult ask(@RequestBody AskRequest request) {
        return ragService.ask(request.question());
    }

    // ---- 工具方法 ----

    /** 发送一个 SSE 事件；客户端可能已断开（complete 之后），忽略这类异常 */
    private void sendEvent(SseEmitter emitter, String name, String data) {
        try {
            emitter.send(SseEmitter.event().name(name).data(data));
        } catch (Exception ignored) {
            // 客户端断开或已 complete，忽略即可
        }
    }

    /** 检索阶段直接失败时：只发一个 fatal 事件就结束 */
    private SseEmitter fatalEmitter(String message) {
        SseEmitter emitter = new SseEmitter(0L);
        sendEvent(emitter, "fatal", message);
        emitter.complete();
        return emitter;
    }

    // ---- 请求/响应体（record：Java 16+ 简洁写法，字段名即 JSON 字段名）----
    public record IngestRequest(String text) {}
    public record IngestResult(int chunks) {}
    public record AskRequest(String question) {}
    public record AskResult(String answer, List<String> sources) {}
}
