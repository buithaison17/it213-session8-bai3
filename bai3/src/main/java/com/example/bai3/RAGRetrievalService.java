package com.example.bai3;

import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class RAGRetrievalService {
    private final PgVectorStore pgVectorStore;
    private final ChatClient chatClient;

    public String chat(String message) {
        SearchRequest searchRequest = SearchRequest.builder()
                .query(message)
                .topK(3)
                .similarityThreshold(0.75)
                .build();
        List<Document> documents = pgVectorStore.similaritySearch(searchRequest);
        if (documents.isEmpty()) {
            return "Xin lỗi, thông tin bạn tìm kiếm không nằm trong tài liệu quy chế của chúng tôi";
        }
        String rawContext = documents.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n--\n"));
        String systemPrompt = """
                Giả sử bạn là trợ lí hệ thống hãy trở lời người dùng dựa theo ngữ cảnh: {context}
                Ràng buộc nghiệp vụ:
                - Chỉ trả lời nội dung có trong hệ thống
                - Nếu nội dung không có trong hệ thống hãy trả lời không biết
                """;
        return chatClient
                .prompt()
                .system(s -> s.text(systemPrompt).param("context", rawContext))
                .user(message)
                .call()
                .content();
    }
}
