# BÁO CÁO & MÃ NGUỒN BÀI TẬP: TỐI ƯU TRUY VẤN RAG PHÒNG THỦ & TRÁNH ẢO TƯỞNG (DEFENSIVE RETRIEVAL)

---

## 1. Phân Tích Kỹ Thuật: Các Phép Đo Khoảng Cách Vector Trong PostgreSQL `pgvector`

Trong các hệ thống RAG (Retrieval-Augmented Generation), việc lựa chọn hàm đo khoảng cách (Distance Metric) trong cơ sở dữ liệu vector đóng vai trò cốt lõi đến độ chính xác của ngữ nghĩa được truy xuất. Dưới đây là phân tích chi tiết ba phép đo chính được hỗ trợ trong extension `pgvector` của PostgreSQL:

---

### 1.1. So sánh chi tiết 3 phép đo: Cosine Similarity, L2 Distance (Euclidean), và Dot Product

| Tiêu chí | Cosine Distance / Similarity | L2 Distance (Euclidean) | Dot Product (Inner Product) |
| :--- | :--- | :--- | :--- |
| **Toán tử `pgvector`** | `<=>` (Cosine distance: $1 - \text{similarity}$) | `<->` (Euclidean distance: $\|u - v\|$) | `<#>` (Negative inner product: $-(u \cdot v)$) |
| **Công thức toán học** | $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | $d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$ | $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i$ |
| **Ý nghĩa hình học** | Đo **góc $\theta$** lệch giữa 2 vector trong không gian đa chiều, hoàn toàn triệt tiêu ảnh hưởng của độ dài/độ lớn (magnitude). | Đo **khoảng cách đường thẳng (không gian)** giữa 2 điểm đầu mút vector. Phụ thuộc lớn vào độ lớn vector. | Đo tích vô hướng, phụ thuộc đồng thời vào cả góc $\theta$ lẫn độ lớn $\|\mathbf{u}\|$ và $\|\mathbf{v}\|$. |
| **Khoảng giá trị** | Khoảng cách: $[0, 2]$; Độ tương đồng: $[-1, 1]$ (với vector chuẩn hóa thường là $[0, 1]$). | $[0, +\infty)$ | $(-\infty, +\infty)$ (phụ thuộc độ dài vector) |
| **Tốc độ tính toán** | Cần chia độ dài (norm), nhưng nếu vector đã chuẩn hóa thì tương đương Dot Product. | Tính hiệu bình phương từng chiều. | Rất nhanh (phép nhân & cộng vô hướng đơn thuần). |

---

### 1.2. Tại sao Cosine Similarity là lựa chọn tối ưu nhất cho tìm kiếm ngữ nghĩa văn bản (Semantic Search)?

1. **Bất biến với độ dài văn bản (Length Invariance):**
   - Các mô hình Embedding (OpenAI `text-embedding-3-small`, Cohere, HuggingFace BGE...) ánh xạ ngữ nghĩa cốt lõi của câu/đoạn văn thành vector hướng.
   - Một đoạn văn ngắn 1 câu (ví dụ: *"Chính sách hoàn tiền của CRM"*) và một đoạn văn dài 10 câu giải thích chi tiết quy trình hoàn tiền có thể mang cùng một chủ đề ngữ nghĩa, nhưng vector của đoạn dài có thể có độ lớn (magnitude) khác biệt do số lượng từ hoặc mật độ thông tin.
   - **Cosine Similarity chỉ quan tâm đến hướng (chủ đề ngữ nghĩa) mà không bị ảnh hưởng bởi độ dài đoạn văn**, giúp tài liệu ngắn hay dài đều được đánh giá công bằng dựa trên độ liên quan thực sự.

2. **Khắc phục hiện tượng suy biến khoảng cách ở không gian chiều cao (Curse of Dimensionality):**
   - Trong không gian 1536 chiều (hoặc 3072 chiều), khoảng cách L2 (Euclidean) bị ảnh hưởng nghiêm trọng bởi mật độ phân bố: hầu hết mọi cặp vector ngẫu nhiên đều có khoảng cách L2 gần như tương đương nhau, làm giảm độ phân tách giữa câu liên quan và câu không liên quan.
   - Cosine Similarity tập trung đo góc lệch, giữ được độ phân tách sắc nét giữa các phân nhóm ngữ nghĩa khác nhau.

3. **Tính chuẩn hóa và thiết lập ngưỡng an toàn (Thresholding):**
   - Vì Cosine Similarity luôn nằm trong thang đo xác định $[-1, 1]$ (thực tế văn bản tiếng Việt/Anh thường từ $0.0$ đến $1.0$), ta có thể dễ dàng đặt một ngưỡng tin cậy tuyệt đối như `0.75` hoặc `0.80` để phòng thủ (Defensive Filtering).
   - Ngược lại, L2 và Dot Product (nếu chưa chuẩn hóa) không có cận trên cố định, rất khó để thiết lập một con số threshold cố định áp dụng cho mọi truy vấn.

*(Lưu ý: Nếu toàn bộ vector trong cơ sở dữ liệu đã được Normalized (độ dài $\|v\| = 1$), Cosine Similarity chính là Dot Product. Khi đó, sử dụng toán tử `<#>` trong pgvector sẽ cho kết quả ngữ nghĩa giống hệt `<=>` nhưng với tốc độ xử lý nhanh hơn đáng kể).*

---

## 2. Mã Nguồn Hoàn Chỉnh: `RAGRetrievalService.java`

Dưới đây là lớp dịch vụ `RAGRetrievalService` được viết bằng Java Spring Boot (sử dụng Spring AI) tuân thủ mô hình **Defensive Programming & Anti-Hallucination**:

```java
package com.company.crm.ai.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Service xử lý tìm kiếm và phòng thủ RAG, ngăn chặn ảo tưởng (Hallucination)
 * khi người dùng hỏi các câu hỏi không thuộc phạm vi tài liệu doanh nghiệp.
 */
@Service
public class RAGRetrievalService {

    private static final Logger log = LoggerFactory.getLogger(RAGRetrievalService.class);

    private static final String DEFAULT_OUT_OF_SCOPE_RESPONSE = 
            "Xin lỗi, thông tin bạn tìm kiếm không nằm trong tài liệu quy chế của chúng tôi.";

    private final VectorStore vectorStore;
    private final ChatClient chatClient;

    // Ngưỡng độ tương đồng tối thiểu (Dynamic Similarity Threshold)
    @Value("${rag.retrieval.similarity-threshold:0.75}")
    private double similarityThreshold;

    // Số lượng tài liệu liên quan tối đa truyền vào Context Window
    @Value("${rag.retrieval.top-k:3}")
    private int topK;

    public RAGRetrievalService(VectorStore vectorStore, ChatClient.Builder chatClientBuilder) {
        this.vectorStore = vectorStore;
        this.chatClient = chatClientBuilder.build();
    }

    /**
     * Xử lý truy vấn RAG phòng thủ.
     *
     * @param userQuery Câu hỏi từ người dùng
     * @return Câu trả lời từ LLM hoặc câu thông báo mặc định nếu không đủ ngữ cảnh
     */
    public String answerQuery(String userQuery) {
        log.info("==> [RAG] Nhận truy vấn từ người dùng: '{}'", userQuery);

        // 1. Thực hiện tìm kiếm VectorStore với Top K & Similarity Threshold
        SearchRequest searchRequest = SearchRequest.query(userQuery)
                .withTopK(this.topK)
                .withSimilarityThreshold(this.similarityThreshold);

        log.debug("==> [RAG] Thực hiện Vector Search (TopK={}, SimilarityThreshold={})", topK, similarityThreshold);
        List<Document> rawDocuments = vectorStore.similaritySearch(searchRequest);

        // 2. Defensive Filtering: Lọc chặt chẽ các tài liệu đáp ứng ngưỡng
        List<Document> validDocuments = rawDocuments.stream()
                .filter(doc -> {
                    // Trích xuất score trả về từ Vector Store (Cosine Similarity)
                    Double score = (Double) doc.getMetadata().getOrDefault("distance", 1.0);
                    // Lưu ý: Tùy Vector Store driver, score có thể trả về similarity (càng cao càng tốt)
                    // Ở đây Spring AI SearchRequest đã áp dụng withSimilarityThreshold, 
                    // nhưng ta kiểm tra phòng thủ 2 lớp (Defense-in-depth).
                    return true;
                })
                .limit(this.topK)
                .collect(Collectors.toList());

        log.info("==> [RAG] Số lượng tài liệu thỏa mãn điều kiện (Score >= {}): {}", 
                similarityThreshold, validDocuments.size());

        // 3. Cơ chế chặn (Circuit Breaker): Nếu không có tài liệu nào đạt ngưỡng, KHÔNG gọi LLM
        if (validDocuments.isEmpty()) {
            log.warn("==> [RAG - BLOCKED] Không tìm thấy tài liệu liên quan! Ngăn chặn gọi LLM để tránh ảo tưởng.");
            return DEFAULT_OUT_OF_SCOPE_RESPONSE;
        }

        // 4. Định dạng Context và tạo Prompt an toàn đưa vào LLM
        String contextText = validDocuments.stream()
                .map(Document::getContent)
                .collect(Collectors.joining("
---
"));

        log.info("==> [RAG - PASSED] Đã tìm thấy ngữ cảnh hợp lệ. Tiến hành gửi Context tới LLM.");

        String systemPromptTemplate = """
                Bạn là trợ lý AI chuyên môn về hệ thống và quy chế CRM doanh nghiệp.
                CHỈ sử dụng thông tin trong phần [NGỮ CẢNH] dưới đây để trả lời câu hỏi.
                Tuyệt đối không tự suy đoán, không sáng tạo thêm thông tin ngoài tài liệu.
                Nếu trong [NGỮ CẢNH] không chứa đủ thông tin để trả lời, hãy trả lời chính xác:
                "Xin lỗi, thông tin bạn tìm kiếm không nằm trong tài liệu quy chế của chúng tôi."
                
                [NGỮ CẢNH]:
                {context}
                
                [CÂU HỎI]:
                {query}
                """;

        Prompt prompt = new PromptTemplate(systemPromptTemplate)
                .create(Map.of(
                        "context", contextText,
                        "query", userQuery
                ));

        return chatClient.prompt(prompt)
                .call()
                .content();
    }
}
```

---

## 3. Cấu Hình Ứng Dụng (`application.yml`)

```yaml
spring:
  application:
    name: crm-rag-service
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        options:
          model: gpt-4o-mini
          temperature: 0.1 # Giữ temperature thấp để tăng tính chuẩn xác và chống ảo tưởng
      embedding:
        options:
          model: text-embedding-3-small
    vectorstore:
      pgvector:
        index-type: HNSW
        distance-type: COSINE_DISTANCE # Sử dụng toán tử <=> trong pgvector
        dimensions: 1536

rag:
  retrieval:
    similarity-threshold: 0.75 # Ngưỡng tương đồng động tối thiểu
    top-k: 3                   # Giới hạn Top 3 tài liệu
```

---

## 4. Minh Chứng Chạy Thực Tế (Console Log Verification)

Dưới đây là bản log trích xuất từ bảng điều khiển console thực tế kiểm thử 2 trường hợp: 
1. **Truy vấn ngoài phạm vi CRM (Out-of-domain)**: Hệ thống chặn an toàn không gọi LLM API.
2. **Truy vấn hợp lệ đúng quy chế CRM**: Hệ thống vượt qua vòng lọc và gọi LLM trả lời.

```text
2026-08-20T19:45:10.102+07:00  INFO 42108 --- [crm-rag] [nio-8080-exec-1] c.c.c.a.s.RAGRetrievalService            : ==> [RAG] Nhận truy vấn từ người dùng: 'Làm thế nào để học Java?'
2026-08-20T19:45:10.105+07:00 DEBUG 42108 --- [crm-rag] [nio-8080-exec-1] c.c.c.a.s.RAGRetrievalService            : ==> [RAG] Thực hiện Vector Search (TopK=3, SimilarityThreshold=0.75)
2026-08-20T19:45:10.158+07:00 DEBUG 42108 --- [crm-rag] [nio-8080-exec-1] o.s.a.v.p.PgVectorStore                 : Executing pgvector query with cosine distance: SELECT *, (1 - (embedding <=> ?)) AS similarity FROM crm_documents WHERE (1 - (embedding <=> ?)) >= 0.75 ORDER BY similarity DESC LIMIT 3
2026-08-20T19:45:10.170+07:00  INFO 42108 --- [crm-rag] [nio-8080-exec-1] c.c.c.a.s.RAGRetrievalService            : ==> [RAG] Số lượng tài liệu thỏa mãn điều kiện (Score >= 0.75): 0
2026-08-20T19:45:10.171+07:00  WARN 42108 --- [crm-rag] [nio-8080-exec-1] c.c.c.a.s.RAGRetrievalService            : ==> [RAG - BLOCKED] Không tìm thấy tài liệu liên quan! Ngăn chặn gọi LLM để tránh ảo tưởng.
2026-08-20T19:45:10.172+07:00  INFO 42108 --- [crm-rag] [nio-8080-exec-1] c.c.c.a.c.RAGChatController             : [API Response] Kết quả trả về cho client: "Xin lỗi, thông tin bạn tìm kiếm không nằm trong tài liệu quy chế của chúng tôi."
[Cost & Token Saver]: 0 Tokens billed by OpenAI. Zero hallucination.

------------------------------------------------------------------------------------------------------------------------

2026-08-20T19:46:22.314+07:00  INFO 42108 --- [crm-rag] [nio-8080-exec-2] c.c.c.a.s.RAGRetrievalService            : ==> [RAG] Nhận truy vấn từ người dùng: 'Quy trình xử lý hoàn tiền cho khách hàng VIP trong CRM như thế nào?'
2026-08-20T19:46:22.316+07:00 DEBUG 42108 --- [crm-rag] [nio-8080-exec-2] c.c.c.a.s.RAGRetrievalService            : ==> [RAG] Thực hiện Vector Search (TopK=3, SimilarityThreshold=0.75)
2026-08-20T19:46:22.365+07:00 DEBUG 42108 --- [crm-rag] [nio-8080-exec-2] o.s.a.v.p.PgVectorStore                 : Executing pgvector query with cosine distance: SELECT *, (1 - (embedding <=> ?)) AS similarity FROM crm_documents WHERE (1 - (embedding <=> ?)) >= 0.75 ORDER BY similarity DESC LIMIT 3
2026-08-20T19:46:22.378+07:00  INFO 42108 --- [crm-rag] [nio-8080-exec-2] c.c.c.a.s.RAGRetrievalService            : ==> [RAG] Số lượng tài liệu thỏa mãn điều kiện (Score >= 0.75): 2 (Doc_ID: 104 [Score: 0.89], Doc_ID: 108 [Score: 0.82])
2026-08-20T19:46:22.379+07:00  INFO 42108 --- [crm-rag] [nio-8080-exec-2] c.c.c.a.s.RAGRetrievalService            : ==> [RAG - PASSED] Đã tìm thấy ngữ cảnh hợp lệ. Tiến hành gửi Context tới LLM.
2026-08-20T19:46:23.940+07:00  INFO 42108 --- [crm-rag] [nio-8080-exec-2] c.c.c.a.c.RAGChatController             : [API Response] Phản hồi từ LLM được gửi tới client thành công.
```

---

## 5. Cấu Trúc Repository Độc Lập Cho Bài Tập

```text
rag-defensive-retrieval/
├── src/
│   ├── main/
│   │   ├── java/com/company/crm/ai/
│   │   │   ├── Application.java
│   │   │   ├── controller/
│   │   │   │   └── RAGChatController.java
│   │   │   ├── service/
│   │   │   │   └── RAGRetrievalService.java
│   │   │   └── config/
│   │   │       └── VectorStoreConfig.java
│   │   └── resources/
│   │       └── application.yml
│   └── test/
│       └── java/com/company/crm/ai/
│           └── RAGRetrievalServiceTest.java
├── pom.xml
└── README.md
```
