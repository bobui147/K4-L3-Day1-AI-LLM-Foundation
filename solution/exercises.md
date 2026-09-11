# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng dần từ 0.0 đến 1.5, phản hồi trở nên đa dạng và sáng tạo hơn. Với temperature 0.0, model luôn cho câu trả lời giống nhau (deterministic), thường là sự thật phổ biến nhất. Khi temperature cao hơn (1.0-1.5), model có thể đưa ra các sự thật ít phổ biến hơn, đôi khi bất ngờ hơn nhưng vẫn đúng sự thật.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.3-0.5 cho chatbot hỗ trợ khách hàng. Temperature thấp (gần 0) giúp câu trả lời nhất quán, chính xác và đáng tin cậy - rất quan trọng cho thông tin hỗ trợ khách hàng. Tuy nhiên, không đặt hoàn toàn 0.0 để tránh câu trả lời quá cứng nhắc và giữ một chút tự nhiên trong giao tiếp.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10,000 users × 3 calls × 350 tokens output = 10.5M tokens/ngày. Chi phí output: GPT-4o ($0.010/1K) = $105/ngày, GPT-4o-mini ($0.0006/1K) = $6.3/ngày. GPT-4o đắt hơn khoảng 16.7 lần. Trường hợp xứng đáng dùng GPT-4o: phân tích hợp đồng pháp lý hoặc tư vấn y tế, nơi độ chính xác tuyệt đối là yêu cầu hàng đầu và sai sót có thể gây hậu quả nghiêm trọng. Trường hợp nên dùng mini: chatbot chào hỏi khách hàng, tóm tắt email, hoặc trả lời FAQ đơn giản - nơi độ chính xác "đủ tốt" là được chấp nhận.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi với persona giáo viên tiểu học sử dụng từ ngữ đơn giản, ví dụ cụ thể gần gũi (như "sổ ghi chép mà ai cũng có thể đọc nhưng không ai sửa được"), câu văn ngắn và dễ hiểu. Ngược lại, phản hồi từ chuyên gia tài chính sử dụng thuật ngữ chuyên ngành (cryptographic hash, distributed ledger, consensus mechanism), câu văn dài và chi tiết hơn. System prompt đóng vai trò như "hướng dẫn vai diễn" cho model, định hình không chỉ nội dung mà cả phong cách, độ phức tạp và cách tiếp cận vấn đề. Điều này cho thấy persona là công cụ mạnh mẽ để điều chỉnh output phù hợp với đối tượng người dùng mục tiêu.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, tiktoken thường đếm khoảng 180-220 tokens, trong khi ước lượng 100/0.75 ≈ 133 tokens, chênh lệch khoảng 35-65%. Tiếng Việt tốn nhiều token hơn vì tokenizer của OpenAI được train chủ yếu trên corpus tiếng Anh. Các từ tiếng Việt (đặc biệt là từ có dấu) thường bị tách thành nhiều token nhỏ hơn hoặc được mã hóa bằng byte-level tokens thay vì word-level tokens như tiếng Anh. Ví dụ, "blockchain" là 1 token nhưng "chuỗi khối" có thể là 3-4 tokens.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng chat/chatbot tương tác trực tiếp với người dùng, đặc biệt khi câu trả lời dài (>500 tokens). Người dùng thấy text xuất hiện dần ngay lập tức thay vì phải chờ 5-10 giây mới thấy gì, tạo cảm giác "AI đang suy nghĩ và trả lời" tự nhiên hơn. Non-streaming phù hợp hơn khi: (1) xử lý batch/background jobs không cần phản hồi tức thì, (2) cần parse toàn bộ JSON response trước khi dùng (streaming có thể gặp lỗi giữa chừng), (3) API calls trong pipeline automation nơi latency không phải vấn đề quan trọng, hoặc (4) khi cần log/audit toàn bộ response một lần.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm tải cho server đang quá tải bằng cách tăng khoảng cách giữa các lần retry (0.1s → 0.2s → 0.4s → 0.8s...), cho server thời gian phục hồi. Với delay cố định, nếu 10,000 clients cùng gặp lỗi và đồng loạt retry sau đúng 1 giây, chúng sẽ tạo ra "thundering herd" - 10,000 requests đổ cùng lúc vào server đang yếu, khiến tình trạng quá tải còn tệ hơn. Exponential backoff phân tán các retry request theo thời gian: một số client retry sớm (0.1s), một số muộn hơn (0.4s, 0.8s), giảm nguy cơ "bão request" và tăng khả năng thành công.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: "Bạn là trợ lý AI thân thiện và chuyên nghiệp. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt. Nếu không chắc chắn, hãy thừa nhận thay vì đoán. Ưu tiên thông tin chính xác hơn là dài dòng." Lựa chọn "trả lời ngắn gọn" vì trong CLI môi trường terminal, text dài gây khó đọc và người dùng thường cần câu trả lời nhanh, không phải bài luận. Chỉ định "tiếng Việt" rõ ràng vì model có xu hướng trả lời tiếng Anh nếu không được yêu cầu cụ thể. Câu "thừa nhận khi không chắc" quan trọng để tránh hallucination - model không nên bịa đặt thông tin sai lệch.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ lưu 3 lượt cuối (6 messages), khiến trợ lý "quên" ngữ cảnh từ đầu cuộc trò chuyện. Cải thiện: thêm "summarization layer" - sau mỗi 5 lượt, tự động tóm tắt 3 lượt cũ nhất thành 1-2 câu ngắn gọn, lưu tóm tắt này vào đầu history thay vì xóa hẳn. Cách triển khai: khi `len(history) > 6`, gọi API với prompt đặc biệt "Tóm tắt cuộc hội thoại sau thành 1-2 câu: [3 lượt cũ nhất]", sau đó thay thế 6 messages cũ bằng 1 message role="system" chứa tóm tắt. Chi phí token tăng chút nhưng giữ được맥락 dài hạn quan trọng.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
