import os
import chainlit as cl
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv
import numpy as np

load_dotenv()
# Log in to HuggingFace (only if needed for private models)
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Load the Vietnamese-compatible sentence transformer
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

# Data
conversation_pairs = [
    {
        "query": "Chào bạn",
        "answer": "Chào Thiên Long, chúc bạn buổi sáng vui vẻ! Mình có thể giúp gì cho bạn hôm nay?"
    },
    {
        "query": "Bạn có thể làm được những gì vậy?",
        "answer": "Mình là một hệ thống AI Agent với các khả năng như:\n\n• Tư vấn tuyển sinh cho Trường Đại học Bách Khoa – ĐHQG-HCM\n• Dịch tiếng Việt sang tiếng Bahnar\n• Hỗ trợ thay đồ ảo trên ảnh của bạn"
    },
    {
        "query": "Trường Đại học Bách Khoa có những phương thức tuyển sinh nào?",
        "answer": "Trường hiện đang áp dụng các phương thức tuyển sinh như sau:\n\n1. Phương thức 1 (TTBO): Xét tuyển thẳng theo quy chế của Bộ GD&ĐT, chiếm khoảng 1% – 5% tổng chỉ tiêu.\n\n2. Phương thức 2 (THOP): Xét tuyển tổng hợp, chiếm khoảng 95% – 99% tổng chỉ tiêu, gồm 5 nhóm đối tượng:\n\n- Đối tượng 1: Thí sinh có kết quả kỳ thi Đánh giá năng lực ĐHQG-HCM năm 2025\n- Đối tượng 2: Thí sinh không có kết quả kỳ thi Đánh giá năng lực\n- Đối tượng 3: Thí sinh tốt nghiệp THPT ở nước ngoài\n- Đối tượng 4: Thí sinh sử dụng chứng chỉ tuyển sinh quốc tế\n- Đối tượng 5: Thí sinh đăng ký chương trình chuyển tiếp quốc tế (Úc, Mỹ, New Zealand)\n\nLưu ý:\n\n• Thí sinh có thể tham gia xét tuyển theo nhiều đối tượng và sẽ được lấy điểm cao nhất để xét tuyển.\n• Nếu có chứng chỉ IELTS ≥ 5.0, TOEFL iBT ≥ 46 hoặc TOEIC đủ điều kiện, thí sinh sẽ được quy đổi điểm môn tiếng Anh tương ứng trong tổ hợp xét tuyển."
    },
    {
        "query": "Trường hè khoa học CSE – Summer School 2025 có chủ đề gì vậy?",
        "answer": "Trường hè năm nay sẽ tập trung vào chủ đề AI Agents. Chương trình dự kiến diễn ra vào đầu tháng 7, dành cho các bạn học sinh THPT quan tâm đến trí tuệ nhân tạo và ngành Khoa học Máy tính."
    },
    {
        "query": "Dịch tiếp câu này nha: 'Trường hè khoa học 2025 sẽ về chủ đề AI Agents.'",
        "answer": "Trương kơtiêng khoa hŏk 2025 liêm sưt kră ŭnh đê tak đon trĭ tuê nhân tao."
    },
    {
        "query": 'Bạn dịch giúp mình câu này sang tiếng Bahnar nhé: "Tôi tên Long, tôi là sinh viên năm 2 ngành Khoa học Máy tính tại Trường Đại học Bách Khoa."',
        "answer": "Inh tên Long, inh la sinh viên sònăm 2 ngành khoa hŏk mồn học máy hmắi lơm Trường Đại học Bách Khoa."
    },
    {
        "query": "Thay giúp tui ảnh của người mẫu này với bộ đồ tui gửi nha.",
        "answer": ""
    },
    {
        "query": "Cảm ơn bạn nha.",
        "answer": "Không có gì bạn. Bạn nhớ đăng ký tham gia CSE – Summer School 2025 nha. 😊 Tạm biệt thiên long!"
    }
]

# Pre-compute embeddings for all queries
query_embeddings = []
for pair in conversation_pairs:
    embedding = model.encode(pair["query"])
    query_embeddings.append({
        "embedding": embedding,
        "answer": pair["answer"],
        "query": pair["query"]
    })


def semantic_search(query, threshold=0.7):
    # Encode the query
    query_embedding = model.encode(query)

    # Find the most similar query
    max_similarity = -1
    best_answer = None

    for item in query_embeddings:
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, item["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
        )

        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = item["answer"]

    if max_similarity >= threshold:
        return best_answer
    return "Tôi không có thông tin về câu hỏi này. Bạn có thể hỏi câu khác được không?"


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="👋 Xin chào! Tôi là AI Agent của Trường Đại học Bách Khoa. Bạn muốn hỏi gì?").send()


@cl.on_message
async def on_message(message: cl.Message):
    time.sleep(3)
    query = message.content

    # Check if there are exactly 2 images attached
    if len(message.elements) == 2 and all(element.type == "image" for element in message.elements):
        time.sleep(7)
        # User sent text with 2 images, respond with the answer.jpg image
        answer_image_path = "answer.jpg"

        if os.path.exists(answer_image_path):
            # Respond with the processed image
            elements = [
                cl.Image(
                    name="processed_image",
                    display="inline",
                    path=answer_image_path
                )
            ]
            await cl.Message(content="Đây là kết quả thay đồ ảo cho bạn:", elements=elements).send()
        else:
            await cl.Message(content="Không tìm thấy file ảnh kết quả. Vui lòng thử lại.").send()
    else:
        # Regular text-based response
        answer = semantic_search(query)

        # Typing effect (streaming token by token)
        msg = cl.Message(content="")
        await msg.send()

        for token in answer.split():
            await msg.stream_token(token + " ")
            time.sleep(0.03)  # typing speed simulation

        msg.content = answer
        await msg.update()