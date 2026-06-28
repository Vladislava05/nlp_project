import gradio as gr
from query_engine import RAGEngine

print("Загрузка RAGEngine...")
engine = RAGEngine()
print("Готово!")

def search(query, top_k, search_type):
    if not query:
        return "Введите вопрос"
    try:
        if search_type == "Гибридный (BM25 + Dense)":
            results = engine.hybrid_search(query, top_k=top_k)
        else:
            results = engine.search(query, top_k=top_k)  # обычный dense

        if not results:
            return "Ничего не найдено"

        output = ""
        for i, (text, meta, dist) in enumerate(results, 1):
            source = meta.get('source', 'Документ')
            if search_type == "Гибридный (BM25 + Dense)":
                output += f"### Результат {i} — {source} (RRF score: {dist:.4f})\n\n{text}\n\n---\n\n"
            else:
                output += f"### Результат {i} — {source} (расстояние: {dist:.4f})\n\n{text}\n\n---\n\n"
        return output
    except Exception as e:
        return f"Ошибка: {e}"

demo = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Ваш вопрос", placeholder="Например: что такое FAISS?", lines=2),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Количество результатов"),
        gr.Radio(choices=["Гибридный (BM25 + Dense)", "Только Dense"], label="Тип поиска", value="Гибридный (BM25 + Dense)")
    ],
    outputs=gr.Markdown(label="Ответ"),
    title=" RAG Q&A — Гибридный поиск",
    description="Гибридный поиск объединяет BM25 (ключевые слова) и Dense-эмбеддинги (смысл)."
)

if __name__ == "__main__":
    demo.launch()