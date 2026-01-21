"""
CLI Chatbot - Terminal interface for testing
Uses rag_module.py for core functionality
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage
from rag_module import create_rag_chain, MAX_HISTORY_MESSAGES

# Correct Ollama imports
from langchain_ollama import ChatOllama

if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv

    print("Initializing RAG Chatbot (local Ollama + local DB)...")

    # Local Ollama LLM
    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0.7
    )

    # HuggingFace embeddings (384 dimensions)
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Pass both LLM + embeddings into RAG
    rag_chain = create_rag_chain(
        llm=llm,
        embeddings=embeddings,
        rebuild=rebuild
    )

    chat_history = []

    print("\nChatbot ready! (type 'q' to quit, 'clear' to clear history)\n")

    while True:
        print("\n_______________________________")
        question = input("Ask your question: ").strip()

        if question.lower() == "q":
            print("Goodbye!")
            break

        if question.lower() == "clear":
            chat_history = []
            print("Chat history cleared!")
            continue

        if not question:
            continue

        print("\nSearching documents...\n")
        answer, docs = rag_chain(question, chat_history)

        print("Answer:")
        print(answer)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        if len(chat_history) > MAX_HISTORY_MESSAGES:
            chat_history = chat_history[-MAX_HISTORY_MESSAGES:]

        print("\n--- Sources ---")
        if docs:
            seen_sources = set()
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                source_name = source.replace("\\", "/").split("/")[-1]
                
                # Skip duplicates
                if source_name in seen_sources:
                    continue
                seen_sources.add(source_name)
                
                # Get URL if available
                url = doc.metadata.get("url")
                
                if url:
                    print(f"{len(seen_sources)}. {source_name}")
                    print(f"   URL: {url}")
                else:
                    print(f"{len(seen_sources)}. {source_name}")
        else:
            print("No sources found")

        exchanges = len(chat_history) // 2
        print(f"\n(Chat history: {exchanges} exchange(s) stored)")