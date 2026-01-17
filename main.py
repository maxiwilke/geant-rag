"""
CLI Chatbot - Terminal interface for testing
Uses rag_module.py for core functionality
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage
from rag_module import create_rag_chain, MAX_HISTORY_MESSAGES

if __name__ == "__main__":
    # Check for rebuild flag
    rebuild = "--rebuild" in sys.argv
    
    print("Initializing RAG Chatbot...")
    rag_chain = create_rag_chain(rebuild=rebuild)
    
    # Initialize chat history
    chat_history = []
    
    print("\nChatbot ready! (type 'q' to quit, 'clear' to clear history)\n")
    print("Tip: Run with --rebuild flag to force database rebuild\n")
    
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
        
        # Add to chat history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        
        # Keep only last N messages
        if len(chat_history) > MAX_HISTORY_MESSAGES:
            chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
        
        print("\n--- Sources ---")
        if docs:
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                source = source.replace('\\', '/').split('/')[-1]
                print(f"{i}. {source}")
        else:
            print("No sources found")
        
        # Show current history count
        exchanges = len(chat_history) // 2
        print(f"\n(Chat history: {exchanges} exchange(s) stored)")