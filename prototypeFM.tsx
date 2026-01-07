import { useState, useRef, useEffect } from 'react';
import { Send, RotateCcw, X } from 'lucide-react';
import headerLogo from 'figma:asset/64af546cfa0dec8b131a482647de406152177ed6.png';


interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isOpen, setIsOpen] = useState(true);
  const [isThinking, setIsThinking] = useState(false);
  const [isLoadingLink, setIsLoadingLink] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking]);

  useEffect(() => {
    // Add welcome message on mount
    const welcomeMessage: Message = {
      id: '1',
      text: "Hello! 👋 I am the GÉANT chatbot, your AI helper.",
      sender: 'bot',
      timestamp: new Date()
    };
    const followUpMessage: Message = {
      id: '2',
      text: "How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    };
    setMessages([welcomeMessage, followUpMessage]);
  }, []);

  const handleSendMessage = () => {
    if (inputValue.trim() === '') return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const userQuestion = inputValue.toLowerCase();
    setInputValue('');
    setIsThinking(true);

    // Check if question is off-topic (doesn't contain GÉANT-related keywords!)
    const geantKeywords = ['géant', 'geant', 'surf', 'university', 'universities', 'network', 'research', 'education', 'report', 'case study', 'annual report', 'collaborate', 'collaboration'];
    const isOffTopic = !geantKeywords.some(keyword => userQuestion.includes(keyword));

    // Simulate bot response
    setTimeout(() => {
      let responseText = '';
      
      if (isOffTopic) {
        responseText = "I'm your GÉANT assistant, so I can only answer questions about GÉANT and related topics.";
      } else {
        responseText = "Of course. The SURF case study highlights in detail how the organization undertook a broad modernization of its digital infrastructure to better support the evolving needs of research and higher education across the Netherlands. It begins by outlining the increasing complexity within the academic landscape, where institutions were relying on fragmented systems, inconsistent data workflows, and outdated technologies that made collaboration difficult. These challenges became especially visible as universities faced rapidly growing data volumes, rising expectations for digital reliability, and an expanding set of research disciplines that required advanced computational capabilities.\n\nThe case study explains that SURF recognized the necessity of creating a more unified, resilient, and scalable infrastructure. To achieve this, they developed a strategy focused on centralizing essential services, standardizing technological frameworks across institutions, and improving the security architecture to handle sensitive academic and research data. A major emphasis was placed on fostering collaboration, not only through technical interoperability but also through shared governance models that allowed universities and research centers to jointly shape digital services.\n\nFurthermore, the study describes how SURF invested in cloud-based solutions, high-performance networking, and automated resource management to ensure that institutions could access computing power and data tools more efficiently. Training initiatives and knowledge-sharing programs were also integrated into the modernization effort to help universities adapt to the new systems and make full use of the digital capabilities being introduced.\n\nThe results section of the study highlights several key outcomes: significantly faster data processing for research projects, more efficient allocation of computational resources, and improved accessibility to digital tools for both educators and students. Institutions reported greater reliability in their digital infrastructures, reduced redundancy in system maintenance, and more opportunities to collaborate on shared datasets and interdisciplinary research programs.\n\nOverall, the case study illustrates how coordinated innovation—spanning technology, governance, and community engagement—can strengthen the digital ecosystem of higher education. It demonstrates that a unified approach not only improves operational efficiency but also creates a foundation for long-term academic innovation and nationwide collaboration.";
      }
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: responseText,
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
      setIsThinking(false);
    }, 7000);
  };

  const handleRecommendationClick = (question: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      text: question,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsThinking(true);

    // Simulate bot response
    setTimeout(() => {
      let responseText = '';
      
      if (question === "Please summarize the SURF case study for me") {
        responseText = "Of course. The SURF case study highlights how the organization modernized its digital infrastructure to better support research and education across the Netherlands. It explains the challenges SURF faced with fragmented systems and growing data needs, and describes the strategy they used to centralize services, improve security, and enhance collaboration between universities. The study also outlines key results, such as faster data processing, more efficient resource allocation, and improved accessibility for academic institutions. Overall, it illustrates how coordinated innovation can strengthen digital ecosystems in higher education.";
      } else if (question === "Show me the Annual Report of 2020") {
        responseText = "Of course. The SURF case study highlights in detail how the organization undertook a broad modernization of its digital infrastructure to better support the evolving needs of research and higher education across the Netherlands. It begins by outlining the increasing complexity within the academic landscape, where institutions were relying on fragmented systems, inconsistent data workflows, and outdated technologies that made collaboration difficult. These challenges became especially visible as universities faced rapidly growing data volumes, rising expectations for digital reliability, and an expanding set of research disciplines that required advanced computational capabilities.\n\nThe case study explains that SURF recognized the necessity of creating a more unified, resilient, and scalable infrastructure. To achieve this, they developed a strategy focused on centralizing essential services, standardizing technological frameworks across institutions, and improving the security architecture to handle sensitive academic and research data. A major emphasis was placed on fostering collaboration, not only through technical interoperability but also through shared governance models that allowed universities and research centers to jointly shape digital services.\n\nFurthermore, the study describes how SURF invested in cloud-based solutions, high-performance networking, and automated resource management to ensure that institutions could access computing power and data tools more efficiently. Training initiatives and knowledge-sharing programs were also integrated into the modernization effort to help universities adapt to the new systems and make full use of the digital capabilities being introduced.\n\nThe results section of the study highlights several key outcomes: significantly faster data processing for research projects, more efficient allocation of computational resources, and improved accessibility to digital tools for both educators and students. Institutions reported greater reliability in their digital infrastructures, reduced redundancy in system maintenance, and more opportunities to collaborate on shared datasets and interdisciplinary research programs.\n\nOverall, the case study illustrates how coordinated innovation—spanning technology, governance, and community engagement—can strengthen the digital ecosystem of higher education. It demonstrates that a unified approach not only improves operational efficiency but also creates a foundation for long-term academic innovation and nationwide collaboration.";
      } else if (question === "How many universities does GEANT collaborate with?") {
        responseText = "GEANT collaborates with over 100 universities across Europe, providing them with high-speed network connectivity to support research, education, and collaboration.";
      }
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: responseText,
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
      setIsThinking(false);
    }, 7000);
  };

  const handleRestart = () => {
    const welcomeMessage: Message = {
      id: Date.now().toString(),
      text: "Hello! 👋 I am the GÉANT chatbot, your AI helper.",
      sender: 'bot',
      timestamp: new Date()
    };
    const followUpMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: "How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    };
    setMessages([welcomeMessage, followUpMessage]);
    setInputValue('');
  };

  const handleClose = () => {
    setIsOpen(false);
  };

  const handleSourceClick = () => {
    setIsLoadingLink(true);
    // Return to chat after 5 seconds
    setTimeout(() => {
      setIsLoadingLink(false);
    }, 5000);
  };

  if (isLoadingLink) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
            .spinner {
              animation: spin 1s linear infinite;
            }
          `}</style>
          <div 
            className="spinner w-16 h-16 border-4 rounded-full" 
            style={{ 
              borderColor: '#E5E7EB',
              borderTopColor: '#810947'
            }}
          ></div>
          <p style={{ color: '#810947' }}>Redirect to source</p>
        </div>
      </div>
    );
  }

  if (!isOpen) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="w-[650px] bg-white rounded-lg shadow-lg p-8 text-center">
          <p className="text-gray-600 mb-4">Chat closed</p>
          <button
            onClick={() => setIsOpen(true)}
            className="text-white px-6 py-2 rounded-lg transition-colors"
            style={{ backgroundColor: '#810947' }}
          >
            Reopen Chat
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="w-[650px] bg-white rounded-2xl shadow-xl flex flex-col h-[1000px]">
        {/* Header */}
        <div className="bg-white p-4 rounded-t-2xl flex items-center justify-between border-b" style={{ borderColor: '#464646' }}>
          <button
            onClick={handleRestart}
            className="p-2 hover:opacity-80 rounded-lg transition-opacity"
            title="Restart chat"
            style={{ color: '#810947' }}
          >
            <RotateCcw className="w-7 h-7" />
          </button>
          <img src={headerLogo} alt="GÉANT Ai" className="h-12 mt-1" />
          <div className="flex items-center gap-3">
            <button
              onClick={handleClose}
              className="p-2 hover:opacity-80 rounded-lg transition-opacity"
              title="Close chat"
              style={{ color: '#810947' }}
            >
              <X className="w-8 h-8" />
            </button>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className="flex flex-col gap-2">
                <div
                  className="rounded-2xl px-4 py-2 text-lg inline-block"
                  style={{
                    backgroundColor: message.sender === 'user' ? '#ffd9e9' : '#FFFFFF',
                    color: message.sender === 'user' ? '#810947' : '#464646'
                  }}
                >
                  {message.text}
                </div>
                {/* Source bubble for bot SURF responses */}
                {message.sender === 'bot' && (message.text.includes('SURF case study highlights')) && (
                  <button
                    onClick={handleSourceClick}
                    className="px-4 py-2 rounded-full border-2 text-left text-lg inline-block w-fit"
                    style={{ backgroundColor: '#ffd9e9', borderColor: '#810947', color: '#810947' }}
                  >
                    Source: SURF Case Study 2021
                  </button>
                )}
              </div>
            </div>
          ))}

          {/* Recommendation Bubbles - show only when there are just welcome messages */}
          {messages.length <= 2 && (
            <div className="space-y-2">
              <div className="flex flex-col gap-2">
                <button
                  onClick={() => handleRecommendationClick("Please summarize the SURF case study for me")}
                  className="px-4 py-2 rounded-full border-2 transition-opacity hover:opacity-80 text-left text-lg inline-block w-fit"
                  style={{ backgroundColor: '#ffd9e9', borderColor: '#810947', color: '#810947' }}
                >
                  Please summarize the SURF case study for me
                </button>
                <button
                  onClick={() => handleRecommendationClick("Show me the Annual Report of 2020")}
                  className="px-4 py-2 rounded-full border-2 transition-opacity hover:opacity-80 text-left text-lg inline-block w-fit"
                  style={{ backgroundColor: '#ffd9e9', borderColor: '#810947', color: '#810947' }}
                >
                  Show me the Annual Report of 2020
                </button>
                <button
                  onClick={() => handleRecommendationClick("How many universities does GEANT collaborate with?")}
                  className="px-4 py-2 rounded-full border-2 transition-opacity hover:opacity-80 text-left text-lg inline-block w-fit"
                  style={{ backgroundColor: '#ffd9e9', borderColor: '#810947', color: '#810947' }}
                >
                  How many universities does GEANT collaborate with?
                </button>
              </div>
            </div>
          )}

          {/* Thinking Indicator */}
          {isThinking && (
            <div className="flex justify-start">
              <div className="rounded-2xl px-4 py-3 inline-flex items-center gap-1" style={{ backgroundColor: '#ffd9e9' }}>
                <style>{`
                  @keyframes bounce {
                    0%, 60%, 100% { transform: translateY(0); }
                    30% { transform: translateY(-8px); }
                  }
                  .dot-left {
                    animation: bounce 1s infinite;
                    animation-delay: 0.001s;
                  }
                  .dot-middle {
                    animation: bounce 1s infinite;
                    animation-delay: 0.18s;
                  }
                  .dot-right {
                    animation: bounce 1s infinite;
                    animation-delay: 0.36s;
                  }
                `}</style>
                <div className="dot-left w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#810947' }}></div>
                <div className="dot-middle w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#810947' }}></div>
                <div className="dot-right w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#810947' }}></div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t p-4" style={{ borderColor: '#464646' }}>
          <div className="flex gap-2 items-center">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Ask anything ..."
              className="flex-1 border border-gray-300 px-4 py-3 focus:outline-none focus:ring-2 text-lg transition-all duration-300"
              style={{ borderRadius: '100px', '--tw-ring-color': '#810947' } as React.CSSProperties}
            />
            <button
              onClick={handleSendMessage}
              className="text-white rounded-full flex items-center justify-center transition-all duration-300"
              style={{ 
                backgroundColor: '#810947',
                opacity: inputValue.trim() !== '' ? 1 : 0,
                transform: inputValue.trim() !== '' ? 'scale(1)' : 'scale(0.8)',
                pointerEvents: inputValue.trim() !== '' ? 'auto' : 'none',
                width: inputValue.trim() !== '' ? '48px' : '0px',
                height: '48px',
                padding: inputValue.trim() !== '' ? '12px' : '0px'
              }}
            >
              <Send className="w-5 h-5" style={{ transform: 'rotate(45deg) translateX(-1px)' }} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
