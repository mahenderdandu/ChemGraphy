export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-lg px-4 py-3 ${
        isUser 
          ? 'bg-blue-600 text-white' 
          : 'bg-white text-gray-900 shadow-md'
      }`}>
        <div className="flex items-start space-x-2">
          <div className="text-2xl">
            {isUser ? '👤' : '🤖'}
          </div>
          <div className="flex-1">
            <div className="text-sm font-semibold mb-1">
              {isUser ? 'You' : 'ChemGraphy'}
            </div>
            <div className="text-sm whitespace-pre-wrap">
              {message.content}
            </div>
            {message.smiles && (
              <div className="mt-2 px-3 py-1 bg-gray-100 rounded text-xs font-mono">
                🧪 {message.smiles}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
