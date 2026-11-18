import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import ChatMessage from './ChatMessage'
import MoleculeViewer from './MoleculeViewer'
import SourcesList from './SourcesList'

const API_URL = '/api'

const EXAMPLES = [
  { question: "What is the mechanism of action of aspirin?", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { question: "What are EGFR inhibitors used for?", smiles: "" },
  { question: "Tell me about acetaminophen", smiles: "CC(=O)Nc1ccc(O)cc1" },
]

export default function ChatInterface() {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [smiles, setSmiles] = useState('')
  const [loading, setLoading] = useState(false)
  const [currentSources, setCurrentSources] = useState([])
  const [currentMolecule, setCurrentMolecule] = useState(null)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!question.trim()) return

    const userMessage = {
      role: 'user',
      content: question,
      smiles: smiles || null
    }

    setMessages(prev => [...prev, userMessage])
    setLoading(true)
    setQuestion('')

    try {
      const response = await axios.post(`${API_URL}/query`, {
        question: question,
        smiles: smiles || null,
        num_results: 5,
        temperature: 0.7
      })

      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer
      }

      setMessages(prev => [...prev, assistantMessage])
      setCurrentSources(response.data.sources || [])
      setCurrentMolecule(response.data.molecule_info)

    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        role: 'assistant',
        content: `❌ Error: ${error.response?.data?.detail || error.message}`
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleExample = (example) => {
    setQuestion(example.question)
    setSmiles(example.smiles)
  }

  const handleClear = () => {
    setMessages([])
    setCurrentSources([])
    setCurrentMolecule(null)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Main Chat Area */}
      <div className="lg:col-span-2">
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          {/* Info Banner */}
          <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-4">
            <h2 className="text-lg font-semibold">Ask anything about chemistry</h2>
            <p className="text-sm text-blue-100 mt-1">
              Powered by: Graphormer (GNN) • RAG (58K docs) • Cross-Attention Fusion • TinyLlama
            </p>
          </div>

          {/* Messages */}
          <div className="h-[500px] overflow-y-auto p-6 space-y-4 bg-gray-50">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 mt-20">
                <div className="text-6xl mb-4">💬</div>
                <p className="text-lg">Start a conversation</p>
                <p className="text-sm mt-2">Try one of the examples below</p>
              </div>
            )}

            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))}

            {loading && (
              <div className="flex items-center space-x-2 text-gray-500">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
                <span>Thinking...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-4 bg-white">
            <form onSubmit={handleSubmit} className="space-y-3">
              <div>
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a chemistry question..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows="3"
                  disabled={loading}
                />
              </div>

              <div className="flex space-x-3">
                <input
                  type="text"
                  value={smiles}
                  onChange={(e) => setSmiles(e.target.value)}
                  placeholder="Optional: Molecule SMILES (e.g., CCO)"
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={loading}
                />
                <button
                  type="submit"
                  disabled={loading || !question.trim()}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {loading ? 'Sending...' : 'Send'}
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Clear
                </button>
              </div>
            </form>

            {/* Examples */}
            <div className="mt-4">
              <p className="text-sm text-gray-600 mb-2">Quick examples:</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLES.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => handleExample(example)}
                    className="text-sm px-3 py-1 bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100 transition-colors"
                  >
                    {example.question.substring(0, 40)}...
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className="lg:col-span-1 space-y-6">
        {/* Molecule Viewer */}
        {currentMolecule && (
          <MoleculeViewer molecule={currentMolecule} />
        )}

        {/* Sources */}
        {currentSources.length > 0 && (
          <SourcesList sources={currentSources} />
        )}

        {/* Info Card */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="font-semibold text-gray-900 mb-3">About Helios</h3>
          <div className="text-sm text-gray-600 space-y-2">
            <p>Multimodal AI for chemistry research</p>
            <div className="space-y-1">
              <div className="flex items-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                <span>GNN: Graphormer</span>
              </div>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                <span>RAG: 58,996 docs</span>
              </div>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
                <span>LLM: TinyLlama-1.1B</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
