import { create } from 'zustand'

// Chat Store - manages all application state
const useChatStore = create((set, get) => ({
  // Sessions
  sessions: [],
  currentSessionId: null,
  
  // Messages
  messages: [],
  isStreaming: false,
  streamingContent: '',
  
  // UI State
  sidebarOpen: true,
  activeTab: 'chat', // chat, molecules, documents, drugDiscovery
  rightPanel: 'auto', // auto, admet, drugLikeness, alerts, pharmacophores, optimization, figures
  
  // Documents
  documents: [],
  
  // Current molecule/sources
  currentMolecule: null,
  currentSources: [],
  moleculeSvg: null,
  moleculeProperties: null,
  inlineMolecules: [],  // molecules detected in current answer for inline display
  
  // Drug Discovery data
  drugDiscovery: null,  // Full drug discovery report for current molecule
  figures: {},          // Textbook-style figures {name: base64DataUri}
  
  // Actions
  setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),
  
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, { ...message, id: Date.now(), timestamp: new Date().toISOString() }]
  })),
  
  setMessages: (messages) => set({ messages }),
  
  setStreaming: (isStreaming, content = '') => set({ isStreaming, streamingContent: content }),
  
  appendStreamContent: (content) => set((state) => ({
    streamingContent: state.streamingContent + content
  })),
  
  setSessions: (sessions) => set({ sessions }),
  
  setDocuments: (documents) => set({ documents }),
  
  setCurrentMolecule: (molecule, svg, properties) => set({
    currentMolecule: molecule,
    moleculeSvg: svg,
    moleculeProperties: properties
  }),
  
  setCurrentSources: (sources) => set({ currentSources: sources }),
  
  setInlineMolecules: (molecules) => set({ inlineMolecules: molecules }),
  
  setDrugDiscovery: (data) => set({ drugDiscovery: data }),
  
  setFigures: (figures) => set({ figures: figures || {} }),
  
  setRightPanel: (panel) => set({ rightPanel: panel }),
  
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  
  setActiveTab: (tab) => set({ activeTab: tab }),
  
  clearChat: () => set({
    messages: [],
    currentMolecule: null,
    currentSources: [],
    moleculeSvg: null,
    moleculeProperties: null,
    inlineMolecules: [],
    drugDiscovery: null,
    figures: {},
  }),
  
  // Create new session
  createSession: async () => {
    try {
      const response = await fetch('/api/sessions', { method: 'POST' })
      const data = await response.json()
      set({ currentSessionId: data.session_id, messages: [] })
      return data.session_id
    } catch (error) {
      console.error('Failed to create session:', error)
      return null
    }
  },
  
  // Load session
  loadSession: async (sessionId) => {
    try {
      const response = await fetch(`/api/sessions/${sessionId}`)
      const data = await response.json()
      set({
        currentSessionId: sessionId,
        messages: data.messages || [],
        documents: data.documents || []
      })
    } catch (error) {
      console.error('Failed to load session:', error)
    }
  },
  
  // Fetch all sessions
  fetchSessions: async () => {
    try {
      const response = await fetch('/api/sessions')
      const data = await response.json()
      set({ sessions: data.sessions || [] })
    } catch (error) {
      console.error('Failed to fetch sessions:', error)
    }
  }
}))

export default useChatStore
