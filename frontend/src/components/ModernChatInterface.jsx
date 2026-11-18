import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send, Sparkles, Loader2, Atom, FileText, Upload, 
  ChevronLeft, ChevronRight, Plus, MessageSquare, Clock,
  Beaker, FlaskConical, BookOpen, Settings, X, Menu,
  Zap, Brain, Search, ExternalLink, Copy, Check,
  Activity, Shield, Target, TrendingUp, BarChart3,
  AlertTriangle, ChevronDown, Pill, Microscope, Dna
} from 'lucide-react'
import useChatStore from '../store/chatStore'
import ReactMarkdown from 'react-markdown'

// Animated gradient background
const GradientBackground = () => (
  <div className="fixed inset-0 -z-10 overflow-hidden">
    <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />
    <div className="absolute top-0 -left-40 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl animate-pulse" />
    <div className="absolute bottom-0 -right-40 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse delay-1000" />
    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-500/10 rounded-full blur-3xl animate-pulse delay-500" />
  </div>
)

// Typing indicator
const TypingIndicator = () => (
  <div className="flex items-center gap-1.5 px-4 py-2">
    <motion.div
      className="w-2 h-2 bg-blue-400 rounded-full"
      animate={{ scale: [1, 1.2, 1] }}
      transition={{ repeat: Infinity, duration: 0.6, delay: 0 }}
    />
    <motion.div
      className="w-2 h-2 bg-blue-400 rounded-full"
      animate={{ scale: [1, 1.2, 1] }}
      transition={{ repeat: Infinity, duration: 0.6, delay: 0.2 }}
    />
    <motion.div
      className="w-2 h-2 bg-blue-400 rounded-full"
      animate={{ scale: [1, 1.2, 1] }}
      transition={{ repeat: Infinity, duration: 0.6, delay: 0.4 }}
    />
  </div>
)

// Message component with animations, inline molecule rendering, and textbook figures
const Message = ({ message, index }) => {
  const [copied, setCopied] = useState(false)
  const [expandedMols, setExpandedMols] = useState({})
  const [expandedFigures, setExpandedFigures] = useState({})
  const [showAllFigures, setShowAllFigures] = useState(true)
  const isUser = message.role === 'user'
  
  const copyToClipboard = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  
  const toggleMolExpand = (smiles) => {
    setExpandedMols(prev => ({ ...prev, [smiles]: !prev[smiles] }))
  }

  const toggleFigure = (key) => {
    setExpandedFigures(prev => ({ ...prev, [key]: !prev[key] }))
  }

  // Figure label mapping for textbook-style captions
  const figureCaptions = {
    molecule_structure_svg: { label: 'Figure — Molecular Structure', icon: Atom, color: 'emerald' },
    admet_radar: { label: 'Figure — ADMET Profile', icon: Activity, color: 'blue' },
    drug_likeness_scorecard: { label: 'Figure — Drug-Likeness Assessment', icon: Pill, color: 'purple' },
    bioavailability_radar: { label: 'Figure — Bioavailability Radar', icon: Target, color: 'teal' },
    structural_alerts: { label: 'Figure — Structural Alert Analysis', icon: AlertTriangle, color: 'amber' },
    pharmacophore_features: { label: 'Figure — Pharmacophore Map', icon: Microscope, color: 'pink' },
    optimization_priorities: { label: 'Figure — Lead Optimization', icon: TrendingUp, color: 'cyan' },
  }

  const colorMap = {
    emerald: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
    blue: 'text-blue-400 border-blue-500/30 bg-blue-500/10',
    purple: 'text-purple-400 border-purple-500/30 bg-purple-500/10',
    teal: 'text-teal-400 border-teal-500/30 bg-teal-500/10',
    amber: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
    pink: 'text-pink-400 border-pink-500/30 bg-pink-500/10',
    cyan: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/10',
  }

  // Render textbook-style figures
  const renderFigures = () => {
    const figures = message.figures
    if (!figures || Object.keys(figures).length === 0) return null

    const figureEntries = Object.entries(figures)
    let figNum = 1

    return (
      <div className="mt-5 space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
            <BarChart3 className="w-3.5 h-3.5" />
            Textbook Figures ({figureEntries.length})
          </h4>
          <button
            onClick={() => setShowAllFigures(!showAllFigures)}
            className="text-xs text-slate-500 hover:text-slate-300 transition-colors flex items-center gap-1"
          >
            {showAllFigures ? 'Collapse All' : 'Expand All'}
            <ChevronDown className={`w-3 h-3 transition-transform ${showAllFigures ? 'rotate-180' : ''}`} />
          </button>
        </div>

        {figureEntries.map(([key, dataUri]) => {
          const caption = figureCaptions[key] || { label: `Figure ${figNum}`, icon: BarChart3, color: 'slate' }
          const Icon = caption.icon
          const colors = colorMap[caption.color] || colorMap.emerald
          const isSvg = key.includes('_svg') && !dataUri.startsWith('data:image')
          const isExpanded = expandedFigures[key] !== false && showAllFigures

          const figLabel = `Fig. ${figNum}`
          figNum++

          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`border rounded-xl overflow-hidden ${colors.split(' ').slice(1).join(' ')}`}
            >
              {/* Caption header (clickable) */}
              <button
                onClick={() => toggleFigure(key)}
                className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-white/5 transition-colors"
              >
                <div className="flex items-center gap-2.5">
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${colors}`}>
                    {figLabel}
                  </span>
                  <Icon className={`w-4 h-4 ${colors.split(' ')[0]}`} />
                  <span className="text-sm font-semibold text-slate-200">{caption.label}</span>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
              </button>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="overflow-hidden"
                  >
                    <div className="px-3 pb-3">
                      {isSvg ? (
                        <div
                          className="bg-white rounded-lg p-3 flex justify-center [&>svg]:max-w-full [&>svg]:h-auto"
                          dangerouslySetInnerHTML={{ __html: dataUri }}
                        />
                      ) : (
                        <div className="rounded-lg overflow-hidden flex justify-center">
                          <img
                            src={dataUri}
                            alt={caption.label}
                            className="max-w-full h-auto rounded-lg"
                            loading="lazy"
                          />
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )
        })}
      </div>
    )
  }

  // Render drug discovery summary badges
  const renderDrugDiscoveryBadges = () => {
    const dd = message.drugDiscovery
    if (!dd) return null

    const admet = dd.admet
    const dl = dd.drug_likeness
    const alerts = dd.structural_alerts
    const sa = dd.synthetic_accessibility

    return (
      <div className="mt-4 flex flex-wrap gap-2">
        {dl && (
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${
            dl.qed_score >= 0.5 ? 'bg-green-500/20 text-green-300' : 'bg-amber-500/20 text-amber-300'
          }`}>
            <Pill className="w-3 h-3" />
            QED: {dl.qed_score?.toFixed(2)}
          </span>
        )}
        {dl && (
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${
            dl.lipinski?.pass ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
          }`}>
            <Target className="w-3 h-3" />
            Lipinski: {dl.lipinski?.pass ? 'PASS' : 'FAIL'}
          </span>
        )}
        {admet && (
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${
            admet.absorption?.hia_class === 'High' ? 'bg-green-500/20 text-green-300' 
              : admet.absorption?.hia_class === 'Moderate' ? 'bg-amber-500/20 text-amber-300'
              : 'bg-red-500/20 text-red-300'
          }`}>
            <Activity className="w-3 h-3" />
            HIA: {admet.absorption?.hia_class}
          </span>
        )}
        {admet && (
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${
            admet.distribution?.bbb_penetrant ? 'bg-blue-500/20 text-blue-300' : 'bg-slate-500/20 text-slate-300'
          }`}>
            <Brain className="w-3 h-3" />
            BBB: {admet.distribution?.bbb_penetrant ? 'Yes' : 'No'}
          </span>
        )}
        {alerts && (
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${
            alerts.clean ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
          }`}>
            <Shield className="w-3 h-3" />
            Alerts: {alerts.total_alerts || 0}
          </span>
        )}
        {admet?.toxicity && (
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${
            admet.toxicity.overall_risk === 'Low' ? 'bg-green-500/20 text-green-300'
              : admet.toxicity.overall_risk === 'Medium' ? 'bg-amber-500/20 text-amber-300'
              : 'bg-red-500/20 text-red-300'
          }`}>
            <AlertTriangle className="w-3 h-3" />
            Tox: {admet.toxicity.overall_risk}
          </span>
        )}
        {sa && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-purple-500/20 text-purple-300">
            <Dna className="w-3 h-3" />
            SA: {sa.sa_score?.toFixed(1)}/10
          </span>
        )}
      </div>
    )
  }
  
  // Render inline molecule cards within the message
  const renderInlineMolecules = () => {
    if (!message.inlineMolecules || message.inlineMolecules.length === 0) return null
    
    return (
      <div className="mt-3 space-y-3">
        {message.inlineMolecules.map((mol, i) => (
          <motion.div
            key={mol.smiles || i}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1 }}
            className="bg-slate-900/60 border border-slate-600/40 rounded-xl overflow-hidden"
          >
            <button
              onClick={() => toggleMolExpand(mol.smiles)}
              className="w-full flex items-center justify-between px-3 py-2 hover:bg-slate-800/50 transition-colors"
            >
              <div className="flex items-center gap-2 text-sm">
                <Atom className="w-4 h-4 text-emerald-400" />
                <span className="font-medium text-emerald-300">
                  {mol.name || mol.smiles}
                </span>
                {mol.gnn_prediction !== null && mol.gnn_prediction !== undefined && (
                  <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded-full">
                    GNN: {Number(mol.gnn_prediction).toFixed(4)}
                  </span>
                )}
              </div>
              <ChevronRight className={`w-4 h-4 text-slate-400 transition-transform ${expandedMols[mol.smiles] ? 'rotate-90' : ''}`} />
            </button>
            
            <AnimatePresence>
              {expandedMols[mol.smiles] && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-3 pb-3">
                    {mol.image_base64 && (
                      <div className="bg-white rounded-lg p-2 mb-2 flex justify-center">
                        <img 
                          src={mol.image_base64} 
                          alt={`${mol.name || mol.smiles} structure`} 
                          className="max-h-48 object-contain"
                        />
                      </div>
                    )}
                    {mol.svg && !mol.image_base64 && (
                      <div 
                        className="bg-white rounded-lg p-2 mb-2 flex justify-center [&>svg]:max-h-48"
                        dangerouslySetInnerHTML={{ __html: mol.svg }} 
                      />
                    )}
                    <div className="text-xs text-slate-400">
                      <code className="bg-slate-800 px-2 py-0.5 rounded text-slate-300 break-all">
                        {mol.smiles}
                      </code>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Always show collapsed image preview */}
            {!expandedMols[mol.smiles] && mol.image_base64 && (
              <div className="px-3 pb-2">
                <div className="bg-white rounded-lg p-1 flex justify-center cursor-pointer" onClick={() => toggleMolExpand(mol.smiles)}>
                  <img 
                    src={mol.image_base64} 
                    alt={`${mol.name || mol.smiles} structure`} 
                    className="max-h-24 object-contain opacity-90 hover:opacity-100 transition-opacity"
                  />
                </div>
              </div>
            )}
          </motion.div>
        ))}
      </div>
    )
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
    >
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser 
          ? 'bg-gradient-to-br from-blue-500 to-purple-600' 
          : 'bg-gradient-to-br from-emerald-500 to-teal-600'
      }`}>
        {isUser ? (
          <span className="text-white text-sm font-medium">U</span>
        ) : (
          <Sparkles className="w-4 h-4 text-white" />
        )}
      </div>
      
      {/* Message content */}
      <div className={`group relative max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div className={`rounded-2xl px-4 py-3 ${
          isUser 
            ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white' 
            : 'bg-slate-800/80 backdrop-blur-sm text-slate-100 border border-slate-700/50'
        }`}>
          {isUser ? (
            <p className="text-sm">{message.content}</p>
          ) : (
            <>
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
              {renderDrugDiscoveryBadges()}
              {renderInlineMolecules()}
              {renderFigures()}
            </>
          )}
          
          {message.smiles && (
            <div className="mt-2 pt-2 border-t border-white/20">
              <code className="text-xs bg-black/20 px-2 py-1 rounded">{message.smiles}</code>
            </div>
          )}
        </div>
        
        {/* Copy button */}
        {!isUser && (
          <button
            onClick={copyToClipboard}
            className="absolute -right-8 top-2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 hover:bg-slate-700 rounded"
          >
            {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-slate-400" />}
          </button>
        )}
        
        {/* Timestamp */}
        <div className={`text-xs text-slate-500 mt-1 ${isUser ? 'text-right' : ''}`}>
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </motion.div>
  )
}

// Timeline Sidebar
const TimelineSidebar = ({ isOpen, onToggle }) => {
  const { sessions, currentSessionId, loadSession, createSession, fetchSessions, messages } = useChatStore()
  
  useEffect(() => {
    fetchSessions()
  }, [])
  
  // Get questions from current session
  const currentQuestions = messages.filter(m => m.role === 'user')
  
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.aside
          initial={{ x: -300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: -300, opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="w-72 bg-slate-900/95 backdrop-blur-xl border-r border-slate-800 flex flex-col h-full"
        >
          {/* Header */}
          <div className="p-4 border-b border-slate-800">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                Research Sessions
              </h2>
              <button onClick={onToggle} className="p-1 hover:bg-slate-800 rounded">
                <ChevronLeft className="w-5 h-5 text-slate-400" />
              </button>
            </div>
            
            <button
              onClick={createSession}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white rounded-xl font-medium transition-all duration-200 shadow-lg shadow-blue-500/20"
            >
              <Plus className="w-4 h-4" />
              New Session
            </button>
          </div>
          
          {/* Current Session Questions */}
          {currentQuestions.length > 0 && (
            <div className="p-4 border-b border-slate-800">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                <Clock className="w-3 h-3" />
                This Session
              </h3>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {currentQuestions.map((q, i) => (
                  <motion.button
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:bg-slate-800/50 rounded-lg transition-colors truncate flex items-center gap-2"
                    onClick={() => {
                      const element = document.getElementById(`message-${q.id}`)
                      element?.scrollIntoView({ behavior: 'smooth' })
                    }}
                  >
                    <MessageSquare className="w-3 h-3 flex-shrink-0 text-slate-500" />
                    <span className="truncate">{q.content.slice(0, 40)}...</span>
                  </motion.button>
                ))}
              </div>
            </div>
          )}
          
          {/* Previous Sessions */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Previous Sessions
            </h3>
            <div className="space-y-2">
              {sessions.map((session) => (
                <button
                  key={session.session_id}
                  onClick={() => loadSession(session.session_id)}
                  className={`w-full text-left p-3 rounded-xl transition-all duration-200 ${
                    session.session_id === currentSessionId
                      ? 'bg-blue-600/20 border border-blue-500/30'
                      : 'hover:bg-slate-800/50 border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <FlaskConical className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-medium text-white truncate">
                      {session.title}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500">
                    {session.message_count} messages · {new Date(session.created_at).toLocaleDateString()}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}

// Molecule Visualization Panel
const MoleculePanel = () => {
  const { moleculeSvg, moleculeProperties, currentMolecule, drugDiscovery } = useChatStore()
  
  if (!moleculeSvg && !currentMolecule) return null
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-slate-800/80 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden"
    >
      <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <Atom className="w-5 h-5 text-emerald-400" />
          Molecule Analysis
        </h3>
      </div>
      
      {moleculeSvg && (
        <div className="p-4 bg-white rounded-lg m-4" dangerouslySetInnerHTML={{ __html: moleculeSvg }} />
      )}
      
      {moleculeProperties && (
        <div className="p-4 grid grid-cols-2 gap-3">
          <div className="bg-slate-900/50 rounded-xl p-3">
            <div className="text-xs text-slate-400">Molecular Weight</div>
            <div className="text-lg font-semibold text-white">{moleculeProperties.molecular_weight}</div>
          </div>
          <div className="bg-slate-900/50 rounded-xl p-3">
            <div className="text-xs text-slate-400">LogP</div>
            <div className="text-lg font-semibold text-white">{moleculeProperties.logp}</div>
          </div>
          <div className="bg-slate-900/50 rounded-xl p-3">
            <div className="text-xs text-slate-400">Formula</div>
            <div className="text-lg font-semibold text-white">{moleculeProperties.formula}</div>
          </div>
          <div className="bg-slate-900/50 rounded-xl p-3">
            <div className="text-xs text-slate-400">Drug-like</div>
            <div className={`text-lg font-semibold ${moleculeProperties.drug_like ? 'text-green-400' : 'text-red-400'}`}>
              {moleculeProperties.drug_like ? 'Yes' : 'No'}
            </div>
          </div>
        </div>
      )}

      {/* ADMET Quick Summary from drug discovery */}
      {drugDiscovery?.admet && (
        <div className="px-4 pb-4">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-blue-400" />
            ADMET Summary
          </h4>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-slate-900/50 rounded-lg p-2">
              <div className="text-xs text-slate-500">Absorption</div>
              <div className={`text-sm font-bold ${
                drugDiscovery.admet.absorption?.hia_class === 'High' ? 'text-green-400' : 'text-amber-400'
              }`}>{drugDiscovery.admet.absorption?.hia_class}</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-2">
              <div className="text-xs text-slate-500">BBB</div>
              <div className={`text-sm font-bold ${
                drugDiscovery.admet.distribution?.bbb_penetrant ? 'text-blue-400' : 'text-slate-400'
              }`}>{drugDiscovery.admet.distribution?.bbb_penetrant ? 'Penetrant' : 'Non-penetrant'}</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-2">
              <div className="text-xs text-slate-500">Metabolism</div>
              <div className="text-sm font-bold text-purple-400">{drugDiscovery.admet.metabolism?.metabolic_stability}</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-2">
              <div className="text-xs text-slate-500">Toxicity Risk</div>
              <div className={`text-sm font-bold ${
                drugDiscovery.admet.toxicity?.overall_risk === 'Low' ? 'text-green-400' 
                  : drugDiscovery.admet.toxicity?.overall_risk === 'Medium' ? 'text-amber-400' : 'text-red-400'
              }`}>{drugDiscovery.admet.toxicity?.overall_risk}</div>
            </div>
          </div>
        </div>
      )}

      {/* Drug-likeness quick info */}
      {drugDiscovery?.drug_likeness && (
        <div className="px-4 pb-4">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
            <Pill className="w-3.5 h-3.5 text-purple-400" />
            Drug-Likeness
          </h4>
          <div className="grid grid-cols-3 gap-2 text-center">
            {['lipinski', 'veber', 'ghose', 'egan', 'muegge'].map((filter) => {
              const data = drugDiscovery.drug_likeness[filter]
              if (!data) return null
              return (
                <div key={filter} className={`rounded-lg p-2 ${data.pass ? 'bg-green-500/10 border border-green-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
                  <div className="text-xs text-slate-400 capitalize">{filter}</div>
                  <div className={`text-xs font-bold ${data.pass ? 'text-green-400' : 'text-red-400'}`}>
                    {data.pass ? 'PASS' : 'FAIL'}
                  </div>
                </div>
              )
            })}
            <div className="rounded-lg p-2 bg-purple-500/10 border border-purple-500/20">
              <div className="text-xs text-slate-400">QED</div>
              <div className="text-xs font-bold text-purple-400">
                {drugDiscovery.drug_likeness.qed_score?.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Synthetic Accessibility */}
      {drugDiscovery?.synthetic_accessibility && (
        <div className="px-4 pb-4">
          <div className="bg-slate-900/50 rounded-lg p-3 flex items-center justify-between">
            <div>
              <div className="text-xs text-slate-400">Synthetic Accessibility</div>
              <div className="text-sm font-bold text-purple-300">{drugDiscovery.synthetic_accessibility.category}</div>
            </div>
            <div className="text-2xl font-bold text-purple-400">
              {drugDiscovery.synthetic_accessibility.sa_score?.toFixed(1)}
              <span className="text-xs text-slate-500">/10</span>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  )
}

// Drug Discovery Lead Optimization Panel
const LeadOptPanel = () => {
  const { drugDiscovery } = useChatStore()
  const opt = drugDiscovery?.lead_optimization
  if (!opt) return null

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-slate-800/80 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden"
    >
      <div className="p-4 border-b border-slate-700/50">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-cyan-400" />
          Lead Optimization
        </h3>
      </div>

      {/* Priority Actions */}
      {opt.priority_actions?.length > 0 && (
        <div className="p-4 space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Priority Actions</h4>
          {opt.priority_actions.map((action, i) => (
            <div key={i} className={`rounded-lg p-3 border-l-4 ${
              action.severity === 'critical' ? 'border-red-500 bg-red-500/10'
                : action.severity === 'high' ? 'border-amber-500 bg-amber-500/10'
                : 'border-blue-500 bg-blue-500/10'
            }`}>
              <div className="flex items-center gap-2 text-sm font-bold text-white">
                <AlertTriangle className={`w-3.5 h-3.5 ${
                  action.severity === 'critical' ? 'text-red-400' : 'text-amber-400'
                }`} />
                {action.issue}
              </div>
              <div className="text-xs text-slate-400 mt-1">{action.value}</div>
              <div className="text-xs text-slate-300 mt-1">{action.action}</div>
            </div>
          ))}
        </div>
      )}

      {/* Bioisostere Suggestions */}
      {opt.bioisostere_opportunities?.length > 0 && (
        <div className="p-4 border-t border-slate-700/30 space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Bioisostere Replacements</h4>
          {opt.bioisostere_opportunities.map((bio, i) => (
            <div key={i} className="bg-slate-900/50 rounded-lg p-3">
              <div className="text-sm font-semibold text-emerald-300 capitalize mb-1">
                {bio.functional_group.replace(/_/g, ' ')}
              </div>
              <ul className="text-xs text-slate-400 space-y-0.5">
                {bio.replacements.map((r, j) => (
                  <li key={j} className="flex items-start gap-1.5">
                    <span className="text-emerald-500 mt-0.5">→</span>
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  )
}

// Sources Panel
const SourcesPanel = () => {
  const { currentSources } = useChatStore()
  
  if (!currentSources?.length) return null
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-slate-800/80 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden"
    >
      <div className="p-4 border-b border-slate-700/50">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-blue-400" />
          Sources ({currentSources.length})
        </h3>
      </div>
      
      <div className="p-4 space-y-3 max-h-64 overflow-y-auto">
        {currentSources.slice(0, 5).map((source, i) => (
          <div key={i} className="bg-slate-900/50 rounded-xl p-3">
            <div className="flex items-start justify-between gap-2">
              <span className="text-xs font-medium text-blue-400">[{i + 1}]</span>
              <span className="text-xs text-slate-500">{(source.score * 100).toFixed(0)}% match</span>
            </div>
            <div className="text-sm font-medium text-white mt-1">{source.title}</div>
            <p className="text-xs text-slate-400 mt-1 line-clamp-2">{source.content?.slice(0, 150)}...</p>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

// Document Upload Panel
const DocumentUpload = () => {
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const { documents, setDocuments, currentSessionId } = useChatStore()
  const fileInputRef = useRef(null)
  
  const handleUpload = async (files) => {
    if (!files?.length) return
    setUploading(true)
    
    try {
      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)
        if (currentSessionId) {
          formData.append('session_id', currentSessionId)
        }
        
        const response = await fetch('/api/documents/upload', {
          method: 'POST',
          body: formData
        })
        
        if (response.ok) {
          const data = await response.json()
          setDocuments([...documents, data])
        }
      }
    } catch (error) {
      console.error('Upload failed:', error)
    } finally {
      setUploading(false)
    }
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-slate-800/80 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden"
    >
      <div className="p-4 border-b border-slate-700/50">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-orange-400" />
          Research Documents
        </h3>
      </div>
      
      <div className="p-4">
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleUpload(e.dataTransfer.files) }}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
            dragOver 
              ? 'border-blue-500 bg-blue-500/10' 
              : 'border-slate-600 hover:border-slate-500'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.csv"
            onChange={(e) => handleUpload(e.target.files)}
            className="hidden"
          />
          
          {uploading ? (
            <Loader2 className="w-8 h-8 mx-auto text-blue-400 animate-spin" />
          ) : (
            <Upload className="w-8 h-8 mx-auto text-slate-400" />
          )}
          <p className="text-sm text-slate-400 mt-2">
            Drop research papers here or click to upload
          </p>
          <p className="text-xs text-slate-500 mt-1">PDF, TXT, MD, CSV</p>
        </div>
        
        {documents.length > 0 && (
          <div className="mt-4 space-y-2">
            {documents.map((doc, i) => (
              <div key={i} className="flex items-center justify-between bg-slate-900/50 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-slate-400" />
                  <span className="text-sm text-white truncate">{doc.filename}</span>
                </div>
                <span className="text-xs text-slate-500">{(doc.size / 1024).toFixed(1)} KB</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}

// Main Chat Interface
export default function ModernChatInterface() {
  const [question, setQuestion] = useState('')
  const [smiles, setSmiles] = useState('')
  const [showSmiles, setShowSmiles] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  
  const {
    messages, addMessage, isStreaming, setStreaming, streamingContent,
    appendStreamContent, sidebarOpen, toggleSidebar, currentSessionId,
    createSession, setCurrentMolecule, setCurrentSources, clearChat,
    setInlineMolecules, setDrugDiscovery, setFigures, drugDiscovery, figures
  } = useChatStore()
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim() || isStreaming) return
    
    // Ensure we have a session
    let sessionId = currentSessionId
    if (!sessionId) {
      sessionId = await createSession()
    }
    
    // Add user message
    addMessage({ role: 'user', content: question, smiles: smiles || null })
    setStreaming(true, 'Thinking...')
    
    const userQuestion = question
    const userSmiles = smiles
    setQuestion('')
    setSmiles('')
    
    try {
      // Use simple non-streaming endpoint for speed
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userQuestion,
          smiles: userSmiles || null,
          session_id: sessionId,
          num_results: 3,
          temperature: 0.3,
          reasoning_mode: 'draft'
        })
      })
      
      const data = await response.json()
      
      if (data.answer) {
        addMessage({ 
          role: 'assistant', 
          content: data.answer,
          inlineMolecules: data.inline_molecules || [],
          drugDiscovery: data.drug_discovery || null,
          figures: data.figures || {},
        })
        if (data.sources) {
          setCurrentSources(data.sources)
        }
        if (data.inline_molecules && data.inline_molecules.length > 0) {
          setInlineMolecules(data.inline_molecules)
        }
        // Store drug discovery data and figures in global state
        if (data.drug_discovery && Object.keys(data.drug_discovery).length > 0) {
          setDrugDiscovery(data.drug_discovery)
        }
        if (data.figures && Object.keys(data.figures).length > 0) {
          setFigures(data.figures)
        }
        // If molecules were auto-extracted, show the primary one in the side panel
        if (data.molecule_info) {
          setCurrentMolecule(
            data.molecule_info,
            data.inline_molecules?.[0]?.svg || null,
            null // properties loaded separately
          )
        } else if (data.auto_extracted_smiles && data.inline_molecules?.length > 0) {
          setCurrentMolecule(
            { smiles: data.auto_extracted_smiles, predicted_property: data.inline_molecules[0]?.gnn_prediction },
            data.inline_molecules[0]?.svg || null,
            null
          )
        }
      } else if (data.detail) {
        addMessage({ role: 'assistant', content: `Error: ${data.detail}` })
      }
      
      setStreaming(false)
    } catch (error) {
      console.error('Error:', error)
      addMessage({ role: 'assistant', content: `Error: ${error.message}` })
      setStreaming(false)
    }
  }
  
  const examples = [
    { q: "What is the mechanism of action of aspirin?", s: "CC(=O)Oc1ccccc1C(=O)O" },
    { q: "Explain EGFR inhibitors in cancer treatment", s: "" },
    { q: "Compare solubility of ethanol vs methanol", s: "CCO" },
  ]
  
  return (
    <div className="h-screen flex overflow-hidden bg-slate-950">
      <GradientBackground />
      
      {/* Timeline Sidebar */}
      <TimelineSidebar isOpen={sidebarOpen} onToggle={toggleSidebar} />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-xl">
          <div className="flex items-center gap-4">
            {!sidebarOpen && (
              <button onClick={toggleSidebar} className="p-2 hover:bg-slate-800 rounded-lg">
                <Menu className="w-5 h-5 text-slate-400" />
              </button>
            )}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                <Beaker className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">Helios</h1>
                <p className="text-xs text-slate-400">Chemistry AI Research Assistant</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 px-3 py-1.5 bg-emerald-500/10 rounded-full">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
              <span className="text-xs text-emerald-400">Online</span>
            </div>
            <button 
              onClick={clearChat}
              className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </header>
        
        {/* Chat Area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6">
            {messages.length === 0 && !isStreaming && (
              <div className="h-full flex flex-col items-center justify-center">
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center max-w-lg"
                >
                  <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-2xl shadow-blue-500/30">
                    <Zap className="w-10 h-10 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold text-white mb-2">
                    Welcome to Helios
                  </h2>
                  <p className="text-slate-400 mb-8">
                    Your AI-powered chemistry research assistant. Ask questions, analyze molecules, 
                    and explore chemical literature with advanced reasoning.
                  </p>
                  
                  <div className="grid gap-3">
                    {examples.map((ex, i) => (
                      <button
                        key={i}
                        onClick={() => { setQuestion(ex.q); setSmiles(ex.s); inputRef.current?.focus() }}
                        className="text-left p-4 bg-slate-800/50 hover:bg-slate-800 rounded-xl border border-slate-700/50 hover:border-slate-600 transition-all group"
                      >
                        <div className="flex items-center gap-2 text-white font-medium">
                          <Search className="w-4 h-4 text-blue-400" />
                          {ex.q}
                        </div>
                        {ex.s && (
                          <code className="text-xs text-slate-500 mt-1 block">SMILES: {ex.s}</code>
                        )}
                      </button>
                    ))}
                  </div>
                </motion.div>
              </div>
            )}
            
            <div className="space-y-6 max-w-3xl mx-auto">
              {messages.map((message, index) => (
                <div key={message.id} id={`message-${message.id}`}>
                  <Message message={message} index={index} />
                </div>
              ))}
              
              {/* Streaming content */}
              {isStreaming && streamingContent && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3"
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <div className="max-w-[80%] bg-slate-800/80 backdrop-blur-sm rounded-2xl px-4 py-3 border border-slate-700/50">
                    <div className="prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown>{streamingContent}</ReactMarkdown>
                    </div>
                    <TypingIndicator />
                  </div>
                </motion.div>
              )}
              
              {isStreaming && !streamingContent && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <div className="bg-slate-800/80 backdrop-blur-sm rounded-2xl px-4 py-3 border border-slate-700/50">
                    <TypingIndicator />
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </div>
          
          {/* Right Panel */}
          <div className="w-80 border-l border-slate-800/50 bg-slate-900/30 overflow-y-auto p-4 space-y-4 hidden xl:block">
            <MoleculePanel />
            <LeadOptPanel />
            <SourcesPanel />
            <DocumentUpload />
          </div>
        </div>
        
        {/* Input Area */}
        <div className="border-t border-slate-800/50 bg-slate-900/50 backdrop-blur-xl p-4">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="relative">
              <textarea
                ref={inputRef}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSubmit(e)
                  }
                }}
                placeholder="Ask a chemistry question..."
                rows={1}
                disabled={isStreaming}
                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-2xl pl-4 pr-32 py-4 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent resize-none"
              />
              
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowSmiles(!showSmiles)}
                  className={`p-2 rounded-lg transition-colors ${showSmiles ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
                >
                  <Atom className="w-5 h-5" />
                </button>
                
                <button
                  type="submit"
                  disabled={!question.trim() || isStreaming}
                  className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-white transition-all shadow-lg shadow-blue-500/20"
                >
                  {isStreaming ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>
            
            {/* SMILES Input */}
            <AnimatePresence>
              {showSmiles && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-2"
                >
                  <input
                    type="text"
                    value={smiles}
                    onChange={(e) => setSmiles(e.target.value)}
                    placeholder="Enter molecule SMILES (e.g., CCO for ethanol)"
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-2.5 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500/50 text-sm"
                  />
                </motion.div>
              )}
            </AnimatePresence>
            
            <div className="flex items-center justify-between mt-2 text-xs text-slate-500">
              <span>Press Enter to send, Shift+Enter for new line</span>
              <span>Powered by Graphormer + RAG + TinyLlama</span>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
