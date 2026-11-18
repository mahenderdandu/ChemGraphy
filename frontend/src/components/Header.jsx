export default function Header() {
  return (
    <header className="bg-white shadow-md border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="text-4xl">🔬</div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">ChemGraphy</h1>
              <p className="text-sm text-gray-600">Chemistry AI Assistant</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
              Online
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}
