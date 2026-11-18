export default function SourcesList({ sources }) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-4">
      <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
        <span className="text-2xl mr-2">📚</span>
        Retrieved Sources
      </h3>
      <div className="space-y-4">
        {sources.map((source, idx) => (
          <div key={idx} className="border-l-4 border-blue-500 pl-4 py-2">
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs font-semibold text-gray-500">SOURCE {idx + 1}</p>
              <p className="text-xs font-mono text-blue-600">
                Score: {source.score?.toFixed(4)}
              </p>
            </div>
            <p className="text-sm font-semibold text-gray-900 mb-1">
              {source.title}
            </p>
            <p className="text-xs text-gray-600 line-clamp-3">
              {source.content}
            </p>
            {source.embedding_shape && (
              <p className="text-xs text-gray-400 mt-1">
                Embedding: [{source.embedding_shape.join(', ')}]
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
