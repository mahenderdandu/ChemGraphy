export default function MoleculeViewer({ molecule }) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
        <span className="text-2xl mr-2">🧪</span>
        Molecule Analysis
      </h3>
      <div className="space-y-3">
        <div>
          <p className="text-xs text-gray-500 uppercase font-semibold mb-1">SMILES</p>
          <p className="text-sm font-mono bg-gray-50 px-3 py-2 rounded break-all">
            {molecule.smiles}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Predicted Property</p>
          <p className="text-2xl font-bold text-blue-600">
            {molecule.predicted_property?.toFixed(4)}
          </p>
        </div>
        {molecule.embedding_shape && (
          <div>
            <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Graph Embedding</p>
            <p className="text-sm text-gray-700">
              Shape: [{molecule.embedding_shape.join(', ')}]
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
