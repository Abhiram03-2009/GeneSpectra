import React from 'react';
import { FiCheckCircle, FiAlertTriangle, FiMaximize } from 'react-icons/fi';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Label } from 'recharts';
import { motion } from 'framer-motion';
import { ClipLoader } from 'react-spinners';
import ActivationProfileChart from './ActivationProfileChart';

interface PredictionData {
  label: string;
  confidence: number;
}

interface Visualizations {
  panel_a: string;
  panel_b: string;
  panel_c: string;
  confusion_matrix: string;
  letter_encoded_output: string;
  base_specific_activations: string;
}

interface ChartData {
  activationProfile: any[];
  baseActivations: any;
}

interface Summary {
  gene: string;
  sequenceLength: string;
  keyFindings: string[];
  performanceMetrics: Record<string, string>;
  clinicalSignificance: string[];
}

interface PredictionCardProps {
  prediction: PredictionData;
  chartData: ChartData;
  summary: Summary;
  onViewVisuals: () => void;
  confusionMatrixImage: string;
  letterEncodedOutputImage: string;
  panelAImage: string;
  panelBImage: string;
  panelCImage: string;
  baseSpecificActivationsImage: string;
  isViewVisualsDisabled: boolean;
  letterEncodedOutputText: string;
}

const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  chartData,
  summary,
  onViewVisuals,
  confusionMatrixImage,
  letterEncodedOutputImage,
  panelAImage,
  panelBImage,
  panelCImage,
  baseSpecificActivationsImage,
  isViewVisualsDisabled,
  letterEncodedOutputText,
}) => {
  const isPathogenic = prediction.label === 'Pathogenic';

  // Use actual chartData for rendering
  const chartDataToRender = chartData; 


  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -50 }}
      transition={{ duration: 0.5 }}
      className="hud-card relative overflow-hidden rounded-3xl border border-cyan-600/40 backdrop-blur-lg bg-gradient-to-br from-[#0a1e33]/90 to-[#142b43]/90 shadow-2xl p-8 hover:border-cyan-300 transition-all duration-300 group w-full"
    >
      <h2 className="text-4xl font-extrabold text-white mb-8 text-center tracking-tight">Analysis Results</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-8 overflow-y-auto overflow-x-hidden max-h-[calc(100vh-200px)]" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
        {/* Prediction, Key Findings & Performance Metrics Consolidated */}
        <div className="bg-gray-800 p-6 rounded-lg border border-light blue-700 shadow-inner flex flex-col items-center justify-center">
          <h3 className="text-2xl font-bold text-white mb-4 tracking-tight text-center">Prediction</h3>
          <div className={`flex items-center gap-3 text-5xl font-extrabold mb-4 ${isPathogenic ? 'text-red-500' : 'text-green-500'}`}>
            {isPathogenic ? <FiAlertTriangle /> : <FiCheckCircle />}
            {prediction.label}
          </div>
          <div className="text-2xl text-gray-300 mb-6">
            Confidence: {(prediction.confidence * 100).toFixed(2)}%
          </div>

          <h4 className="text-cyan-400 font-bold mb-3 text-lg mt-8">Key Findings</h4>
          <ul className="text-gray-300 mx-auto max-h-36 overflow-y-auto space-y-1">
            {summary.keyFindings.map((item, idx) => (
              <li key={idx} className="text-sm text-center">
                 {item}
              </li>
            ))}
          </ul>

          <h4 className="text-cyan-400 font-bold mt-6 mb-3 text-lg">Performance Metrics</h4>
          <ul className="text-gray-300 mx-auto max-h-36 overflow-y-auto space-y-1">
            {Object.entries(summary.performanceMetrics).map(([key, value], idx) => (
              <li key={idx} className="text-sm text-center">
                 {key}: {value}
              </li>
            ))}
          </ul>
        </div>

        {/* Activation Profile Chart */}
        <div className="hud-card p-4 flex flex-col items-center border border-gray-700 rounded-lg w-full">
          
          <ActivationProfileChart chartData={chartData} />
        </div>

        {/* Panel A */}
        <div className="hud-card p-4 flex flex-col items-center border border-gray-700 rounded-lg w-full">
          <h5 className="text-[20px] font-semibold text-cyan-300 mb-3 text-center">Original 2D DNA Sequence 4xN Binary Matrix</h5>
          <img src={`data:image/png;base64,${panelAImage}`} alt="Panel A" className="max-w-full h-auto rounded-md shadow-md block mx-auto" />
        </div>

        {/* Panel B */}
        <div className="hud-card p-4 flex flex-col items-center border border-gray-700 rounded-lg w-full">
          <h5 className="text-[20px] font-semibold text-cyan-300 mb-3 text-center">Grad-CAM Layer Sequence Regions Activation Heatmap</h5>
          <img src={`data:image/png;base64,${panelBImage}`} alt="Panel B" className="max-w-full h-auto rounded-md shadow-md block mx-auto" />
        </div>

        {/* Panel C */}
        <div className="hud-card p-4 flex flex-col items-center border border-gray-700 rounded-lg w-full">
          <h5 className="text-[20px] font-semibold text-cyan-300 mb-3 text-center">2D DNA Sequence + Grad-CAM Activation Heatmap Overlay Matrix</h5>
          <img src={`data:image/png;base64,${panelCImage}`} alt="Panel C" className="max-w-full h-auto rounded-md shadow-md block mx-auto" />
        </div>

        {/* Confusion Matrix */}
        <div className="hud-card p-4 flex flex-col items-center border border-gray-700 rounded-lg w-full">
          <h5 className="text-[20px] font-semibold text-cyan-300 mb-3 text-center">Confusion Matrix (True vs Predicted)</h5>
          <img src={`data:image/png;base64,${confusionMatrixImage}`} alt="Confusion Matrix" className="max-w-full h-auto rounded-md shadow-md block mx-auto ml-[-0.5rem]" />
        </div>

        {/* Letter Encoded Output */}
        <div className="hud-card p-4 flex flex-col items-center border border-gray-700 rounded-lg w-full">
          <h5 className="text-[20px] font-semibold text-cyan-300 mb-3 text-center">Letter Encoded Output (ATCG Sequential Repetition)</h5>
          <div className="overflow-auto p-2 rounded bg-gray-900 font-mono text-gray-200 mx-auto max-w-md" style={{ maxHeight: '400px' }}>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: letterEncodedOutputText.length > 200 ? '0.75rem' : (letterEncodedOutputText.length > 100 ? '1rem' : '1.25rem'), textAlign: 'center' }}>
              {letterEncodedOutputText.split('').map((char, idx) => {
                let color = '#808080'; // Default for N or unknown
                switch (char.toUpperCase()) {
                  case 'A':
                    color = '#ff4d4d';
                    break;
                  case 'T':
                    color = '#4dff4d';
                    break;
                  case 'C':
                    color = '#4d4dff';
                    break;
                  case 'G':
                    color = '#ffff4d';
                    break;
                }
                return <span key={idx} style={{ color }}>{char}</span>;
              })}
            </pre>
          </div>
        </div>

        {/* Base Specific Activations */}
        <div className="hud-card p-4 flex flex-col items-center border border-gray-700 rounded-lg w-full">
          <h5 className="text-[20px] font-semibold text-cyan-300 mb-3 text-center">Base Specific Activations (Box Plots)</h5>
          <img src={`data:image/png;base64,${baseSpecificActivationsImage}`} alt="Base Specific Activations" className="max-w-full h-auto rounded-md shadow-md block mx-auto" />
        </div>
      </div>

      
    </motion.div>
  );
};

export default PredictionCard;